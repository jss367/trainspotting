"""Which sampled training examples a model's behaviour is sensitive to, by Bayesian influence.

Every other layer counts. `grep` says how many rows of a mix hold a phrase,
`influence` ranks the stages by that rate, and its docstring says what the rate
cannot do: weight the stages against each other, because a count does not
measure how much an example moved the model. This layer measures that, for the
examples the other layers already found.

The estimator is the Bayesian influence function of Lau, Wang, Baker, Murfet and
Hoogland (2025). Around the released weights w* there is a local posterior,

    p(w) ∝ exp(-nβ · L(w) - γ/2 · ‖w - w*‖²)

where L is the mean loss over the data used to localize it. Upweighting one
example z_j in that data by ε moves the posterior, and the derivative of the
expected loss on a query z_q with respect to ε is -nβ · Cov(ℓ_q, ℓ_j) under the
posterior. So the number reported per example is the posterior covariance of its
loss with the query's loss, sampled by SGLD: a positive covariance means that
training harder on the example would lower the loss the model assigns to the
query, which is to say the example pulls the model *toward* saying it. No Hessian
is formed or inverted, which is what lets this run at model scale.

Two things bound what the number means. First, the posterior is localized on the
examples this layer is given — the committed context records and corpus samples,
a few hundred rows — not on the training set, and the covariances are about that
sample. Second, the checkpoint is the one named in the result, and the influence
is of an example on *that* model's loss; an early Pythia checkpoint and the final
one can disagree about the same document. Both are recorded in every result file
rather than assumed.

The per-example loss is the mean token loss over the text the example fits the
model to, with the rest of the example as context and masked out of the loss:
the assistant turns of an SFT example, the chosen completion of a DPO pair, the
whole of a corpus document. A think model's response is its reasoning and its
answer together, in the `<think>` markers it was trained with, so the reasoning
the context record stores beside the answer is put back before scoring. A DPO
pair yields two candidates, its chosen and its rejected side, because the two
carry opposite signs in training: the objective pushes the model off the
rejected text. The turns the two sides share before they branch are the
conversation the pair was judged in, not either completion, so an assistant turn
in that shared history is context on both sides rather than target. The
posterior is localized on the text the model was fit *toward* — documents, SFT
responses, chosen completions — and the rejected completions are scored at every
draw but never put in a minibatch, since fitting the chain to them would localize
it on the opposite of what training did. Their covariance with the query is
still the number of interest:
a positive covariance on a rejected completion is evidence that the pair taught
the model *away* from the query, where the same number on the chosen side is
evidence it taught the model toward it. This is a reading of the loss
covariance, not the influence function of the pairwise DPO objective itself,
which would need the reference model's log-ratio as well. An RL row stores no
response and is skipped, and the skip is counted.

The sampler is plain SGLD: w ← w - (ε/2)(nβ ∇L̂(w) + γ(w - w*)) + N(0, ε). Every
chain starts at w*, discards a burn-in, then records the loss of the query and of
every candidate at each retained draw. The covariance is taken within a chain and
averaged across chains, and the spread across chains is the reported uncertainty
— a chain that wandered somewhere the others did not shows up as a wide interval
rather than as a confident number. The local learning coefficient nβ(E[L] - L(w*))
comes out of the same draws and is printed as the sanity check on the step size:
if it is negative or enormous the chain is not sampling the posterior it was
asked to.
"""

import json
import math
import random
import re
import sys

from . import context, paths, registry, search

# The turn roles whose text is a training target. `text` is a corpus document,
# which is all target. Anything else — user, system, tool — is context the loss
# is masked over. A turn can say otherwise with an explicit `fit` flag, which is
# how an assistant turn in a DPO pair's shared history is kept as context; the
# role is the default for turns built without one.
FIT_ROLES = {"assistant", "text"}

# How a think model's response was written in its training data: the reasoning
# inside `<think>` markers, then the answer. `context._split_think` strips the
# markers and the whitespace around them when it stores the two halves, and the
# rows vary in that whitespace (zero to two newlines after `</think>`), so the
# round trip is faithful to the markers and takes the most common spacing — the
# form of every sampled Dolci-Think-DPO turn and the plurality of the SFT ones.
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>"
# `<think>` and `</think>` themselves, the fixed part of what the markers cost.
THINK_MARKERS = len("<think>") + len("</think>")

SNIPPET = 120  # characters of the fit text shown per ranked candidate

# Turn fields the context record keeps beside `content`, or notes it dropped,
# that the model was trained on in a template-specific form this layer cannot
# rebuild: the tool menu a system turn offered, an assistant turn's function
# call, a refusal. A record carrying any of these is not the conversation the
# model saw, so it is skipped and counted rather than scored as if it were.
STRUCTURED = ("omitted", *search.STRUCTURED_TURN_FIELDS, *search.INPUT_TURN_FIELDS)


# --- Candidates -----------------------------------------------------------------


def stale(data: dict, dataset: str | None) -> str | None:
    """Why a stored sample cannot stand for `dataset`, or None.

    The two questions `cli._stale_context` asks before `stance` spends a judge
    on stored examples: are they from this dataset, and were they drawn from
    one tree. A stage repointed at another mix keeps its filename, and a draw
    that straddled a republish records `revision_moved_to`; either way the file
    is not a sample of the stage it is named for, and a model run over it would
    attribute its examples to that stage.
    """
    if dataset and data.get("dataset") and data["dataset"] != dataset:
        return f"the stored examples are from {data['dataset']} but this stage names {dataset}"
    if data.get("revision_moved_to"):
        return "the stored examples straddled a republish while they were drawn"
    return None


def incomplete(turns, shared: int = 0) -> bool:
    """Whether a stored record is not the example the model was trained on.

    Two ways: a turn carries or notes fields this layer cannot rebuild
    (STRUCTURED), or any turn was cut at the context record's 4,000-character
    field limit. A cut fit turn is not a prefix of the training example the way
    a prefix would be: the reasoning of a think turn is closed early and the
    answer appended after it. A cut context turn is no better: the record keeps
    the *first* 4,000 characters of a prompt, and `encode` keeps the *last*
    tokens of the example, so what would sit before the response is the middle
    of the prompt rather than the text that actually preceded it. A 6,551-
    character user turn in the committed Instruct SFT sample is the case.

    `shared` is the DPO branch point; it is accepted for the caller's
    convenience and no longer changes the answer, since a cut anywhere counts.
    """
    if any(t.get(k) for t in turns for k in STRUCTURED):
        return True
    for t in turns:
        fields = [t] + ([t["reasoning"]] if t.get("reasoning") else [])
        if any((f.get("chars") or 0) > len(f.get("text") or "") for f in fields):
            return True
        # A recorded length the markers and whitespace cannot be fitted to.
        if t.get("chars_raw") is not None and len(think_form(t)) != t["chars_raw"]:
            return True
    return False


def _context_candidates(target_name: str, stage: dict) -> tuple[list[dict], str | None, int]:
    """The committed context records of one post-training stage as candidates.

    Returns (candidates, skip reason, records skipped as incomplete). A stage
    with no context file is a skip with its reason, as is one whose kind stores
    no response at all or whose file is a sample of some other mix; all are
    reported, since a stage silently absent from the ranking reads as a stage
    with nothing to say. A record with tool use is skipped on its own and
    counted: the context record does not hold it as the model was trained on
    it (see STRUCTURED).
    """
    kind = registry.stage_kind(stage)
    name = stage["stage"]
    if kind == "rlvr":
        return [], "an RL row stores no response, so there is no text the model was fit to", 0
    if kind == "chat":
        return [], "a conversation log was not trained on, so nothing in it was fit", 0
    path = paths.find(f"{target_name}.{name}.context.json")
    if path is None:
        return [], f"no committed context records — run `trainspotting context {target_name} --stage {name}`", 0
    data = json.loads(path.read_text())
    why = stale(data, stage.get("hf_dataset"))
    if why:
        return [], why, 0
    out = []
    skipped = 0
    for ordinal, rec in enumerate(data.get("records", [])):
        sides = [rec.get("turns") or []] if kind == "sft" else [
            (rec.get(side) or {}).get("turns") or [] for side in ("chosen", "rejected")
        ]
        shared_turns = context.branch_point(*sides) if kind == "dpo" else 0
        if any(incomplete(t, shared_turns) for t in sides):
            skipped += 1
            continue
        base = {
            "stage": name,
            "kind": kind,
            "rec": ordinal,
            "id": rec.get("id"),
            "row": rec.get("row"),
            "source": _source(rec.get("meta") or {}, stage),
        }
        if kind == "sft":
            turns = rec.get("turns") or []
            out.append({**base, "side": "response", "turns": _turns(turns), "cut": _cut(turns)})
        elif kind == "dpo":
            # The two sides share every turn before they branch, so the branch
            # is found once per pair and both sides are cut at it: what comes
            # before is the conversation the pair was judged in and is context
            # on both, assistant turns included.
            chosen = (rec.get("chosen") or {}).get("turns") or []
            rejected = (rec.get("rejected") or {}).get("turns") or []
            shared = context.branch_point(chosen, rejected)
            for side, turns in (("chosen", chosen), ("rejected", rejected)):
                out.append({**base, "side": side, "turns": _turns(turns, shared), "cut": _cut(turns)})
    return out, None, skipped


def _docs_candidates(target_name: str, stage: dict) -> tuple[list[dict], str | None, int]:
    """The committed document sample of one pretraining corpus as candidates,
    with the count of documents skipped.

    A document longer than the sample's budget is stored as `extract.excerpt`
    leaves it: three spans from across the document joined with an elision
    marker, so the classifier judges the whole document rather than its opening.
    That is not a prefix of anything the model saw — the joins and the marker
    are text no training sequence held — so such a document is skipped and
    counted rather than scored. On the committed Pythia sample that is 22 of
    300; on the OLMo long-context sample, 128.
    """
    name = stage["stage"]
    path = paths.find(f"{target_name}.{name}.docs.json")
    if path is None:
        return [], f"no committed document sample — run `trainspotting pretrain {target_name} --stage {name}`", 0
    data = json.loads(path.read_text())
    why = stale(data, stage.get("sample_dataset"))
    if why:
        return [], why, 0
    out = []
    skipped = 0
    for ordinal, doc in enumerate(data.get("records", [])):
        text = doc.get("text") or ""
        if (doc.get("chars") or len(text)) > len(text):
            skipped += 1
            continue
        out.append({
            "stage": name,
            "kind": "pretrain",
            "rec": ordinal,
            "id": doc.get("id"),
            "row": doc.get("row"),
            "source": doc.get("source") or None,
            "side": "document",
            "turns": [{"role": "text", "text": text}],
            "cut": False,
        })
    return out, None, skipped


def _source(meta: dict, stage: dict) -> str | None:
    for col in stage.get("source_columns") or []:
        if meta.get(col):
            return str(meta[col])
    return None


def _turns(turns, shared: int = 0) -> list[dict]:
    """The stored turns as candidate turns: role, the text the model read, and
    whether the loss is taken over it.

    The context record stores a think model's reasoning beside its answer, with
    the `<think>` markers stripped. The model was fit to the whole response, and
    on the think mixes the reasoning is most of it, so the two are joined back
    in the trained form. The chat template of a think model ends its assistant
    header with `<think>`, so a response that lacks the marker also breaks the
    cumulative rendering in `pieces` and sends the example to the role fallback.

    `shared` is how many leading turns are shared history — the turns before a
    DPO pair branches — and an assistant turn among them is not fit.

    An *empty* thinking span leaves no reasoning field at all: `context._turns`
    strips the markers and stores nothing, and the only trace is that the turn
    as written (`chars_raw`) was longer than the answer (`chars`). The committed
    32B-think sample has such a turn — 192 characters written, 173 of answer. The
    model was still fit to the markers, and the think template's cumulative
    rendering still expects them, so the empty block is put back too.
    """
    out = []
    for i, t in enumerate(turns):
        role = t.get("role") or "user"
        text = think_form(t)
        out.append({"role": role, "text": text, "fit": role in FIT_ROLES and i >= shared})
    return out


def think_form(t: dict) -> str:
    """The turn's text as written, with its thinking span and markers put back.

    `context._turns` strips `<think>`, `</think>` and the whitespace around them
    and records only how long the turn was as written (`chars_raw`). The
    markers are fixed; the whitespace is not — most Think SFT turns cost 18
    characters over their two fields, one newline each side of the reasoning
    and one after the closing marker, while an empty span costs 19 — so the
    newlines after `</think>` are however many the recorded length calls for.
    A turn whose recorded length cannot be met that way is left as its answer;
    `incomplete` refuses it before it is scored.
    """
    text = t.get("text") or ""
    reasoning = (t.get("reasoning") or {}).get("text")
    raw = t.get("chars_raw")
    if raw is None:
        return text
    span = t.get("chars") or len(text)
    inner = (t.get("reasoning") or {}).get("chars") or len(reasoning or "")
    trailing = raw - span - inner - THINK_MARKERS - len(THINK_OPEN) + len("<think>") - 1
    if trailing < 0:
        return text
    return f"{THINK_OPEN}{reasoning or ''}{THINK_CLOSE}{chr(10) * trailing}{text}"


def _cut(turns) -> bool:
    """Whether the context record shortened any field of any turn (it stores
    4,000 characters a field, and a think turn's reasoning is its own field)."""
    fields = []
    for t in turns:
        fields.append(t)
        if t.get("reasoning"):
            fields.append(t["reasoning"])
    return any((f.get("chars") or 0) > len(f.get("text") or "") for f in fields)


def fits(turn: dict) -> bool:
    """Whether a turn's text is a loss target: its `fit` flag, or its role."""
    return turn.get("fit", turn["role"] in FIT_ROLES)


def candidates(
    target_name: str,
    target: dict,
    stages: list[str] | None = None,
    match: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    incomplete: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Every candidate example for a target, and the stages that yielded none.

    `match` keeps only records whose text holds the regex — either side of a
    DPO pair, and then both sides of it — which is how a phrase `grep` found is
    turned into the set of examples to weigh. `limit`
    caps the records per stage, drawn at random with `seed`, so a run on a big
    model can be sized to the machine; a DPO pair is one record and keeps both
    its sides. `incomplete`, if given, is filled with the records per stage
    skipped for carrying tool use the record cannot hold in trained form.
    """
    out: list[dict] = []
    skipped: dict[str, str] = {}
    pattern = re.compile(match, re.IGNORECASE) if match else None
    rng = random.Random(seed)
    for stage in target["stages"]:
        name = stage["stage"]
        if stages and name not in stages:
            continue
        if stage.get("hf_dataset"):
            cands, why, dropped = _context_candidates(target_name, stage)
            if dropped and incomplete is not None:
                incomplete[name] = dropped
        elif stage.get("sample_dataset"):
            cands, why, dropped = _docs_candidates(target_name, stage)
            if dropped and incomplete is not None:
                incomplete[name] = dropped
        else:
            continue  # a facts-only stage: nothing sampled, nothing to weigh
        if why:
            skipped[name] = why
            continue
        if pattern:
            cands = _matching(cands, pattern)
            if not cands:
                skipped[name] = f"no sampled example matches {match!r}"
                continue
        if limit:
            cands = _limit(cands, limit, rng)
        out.extend(cands)
    return out, skipped


def _matching(cands: list[dict], pattern) -> list[dict]:
    """The candidates whose *record* holds the pattern, both sides of a pair.

    A phrase found only in a rejected completion still names the pair as the
    example to weigh: the chosen side is what that pair taught instead, and
    dropping it would leave a rejected completion with nothing to localize on.
    So the regex is matched per record and every side of a matching record is
    kept.
    """
    hit = {_record(c) for c in cands if pattern.search(candidate_text(c))}
    return [c for c in cands if _record(c) in hit]


def _record(c: dict):
    """What makes two candidates sides of one stored record.

    The record's position in its file, assigned when the candidates were built:
    an id and a row are what a reader wants to see, but a record can lack both,
    and two such records keyed on (None, None) would be grouped as one — every
    side kept when one matched, and a limit of one returning them all.
    """
    return c["rec"] if "rec" in c else (c["id"], c["row"])


def _limit(cands: list[dict], limit: int, rng: random.Random) -> list[dict]:
    """At most `limit` *records* of a stage, drawn at random, sides kept together.

    A DPO pair is two candidates from one row. Sampling the candidates one by
    one could keep a rejected completion and drop its chosen one, and a stage
    reduced to rejected sides alone has nothing to localize the posterior on,
    so the draw is over rows and both sides of a drawn row come along.
    """
    units: dict = {}
    for c in cands:
        units.setdefault(_record(c), []).append(c)
    keys = list(units)
    if len(keys) <= limit:
        return cands
    keep = set(rng.sample(range(len(keys)), limit))
    return [c for i, k in enumerate(keys) if i in keep for c in units[k]]


def candidate_text(c: dict) -> str:
    return "\n".join(t["text"] for t in c["turns"])


def fit_text(c: dict) -> str:
    return "\n".join(t["text"] for t in c["turns"] if fits(t))


def query_candidate(query: str, prompt: str | None = None, chat: bool = True) -> dict:
    """The query as the same shape as a candidate: the text the model produced,
    fit, behind whatever it was replying to, context.

    `chat` says whether the checkpoint has a chat template. With one, a prompt
    is a user turn and the query an assistant turn, rendered through the
    template; without a prompt the query cannot be rendered at all, so `cmd_bif`
    requires `--prompt` there rather than score a string the model was never
    fit to in that form. Without a template — a base model such as Pythia — the
    model received the prompt followed directly by its continuation, so the two
    are raw spans: the prompt as context, the query as target, no role labels
    and nothing between them."""
    turns = []
    if prompt and chat:
        turns.append({"role": "user", "text": prompt})
        turns.append({"role": "assistant", "text": query})
    elif prompt:
        turns.append({"role": "text", "text": prompt, "fit": False})
        turns.append({"role": "text", "text": query, "fit": True})
    else:
        turns.append({"role": "text", "text": query})
    return {"stage": None, "kind": "query", "side": "query", "turns": turns, "id": None, "row": None,
            "source": None, "cut": False}


# --- Encoding -------------------------------------------------------------------


def pieces(tokenizer, turns: list[dict]) -> list[tuple[str, bool]]:
    """The example as (text, fit) spans in the form the model was trained on.

    A chat model is fit to its template's rendering of the turns, not to the raw
    text, so the spans come from rendering the conversation cumulatively and
    taking each turn as the text the rendering grew by. A base model, or a
    conversation the template cannot render, gets the turns joined with their
    roles — the honest fallback, and what a corpus document is anyway.
    """
    roles = {t["role"] for t in turns}
    template = getattr(tokenizer, "chat_template", None)
    if template and "text" not in roles:
        texts = [t["text"] for t in turns]
        rendered = _render(tokenizer, turns, texts)
        if rendered and not _nested(rendered):
            # A think model's generation prompt ends inside the response — OLMo
            # 3 Think's assistant header closes with `<think>` — so a reply
            # decoded from the model, or a query typed as one, starts after the
            # marker and the full rendering no longer begins with the prompt
            # rendering. Put the missing tail of the generation prompt back in
            # front of that turn's text: it was the model's context, and the
            # span the turn grows the rendering by then excludes it.
            for k, t in enumerate(turns, start=1):
                if t["role"] != "assistant" or rendered[k].startswith(rendered[k - 1]):
                    continue
                shared = _common_prefix(rendered[k - 1], rendered[k])
                tail = rendered[k - 1][shared:]
                if tail and not texts[k - 1].startswith(tail):
                    texts[k - 1] = tail + texts[k - 1]
            rendered = _render(tokenizer, turns, texts)
        if rendered and _nested(rendered):
            out = []
            for k, t in enumerate(turns, start=1):
                grown = rendered[k][len(rendered[k - 1]):]
                out.append((grown, fits(t)))
            return out
    if roles == {"text"}:
        return [(t["text"], fits(t)) for t in turns]
    out = []
    for t in turns:
        out.append((f"{t['role']}: ", False))
        out.append((t["text"] + "\n\n", fits(t)))
    return out


def _render(tokenizer, turns: list[dict], texts: list[str]) -> list[str] | None:
    """The conversation rendered cumulatively: `rendered[k]` is the template
    over the first k turns, with the generation prompt added only where an
    assistant turn comes next. None when the template rejects the shape."""
    rendered = [""]
    try:
        for k in range(1, len(turns) + 1):
            msgs = [{"role": t["role"], "content": text} for t, text in zip(turns[:k], texts[:k])]
            # The generation prompt is the assistant header, so it belongs only
            # where an assistant turn comes next. Added after a system turn that
            # a user turn follows, it would sit before the user text, the longer
            # rendering would no longer start with the shorter one, and the
            # whole example would fall back to roles.
            add_prompt = k < len(turns) and turns[k]["role"] == "assistant"
            rendered.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_prompt
            ))
    except Exception:  # a template that rejects these roles or this shape
        return None
    return rendered


def _nested(rendered: list[str]) -> bool:
    """Whether each rendering starts with the one before it, which is what lets
    a turn's span be read off as the text the rendering grew by."""
    return all(b.startswith(a) for a, b in zip(rendered, rendered[1:]))


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def encode(tokenizer, turns: list[dict], max_tokens: int) -> dict:
    """Token ids and loss labels for one example.

    Labels are the ids over fit spans and -100 elsewhere, which is what a causal
    LM loss reads as "context, not target". Over `max_tokens`, the *front* is
    dropped: the fit text sits at the end of an example, and cutting the end
    would keep the prompt and lose the thing being measured.
    """
    spans = pieces(tokenizer, turns)
    ids: list[int] = []
    labels: list[int] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    first = spans[0][0] if spans else ""
    bos_text = getattr(tokenizer, "bos_token", None)
    if prepends_bos(tokenizer) and not (bos_text and first.startswith(bos_text)):
        ids.append(bos)
        labels.append(-100)
    body, body_labels = _tokenize(tokenizer, spans)
    ids.extend(body)
    labels.extend(body_labels)
    truncated = len(ids) > max_tokens
    if truncated:
        ids, labels = ids[-max_tokens:], labels[-max_tokens:]
    # The label at position t is predicted from position t-1, so the first token
    # of a sequence is never a target however it is marked.
    if labels:
        labels[0] = -100
    return {
        "ids": ids,
        "labels": labels,
        "tokens": len(ids),
        "fit_tokens": sum(1 for x in labels if x != -100),
        "truncated": truncated,
    }


class SlowTokenizer(ValueError):
    """A tokenizer without offsets asked to mask an example with an internal boundary."""


def prepends_bos(tokenizer) -> bool:
    """Whether this tokenizer's own encoding puts its BOS token first.

    Having a `bos_token_id` is not the same thing: Pythia's tokenizer names the
    end-of-text token as BOS and never prepends it, and its training stream had
    no such separator, so inserting one gave every document and the query a
    first token none of them was trained with. The policy is read off the
    tokenizer by encoding a probe with and without special tokens and seeing
    whether the BOS id appears in front. Cached on the tokenizer, since it is
    asked once per candidate.
    """
    cached = getattr(tokenizer, "_bif_prepends_bos", None)
    if cached is not None:
        return cached
    bos = getattr(tokenizer, "bos_token_id", None)
    answer = False
    if bos is not None:
        try:
            with_special = tokenizer.encode("probe", add_special_tokens=True)
            without = tokenizer.encode("probe", add_special_tokens=False)
            answer = bool(with_special) and with_special[0] == bos and (
                not without or without[0] != bos
            )
        except TypeError:  # a tokenizer whose encode takes no such flag
            answer = False
    try:
        tokenizer._bif_prepends_bos = answer
    except (AttributeError, TypeError):
        pass
    return answer


def _tokenize(tokenizer, spans: list[tuple[str, bool]]) -> tuple[list[int], list[int]]:
    """Token ids for the whole rendering, and a label per token.

    The example is tokenized as one string, the way training saw it: a tokenizer
    that merges across a span boundary — the space closing an assistant header
    with the first word of the reply — gives a different sequence when the spans
    are encoded one at a time, and a loss over that sequence is a loss over
    tokens the model was never shown together. Labels come from character
    offsets: a token is a target when any character of it lies in a fit span,
    so the merged boundary token counts as the reply it begins.

    A tokenizer without offsets still encodes the whole string once, so the ids
    are the sequence the model saw, and aligns the mask by prefix token counts.
    That is exact only when no token crosses a fit boundary: a header's closing
    space merged into the first word of the reply can leave a fit span with no
    token of its own. So a tokenizer that reports itself slow (`is_fast` False)
    is refused for any example with an internal boundary — every model this
    layer has been run on ships a fast tokenizer, so the refusal costs nothing
    — while a standalone document, one span, is encoded either way. A tokenizer
    that does not report at all is a test stub, and those align exactly.
    """
    fit_ranges = []
    pos = 0
    for text, fit in spans:
        if fit and text:
            fit_ranges.append((pos, pos + len(text)))
        pos += len(text)
    full = "".join(text for text, _ in spans)
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        labels = []
        for tok, (s, e) in zip(ids, enc["offset_mapping"]):
            inside = any(s < hi and e > lo for lo, hi in fit_ranges)
            labels.append(tok if inside else -100)
        return ids, labels
    if getattr(tokenizer, "is_fast", None) is False and len(spans) > 1:
        raise SlowTokenizer(
            "this tokenizer has no character offsets, so the loss mask of an example with a "
            "prompt cannot be placed exactly; use the model's fast tokenizer"
        )
    ids = list(tokenizer.encode(full, add_special_tokens=False))
    ranges = []
    for lo, hi in fit_ranges:
        start = len(tokenizer.encode(full[:lo], add_special_tokens=False)) if lo else 0
        end = len(tokenizer.encode(full[:hi], add_special_tokens=False))
        ranges.append((start, end))
    labels = [tok if any(s <= i < e for s, e in ranges) else -100 for i, tok in enumerate(ids)]
    return ids, labels


# --- Sampling -------------------------------------------------------------------


def default_nbeta(n: int) -> float:
    """The inverse temperature for a posterior localized on `n` examples,
    n / ln(n), Watanabe's scale for the loss's pull against the prior.

    `n` is the size of the localized sample, the candidates whose mean loss is
    L, and not the minibatch: a minibatch only estimates L's gradient, and a
    temperature set from its size would make the posterior — every covariance
    and the learning coefficient with it — a function of a memory setting.
    devinterp's estimator does use the batch size here, as a convention for
    comparing coefficients across runs; this layer's posterior is a property of
    its data, and `--nbeta` overrides either way."""
    return n / math.log(n) if n > 1 else 1.0


def pick_device(name: str = "auto") -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id: str):
    """The tokenizer alone: cheap, and what decides whether the query needs a
    prompt before the weights are worth downloading."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load(model_id: str, device: str, dtype: str, tokenizer=None):
    """The checkpoint and its tokenizer, and the commit the weights came from."""
    import torch
    from transformers import AutoModelForCausalLM

    if tokenizer is None:
        tokenizer = load_tokenizer(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    except TypeError:
        # Transformers before 4.56 knows the precision keyword only as
        # `torch_dtype`, and hands an unknown `dtype` on to the model's
        # constructor, which rejects it. The extra's floor is 4.40.
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype))
    model.to(device)
    model.train(False)
    revision = getattr(model.config, "_commit_hash", None)
    return model, tokenizer, revision


def batch_losses(model, encoded: list[dict], device: str, batch: int) -> list[float]:
    """Mean token loss over the fit tokens of each example, evaluated at the
    model's current weights, in batches, without gradient."""
    import torch

    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(encoded), batch):
            chunk = encoded[i : i + batch]
            out.extend(_losses(model, chunk, device).tolist())
    return out


def _losses(model, chunk: list[dict], device: str):
    """Per-example mean fit-token loss for one right-padded batch, with grad."""
    import torch

    width = max(e["tokens"] for e in chunk)
    pad = 0
    ids = torch.full((len(chunk), width), pad, dtype=torch.long)
    labels = torch.full((len(chunk), width), -100, dtype=torch.long)
    mask = torch.zeros((len(chunk), width), dtype=torch.long)
    for r, e in enumerate(chunk):
        n = e["tokens"]
        ids[r, :n] = torch.tensor(e["ids"])
        labels[r, :n] = torch.tensor(e["labels"])
        mask[r, :n] = 1
    ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits
    target = labels[:, 1:]
    # One row at a time, widened to float32 only there: the whole batch's
    # logits in float32, plus a log-softmax of them, is two more tensors the
    # size of batch × tokens × vocabulary — some 3 GB each for a 7B model at
    # the default settings — on a device already holding the weights twice.
    out = []
    for r in range(len(chunk)):
        row = torch.nn.functional.cross_entropy(
            logits[r, :-1].float(), target[r], ignore_index=-100, reduction="sum"
        )
        count = (target[r] != -100).sum().clamp(min=1)
        out.append(row / count)
    return torch.stack(out)


def sample(
    model,
    encoded: list[dict],
    query: dict,
    *,
    device: str,
    chains: int,
    draws: int,
    burn_in: int,
    every: int,
    lr: float,
    nbeta: float,
    gamma: float,
    batch: int,
    eval_batch: int,
    seed: int,
    localize: list[int] | None = None,
    log=None,
) -> dict:
    """Run the SGLD chains and record every loss they pass through.

    Returns the loss of the query and of each candidate at each retained draw,
    per chain — `losses[chain][draw]` is `[query, cand_0, cand_1, ...]` — plus
    the minibatch loss at every step, and the losses at w* the chains started
    from. Everything downstream is arithmetic on that.

    `localize` is the indices into `encoded` the minibatches are drawn from,
    which is to say the data the posterior is localized on; every candidate is
    scored whether or not it is in it. The default is all of them. The caller
    leaves out what the model was trained *off* rather than toward — a DPO
    rejected completion — so the chain is not fit to text training pushed away
    from.
    """
    import torch

    if localize is None:
        localize = list(range(len(encoded)))
    if not localize:
        raise ValueError("nothing to localize the posterior on")
    params = [p for p in model.parameters() if p.requires_grad]
    # The chain walks in float32 whatever the model holds. At the default step
    # the noise is about 2e-4 a step, under the spacing between neighbouring
    # bfloat16 values for an ordinary weight, so a step applied to a bfloat16
    # parameter rounds to nothing or to a whole ULP and the chain is quantized
    # rather than Gaussian. A float32 model is its own master copy; a reduced
    # one gets a float32 master the updates accumulate in, cast back into the
    # parameter for the forward pass.
    master = [p if p.dtype == torch.float32 else p.detach().float().clone() for p in params]
    # w* stays in the checkpoint's own dtype: those are the values as loaded,
    # exactly representable there, and a second float32 copy of a 7B model is
    # 28 GB that buys nothing. The prior term widens it on the fly.
    origin = [p.detach().clone() for p in params]
    # One generator per device the parameters live on. Drawing on the CPU and
    # copying would move a full parameter's worth of noise over the bus every
    # step; for a 7B model over the default run that is tens of terabytes.
    gens: dict = {}

    def generator(dev, chain):
        key = str(dev)
        if key not in gens:
            try:
                gens[key] = torch.Generator(device=dev)
            except (RuntimeError, TypeError):  # a device without generator support
                gens[key] = torch.Generator(device="cpu")
            gens[key].manual_seed(seed * 1000 + chain)
        return gens[key]

    everything = [query, *encoded]
    at_origin = batch_losses(model, everything, device, eval_batch)
    losses: list[list[list[float]]] = []
    trajectory: list[list[float]] = []
    # The last retained draw is at step burn_in + (draws - 1) * every; nothing
    # after it reaches a covariance, and steps taken there would still land in
    # the trajectory and could trip the drift check on movement no draw saw.
    steps = burn_in + (draws - 1) * every + 1
    for chain in range(chains):
        with torch.no_grad():
            for p, m, w in zip(params, master, origin):
                m.copy_(w)
                if m is not p:
                    p.copy_(m.to(p.dtype))
        gens.clear()
        rng = random.Random(seed * 1000 + chain)
        chain_losses: list[list[float]] = []
        chain_traj: list[float] = []
        for step in range(steps):
            idx = [rng.choice(localize) for _ in range(batch)]
            for p in params:
                p.grad = None
            # One example per forward and backward pass, gradients accumulated,
            # rather than one pass over the minibatch. Under gradient tracking
            # the widened logits of every example in a batch stay alive until
            # backward runs, and for a 7B model that is a sequence-by-vocabulary
            # float32 tensor per example held at once; example by example, each
            # is freed before the next is made. Scaling by 1/batch keeps the
            # accumulated gradient the minibatch mean's.
            total = 0.0
            for i in idx:
                part = _losses(model, [encoded[i]], device)[0] / batch
                part.backward()
                total += float(part.detach())
            chain_traj.append(total)
            if not math.isfinite(chain_traj[-1]):
                raise RuntimeError(
                    f"chain {chain} diverged at step {step} (loss {chain_traj[-1]}); lower --lr"
                )
            with torch.no_grad():
                for p, m, w in zip(params, master, origin):
                    # A parameter the forward pass never reaches has no
                    # gradient; the prior still pulls it back to w*.
                    drift = gamma * (m - w.float())
                    if p.grad is not None:
                        drift = drift + nbeta * p.grad.float()
                    gen = generator(m.device, chain)
                    noise = torch.randn(m.shape, generator=gen, dtype=torch.float32,
                                        device=gen.device)
                    if noise.device != m.device:
                        noise = noise.to(m.device)
                    m.add_(-0.5 * lr * drift + math.sqrt(lr) * noise)
                    if m is not p:
                        p.copy_(m.to(p.dtype))
            if step >= burn_in and (step - burn_in) % every == 0:
                draw = batch_losses(model, everything, device, eval_batch)
                # The minibatch check above saw the weights before this step's
                # update; these losses are the first look at the weights after
                # it, and on the last draw the only one. A NaN here would ride
                # through every statistic and compare as false to every bound.
                if not all(math.isfinite(x) for x in draw):
                    raise RuntimeError(
                        f"chain {chain} diverged at step {step}: a recorded loss is not finite; lower --lr"
                    )
                chain_losses.append(draw)
                if log:
                    log(f"chain {chain + 1}/{chains}: draw {len(chain_losses)}/{draws}, "
                        f"minibatch loss {chain_traj[-1]:.3f}")
        losses.append(chain_losses)
        trajectory.append(chain_traj)
    with torch.no_grad():
        for p, w in zip(params, origin):
            p.copy_(w)
    # `localized` is the indices themselves, not only their count, because the
    # learning coefficient has to be taken over the same loss the chains were
    # localized on, and a candidate scored but never fit is not part of it.
    return {"at_origin": at_origin, "losses": losses, "trajectory": trajectory, "nbeta": nbeta,
            "burn_in": burn_in, "localized": list(localize), "localized_on": len(localize)}


# --- Statistics -----------------------------------------------------------------


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _cov(xs, ys) -> float:
    mx, my = _mean(xs), _mean(ys)
    return _mean((x - mx) * (y - my) for x, y in zip(xs, ys))


def _std(xs) -> float:
    m = _mean(xs)
    return math.sqrt(_mean((x - m) ** 2 for x in xs))


def _stderr(xs: list[float]) -> float | None:
    """Standard error of the mean across chains; None from one chain, which has
    no spread to report and should not print as a confident zero."""
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def _residual(xs: list[float], ms: list[float]) -> list[float]:
    """`xs` with its least-squares projection on `ms` removed: what is left of a
    loss series once the part that moves with the chain's average is gone."""
    vm = _cov(ms, ms)
    if vm <= 0:
        return [x - _mean(xs) for x in xs]
    b = _cov(xs, ms) / vm
    mx, mm = _mean(xs), _mean(ms)
    return [(x - mx) - b * (m - mm) for x, m in zip(xs, ms)]


def influence(losses: list[list[list[float]]]) -> list[dict]:
    """Per-candidate covariance and correlation with the query, within each
    chain then averaged, with the across-chain standard error.

    `losses[chain][draw]` is `[query, cand_0, ...]`. Correlation is reported
    beside covariance because covariance scales with how much a loss moves, and
    a long, high-entropy document moves more than a short answer whether or not
    it has anything to do with the query.

    `partial` is the covariance after the chain's shared movement is taken out
    of both series — each is regressed on the per-draw mean loss of the *other*
    candidates and the residuals covaried. Every loss rises and falls with the
    chain's excursion from w*, and on Pythia-70m that one component gave all 200
    documents a correlation near 0.85 with the query and ranked them by how far
    their own loss swings. What is left after removing it is the part of an
    example's movement that is specific to the query, and that is what the
    ranking uses.

    The control leaves the candidate being scored out of the mean, because a
    candidate in its own control is partly regressed on itself, and a lone
    candidate would be *entirely* its own control: the residual would be zero
    and every partial would print as an exact, meaningless tie. With one
    candidate there is no other example to control on, so its partial is the
    raw covariance, and `partial_control` says which was done.
    """
    n = len(losses[0][0]) - 1
    control = "leave-one-out" if n > 1 else "none"
    out = []
    for j in range(n):
        covs, corrs, partials = [], [], []
        for chain in losses:
            q = [d[0] for d in chain]
            c = [d[j + 1] for d in chain]
            cov = _cov(q, c)
            sq, sc = _std(q), _std(c)
            covs.append(cov)
            corrs.append(cov / (sq * sc) if sq > 0 and sc > 0 else 0.0)
            if n > 1:
                m = [_mean(d[k] for k in range(1, n + 1) if k != j + 1) for d in chain]
                partials.append(_cov(_residual(q, m), _residual(c, m)))
            else:
                partials.append(cov)
        out.append({
            "cov": _mean(covs),
            "cov_stderr": _stderr(covs),
            "corr": _mean(corrs),
            "partial": _mean(partials),
            "partial_stderr": _stderr(partials),
            "partial_control": control,
            "chain_covs": covs,
        })
    return out


def llc(run: dict) -> dict:
    """The local learning coefficient nβ(E[L] - L(w*)) over the candidates the
    posterior was localized on, per chain and pooled, as the check that the
    chains sampled a posterior around w* rather than fell off it.

    L here is the loss the chains were fit to — the mean over `localized`, the
    candidates the minibatches were drawn from — and not over every candidate
    scored. A DPO rejected completion is scored at every draw and never fit, and
    its loss under the posterior has no reason to sit near its loss at w*; taking
    it into E[L] would report a coefficient for a loss no chain sampled, and a
    few rejected sides far from the origin could call the chains invalid, or
    valid, on their own. `over` says how many candidates the coefficient covers.

    Beside it, the drift: how much the chain's loss over the localized
    candidates — the same fixed set at every retained draw — moved between the
    first and last quarter of its draws. A positive coefficient alone does not
    say the chain was stationary: a chain climbing steadily away from w* has one
    too, and the run at ten times the step size that motivated this check
    reported "sat near w*" while its loss doubled. The minibatch trajectory is
    not what is compared, since each step's minibatch is a fresh random draw
    and a run of harder documents late in the chain would read as drift while a
    real drift could hide behind easier ones.
    """
    # Column 0 of every draw is the query; candidate i is column i + 1. A run
    # without `localized` (one written before it was recorded) was localized on
    # every candidate.
    localized = run.get("localized")
    cols = [i + 1 for i in (localized if localized is not None else range(len(run["at_origin"]) - 1))]
    base = _mean(run["at_origin"][i] for i in cols)
    per_chain = []
    for chain in run["losses"]:
        expected = _mean(_mean(d[i] for i in cols) for d in chain)
        per_chain.append(run["nbeta"] * (expected - base))
    drift = []
    for chain in run["losses"]:
        series = [_mean(d[i] for i in cols) for d in chain]
        q = max(1, len(series) // 4)
        drift.append(_mean(series[-q:]) - _mean(series[:q]) if series else 0.0)
    return {"per_chain": per_chain, "mean": _mean(per_chain), "loss_at_origin": base, "drift": drift,
            "over": len(cols)}


# A chain whose loss moved, up or down, by more than this share of the loss at
# w* between its first and last retained quarter was still travelling, not
# sampling around a point.
MAX_DRIFT = 0.25


def baseline(losses: list[list[list[float]]]) -> float:
    """The query's covariance with the *average* candidate loss, within chain.

    Every loss moves with the chain's overall excursion from w*, so every
    covariance carries a shared component with the query's. This is that
    component, and the per-example numbers are read against it: a candidate
    above the line pulls the query harder than the sample as a whole does.

    Unlike `llc`, this is over every candidate scored, rejected sides included:
    it is a control for the ranking, and the ranking holds the rejected sides,
    so the line each is read against has to be the average of the same set.
    """
    covs = []
    for chain in losses:
        q = [d[0] for d in chain]
        avg = [_mean(d[1:]) for d in chain]
        covs.append(_cov(q, avg))
    return _mean(covs)


def summarize(cands: list[dict], stats: list[dict], key) -> list[dict]:
    """Candidates grouped by `key(c)` — a tuple of (field, value) pairs — with
    the group's mean covariance, the share pulling toward the query, and its
    strongest member.

    Direction is read off `pull`, not off the partial covariance: on a DPO
    rejected side the two have opposite signs, since the objective pushed the
    model off that text, and a table that counted a positive partial there as
    "toward" would report the training direction backwards.
    """
    groups: dict = {}
    for c, s in zip(cands, stats):
        groups.setdefault(key(c), []).append((c, s))
    out = []
    for k, members in groups.items():
        covs = [s["cov"] for _, s in members]
        partials = [s.get("partial", s["cov"]) for _, s in members]
        pulls = [pull(c, s) for c, s in members]
        best = max(members, key=lambda m: pull(*m))
        out.append({
            **dict(k),
            "n": len(members),
            "mean_cov": _mean(covs),
            "mean_corr": _mean(s["corr"] for _, s in members),
            "mean_partial": _mean(partials),
            "mean_pull": _mean(pulls),
            "toward": sum(1 for x in pulls if x > 0),
            "best": {"id": best[0]["id"], "row": best[0]["row"], "pull": pull(*best)},
        })
    return sorted(out, key=lambda g: -g["mean_pull"])


def pull(c: dict, s: dict) -> float:
    """How hard an example pulls the model toward the query, signed.

    The posterior covariance — the Bayesian influence itself — except on a DPO
    rejected completion, where training pushed the model off the text: a loss
    that moves with the query's there is a pair that taught the model away from
    it, so the sign is flipped. The partial covariance is not used here: the
    identity this layer rests on is about Cov(ℓ_query, ℓ_example), and a
    covariance with the other candidates regressed out is a different quantity
    that can shrink or reverse a real influence. It stays beside the ranking as
    the diagnostic it is, the part of the movement specific to the query.
    """
    value = s["cov"]
    return -value if c.get("side") == "rejected" else value


# --- Result ---------------------------------------------------------------------


def result(target_name, model_id, revision, query, prompt, cands, encoded, run, skipped, settings) -> dict:
    stats = influence(run["losses"])
    records = []
    for c, e, s, base in zip(cands, encoded, stats, run["at_origin"][1:]):
        records.append({
            "stage": c["stage"],
            "kind": c["kind"],
            "side": c["side"],
            "source": c["source"],
            "id": c["id"],
            "row": c["row"],
            "tokens": e["tokens"],
            "fit_tokens": e["fit_tokens"],
            "truncated": e["truncated"],
            "cut": c["cut"],
            "loss_at_origin": base,
            **s,
            "snippet": fit_text(c)[:SNIPPET],
        })
    common = baseline(run["losses"])
    for r in records:
        r["above_baseline"] = r["cov"] - common
        r["pull"] = pull(r, r)
    ranked = sorted(range(len(records)), key=lambda i: -records[i]["pull"])
    for rank, i in enumerate(ranked, start=1):
        records[i]["rank"] = rank
    query_losses = [d[0] for chain in run["losses"] for d in chain]
    return {
        "target": target_name,
        "model": model_id,
        "model_revision": revision,
        "query": query,
        "prompt": prompt,
        "settings": settings,
        "candidates": len(cands),
        # How many of the candidates the SGLD minibatches were drawn from; the
        # rest (DPO rejected completions) were scored at every draw but never
        # fit to.
        "localized_on": run.get("localized_on", len(cands)),
        "skipped": skipped,
        "llc": llc(run),
        "baseline_cov": common,
        "partial_control": stats[0]["partial_control"] if stats else None,
        "query_loss": {
            "at_origin": run["at_origin"][0],
            "posterior_mean": _mean(query_losses),
            "posterior_std": _std(query_losses),
        },
        "stages": summarize(cands, stats, lambda c: (("stage", c["stage"]), ("side", c["side"]))),
        "sources": summarize(
            cands, stats,
            lambda c: (("stage", c["stage"]), ("side", c["side"]), ("source", c["source"])),
        ),
        "trajectory": run["trajectory"],
        "records": records,
        # The raw draws at full precision, so any statistic can be recomputed
        # from the file and match: `draws[chain][draw]` is the query's loss
        # followed by each record's, in record order. Rounding them to four
        # places moved partial covariances that sit below 1e-6.
        "draws": [[list(d) for d in chain] for chain in run["losses"]],
    }


def committed(target_name: str) -> list[dict]:
    """Every result file this layer has written for a target, either directory."""
    out = []
    seen = set()
    for directory in (paths.RESULTS, paths.SITE_DATA):
        for path in sorted(directory.glob(f"{target_name}.bif-*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            out.append(json.loads(path.read_text()))
    return out


# --- Rendering ------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{digits}f}"


def _one_line(text: str) -> str:
    """One printable line: whitespace collapsed, and any other control
    character shown as its escape. Snippets and source names are text from the
    training data, and a stored ESC sequence would otherwise reach the terminal
    of whoever reads the report as an instruction to it."""
    flat = re.sub(r"\s+", " ", text).strip()
    return "".join(ch if ch.isprintable() else repr(ch)[1:-1] for ch in flat)


def render(res: dict, top: int = 5) -> list[str]:
    """The result as the lines `report` prints: what was measured on which
    checkpoint, whether the sampler behaved, the stages ranked, and the examples
    at either end."""
    lines = [f"### Bayesian influence — {_one_line(res['query'])[:80]!r}", ""]
    rev = res.get("model_revision")
    lines.append(
        f"Checkpoint `{res['model']}`" + (f" at `{rev[:12]}`" if rev else "")
        + f"; {res['candidates']} candidate examples from the committed samples."
    )
    localized = res.get("localized_on", res["candidates"])
    if localized < res["candidates"]:
        lines.append(
            f"The posterior is localized on {localized} of them, the text the model was fit toward; "
            f"the other {res['candidates'] - localized} are DPO rejected completions, scored at every "
            f"draw but not trained on."
        )
    if res.get("prompt"):
        lines.append(f"Query is the reply to: {_one_line(res['prompt'])[:120]!r}")
    s = res["settings"]
    lines.append(
        f"SGLD: {s['chains']} chains × {s['draws']} draws after {s['burn_in']} burn-in, "
        f"lr {s['lr']:g}, nβ {s['nbeta']:.3g}, γ {s['gamma']:g}, minibatch {s['batch']}, "
        f"{s['max_tokens']} tokens per example."
    )
    q = res["query_loss"]
    lines.append(
        f"Query loss {q['at_origin']:.3f} at w*, {q['posterior_mean']:.3f} ± {q['posterior_std']:.3f} "
        f"under the posterior."
    )
    lc = res["llc"]
    per = ", ".join(f"{x:.1f}" for x in lc["per_chain"])
    drift = lc.get("drift") or []
    # Either direction: a chain still descending into a lower-loss region of
    # the sample is as far from stationary as one still climbing, and a shared
    # trend in every loss series makes covariances out of nothing either way.
    moving = [d for d in drift if abs(d) > MAX_DRIFT * lc["loss_at_origin"]]
    if moving:
        verdict = (f"{len(moving)} of {len(drift)} chains were still drifting from w* "
                   f"(loss over the localized candidates {', '.join(f'{d:+.2f}' for d in moving)} "
                   f"across the retained draws), so lower --lr before reading the covariances as "
                   f"posterior covariances")
    else:
        drifted = ", ".join(f"{d:+.2f}" for d in drift) if drift else "n/a"
        verdict = f"the chains sat near w* (loss drift per chain over the localized candidates: {drifted})"
    if lc["mean"] <= 0 or any(x <= 0 for x in lc["per_chain"]):
        # Not a fault by itself. w* minimizes the training set, not this sample
        # of a few hundred candidates, so a chain that sampled the posterior
        # correctly can still sit at a lower loss on the sample than w* does.
        # The drift check above is what says whether the chain was stationary.
        verdict += (
            "; the coefficient is at or below zero, so on the localized sample the chains sat "
            "at a lower loss than w*, which a posterior localized on a few hundred examples "
            "rather than the training set can do without anything being wrong"
        )
    over = lc.get("over")
    scope = (f" over the {over} localized candidates" if over is not None and over < res["candidates"]
             else "")
    lines.append(f"Local learning coefficient {lc['mean']:.1f}{scope} (per chain: {per}); {verdict}.")
    lines.append("")
    lines.append("Positive covariance: training harder on the example would lower the query's loss, "
                 "so it pulls the model toward the query. On a DPO *rejected* side, which the objective "
                 "pushed the model off, the same sign reads as evidence the pair taught the model away "
                 "from it.")
    if res.get("baseline_cov") is not None:
        lines.append(
            f"Every loss moves with the chain's excursion from w*, and the query covaries "
            f"{_fmt(res['baseline_cov'])} with the average candidate. The ranking below is by the "
            f"covariance itself, which is the influence; read it against that line, since most "
            f"of every value is the shared movement. The *partial* covariance beside it has that "
            f"movement regressed out of both series and is the part of each example's movement "
            f"specific to the query — a diagnostic, not the estimator. On a DPO rejected side "
            f"the sign is flipped before anything is called toward or away, since training "
            f"pushed the model off that text (`pull` in the file; `cov` itself is kept as is)."
        )
        if res.get("partial_control") == "none":
            lines.append(
                "With one candidate there is no other example to regress on, so its partial "
                "covariance here is the raw covariance."
            )
    lines.append("")
    lines.append("| stage | side | n | mean pull | mean cov | mean partial | mean corr | toward |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for g in res["stages"]:
        partial = g.get("mean_partial", g["mean_cov"])
        lines.append(
            f"| {g['stage']} | {g['side']} | {g['n']} | {_fmt(g.get('mean_pull', g['mean_cov']))} | "
            f"{_fmt(g['mean_cov'])} | {_fmt(partial)} | {_fmt(g['mean_corr'], 3)} | "
            f"{g['toward']}/{g['n']} |"
        )
    for stage, why in (res.get("skipped") or {}).items():
        lines.append(f"| {stage} | — | 0 | — | — | — | — | not weighed: {why} |")
    lines.append("")
    records = sorted(res["records"], key=lambda r: r["rank"])
    if records:
        lines.append(f"Pulls hardest toward the query (top {min(top, len(records))}):")
        lines.append("")
        for r in records[:top]:
            lines.append(_record_line(r))
        lines.append("")
        lines.append(f"Pulls hardest away (bottom {min(top, len(records))}):")
        lines.append("")
        for r in records[-top:][::-1]:
            lines.append(_record_line(r))
        lines.append("")
    lines.append(
        "The posterior is localized on these candidates, not on the training set, and the "
        "covariance is of the loss on this checkpoint. Neither is what the model's trainer "
        "saw; both are what can be measured from what it published."
    )
    return lines


def _record_line(r: dict) -> str:
    where = " / ".join(_one_line(str(x)) for x in (r["stage"], r["side"], r["source"]) if x)
    partial = r.get("partial", r["cov"])
    value = r.get("pull", r["cov"])
    se = r.get("cov_stderr")
    err = f" ± {se:.4f}" if se is not None else ""
    ident = _one_line(str(r["id"])) if r["id"] is not None else (f"row {r['row']}" if r["row"] is not None else "")
    flags = "".join(
        f" [{f}]" for f, on in (("truncated", r["truncated"]), ("cut", r["cut"])) if on
    )
    detail = f"partial {_fmt(partial)}, corr {_fmt(r['corr'], 3)}"
    if r.get("side") == "rejected":
        detail = f"rejected side, cov {_fmt(r['cov'])} flipped; {detail}"
    return (
        f"- {_fmt(value)}{err} ({detail}) {where} {ident}{flags}: "
        f"“{_one_line(r['snippet'])[:SNIPPET]}”"
    )


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)
