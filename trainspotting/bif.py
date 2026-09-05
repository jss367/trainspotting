"""Experimental local loss sensitivity on standalone text.

The public command is restricted to Pythia-70m-deduped and corpus documents.
It samples p(w) proportional to exp(-nbeta * L(w) - gamma/2 * ||w-w*||^2),
where L is the mean of the supplied standalone texts' mean token losses.
For the perturbation L + delta * loss_j, the derivative of the posterior
query-loss expectation is -nbeta * Cov(loss_query, loss_j). This is not an
estimate of historical training attribution or of a DPO training objective.

The SGLD loop follows Kreer et al., Bayesian Influence Functions for
Hessian-Free Data Attribution (2025), https://arxiv.org/abs/2509.26544.
Recorded draws are retained even when diagnostics are inconclusive. Passing
finite-sample checks is not proof of convergence or validation on a language
model; the exact Gaussian validation only checks the sampler and identity.

The private chat reconstruction helpers are retained for follow-up research;
they are not reachable through candidate selection in this release.
"""

import json
import math
import random
import re
import sys

from . import context, paths, registry, search
from .bif_diagnostics import diagnostics

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

SUPPORTED_MODEL = "EleutherAI/pythia-70m-deduped"

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


def marks_fidelity(records: list[dict]) -> bool:
    """Whether a context file records how faithfully each turn was stored.

    `context._turns` marks a turn `raw` when it is stored as written and gives
    it `chars_raw` when only its think markers and whitespace are gone. A turn
    with neither, in a file that marks others, is structured content — a list
    of parts, a dict — stored as a Python repr, which is not text the model was
    scored on. A file with no marks anywhere predates the markers (the committed
    Instruct samples), and its turns are taken at face value.
    """
    for rec in records:
        turns = list(rec.get("turns") or []) + [
            t for side in ("chosen", "rejected") for t in ((rec.get(side) or {}).get("turns") or [])
        ]
        if any(t.get("raw") or t.get("chars_raw") is not None for t in turns):
            return True
    return False


def incomplete(turns, shared: int = 0, marked: bool = False) -> bool:
    """Whether a stored record is not the example the model was trained on.

    Three ways: a turn carries or notes fields this layer cannot rebuild
    (STRUCTURED); any turn was cut at the context record's 4,000-character
    field limit; or, in a file that marks fidelity (`marked`), a turn carries
    neither mark, which is structured content stored as its Python repr. A cut fit turn is not a prefix of the training example the way
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
    if marked and any(not t.get("raw") and t.get("chars_raw") is None for t in turns):
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
    marked = marks_fidelity(data.get("records", []))
    for ordinal, rec in enumerate(data.get("records", [])):
        sides = [rec.get("turns") or []] if kind == "sft" else [
            (rec.get(side) or {}).get("turns") or [] for side in ("chosen", "rejected")
        ]
        shared_turns = context.branch_point(*sides) if kind == "dpo" else 0
        if any(incomplete(t, shared_turns, marked) for t in sides):
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
    """Standalone corpus documents only; deferred stages are skipped explicitly.

    Filtering changes the objective as well as the scored candidates. The
    sample does not reproduce the original packed pretraining sequences.
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
            skipped[name] = "post-training and conversation objectives are deferred in this experiment"
            continue
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
    taking each turn as the text the rendering grew by; a conversation its
    template cannot render that way is refused (Unrenderable) rather than
    guessed at. A base model has no template, so its chat-shaped candidates are
    the turns joined with their roles — no trained form exists to match, and a
    corpus document is a single span either way.
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
        # The template exists and will not render this conversation, or renders
        # it in a way that cannot be read off turn by turn. Inventing role text
        # instead would score bytes the checkpoint's template never produced
        # and report covariances for a different input; the caller drops the
        # example and counts it.
        raise Unrenderable(
            "the checkpoint's chat template cannot render this conversation turn by turn"
        )
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


class Unrenderable(ValueError):
    """A chat template that cannot render a conversation turn by turn."""


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
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
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
    loss_fn=None,
) -> dict:
    """Run the SGLD chains and record every loss they pass through.

    Returns the loss of the query and of each candidate at each retained draw,
    per chain — `losses[chain][draw]` is `[query, cand_0, cand_1, ...]` — plus
    the minibatch loss at every step, and the losses at w* the chains started
    from. Everything downstream is arithmetic on that.

    `localize` selects which candidates define the mean sampling loss. All
    candidates are scored. The public text command localizes on all documents;
    the exact-posterior validation also uses score-only observables.
    """
    import torch

    # A differentiable per-example loss adapter lets the exact-posterior
    # validation exercise this very loop, including its noise and prior.
    loss_fn = loss_fn or _losses

    def evaluate(examples):
        with torch.no_grad():
            return [float(x) for start in range(0, len(examples), eval_batch)
                    for x in loss_fn(model, examples[start:start + eval_batch], device)]

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
    at_origin = evaluate(everything)
    losses: list[list[list[float]]] = []
    trajectory: list[list[float]] = []
    # The last retained draw is at step burn_in + (draws - 1) * every; nothing
    # after it reaches a covariance, and steps taken there would still land in
    # the trajectory and could trip the drift check on movement no draw saw.
    steps = burn_in + (draws - 1) * every + 1
    try:
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
                    part = loss_fn(model, [encoded[i]], device)[0] / batch
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
                    draw = evaluate(everything)
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
    finally:
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
    example's movement that is specific to the query, and is a diagnostic only.

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
    """Descriptive covariance summaries; no historical training interpretation."""
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
            "positive_covariances": sum(1 for x in pulls if x > 0),
            "best": {"id": best[0]["id"], "row": best[0]["row"], "cov": best[1]["cov"]},
        })
    return sorted(out, key=lambda g: -g["mean_cov"])


def pull(c: dict, s: dict) -> float:
    """Unmodified sample covariance; no DPO sign interpretation."""
    return s["cov"]


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
    diagnostic = diagnostics(run["losses"], run.get("localized"))
    common = baseline(run["losses"])
    for r in records:
        r["above_baseline"] = r["cov"] - common
    ranked = sorted(range(len(records)), key=lambda i: -records[i]["cov"])
    for rank, i in enumerate(ranked, start=1):
        records[i]["rank"] = rank if diagnostic["status"] == "checks_passed" else None
    query_losses = [d[0] for chain in run["losses"] for d in chain]
    return {
        "schema_version": 2,
        "experimental": True,
        "objective": "standalone_text_mean_token_loss",
        "diagnostics": diagnostic,
        "target": target_name,
        "model": model_id,
        "model_revision": revision,
        "query": query,
        "prompt": prompt,
        "settings": settings,
        "candidates": len(cands),
        # Keep the exact objective subset for diagnostics and reproducibility.
        "localized_on": run.get("localized_on", len(cands)),
        "localized": run.get("localized", list(range(len(cands)))),
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
    """Recheck raw draws, including older files, before displaying any ranking."""
    diag = diagnostics(res.get("draws", []), res.get("localized"))
    legacy = res.get("schema_version") != 2
    supported = (res.get("model") == SUPPORTED_MODEL
                 and all(r.get("side") == "document" for r in res["records"]))
    lines = [f"### Experimental loss sensitivity — {_one_line(res['query'])[:80]!r}", ""]
    rev = res.get("model_revision")
    lines.append(f"Checkpoint `{res['model']}`" + (f" at `{rev[:12]}`" if rev else "")
                 + f"; {res['candidates']} standalone candidate texts.")
    if res.get("prompt"):
        lines.append(f"Continuation context: {_one_line(res['prompt'])[:120]!r}")
    s = res["settings"]
    lines.append(f"SGLD: {s['chains']} chains × {s['draws']} retained draws, "
                 f"{s['burn_in']} burn-in steps, lr {s['lr']:g}, nβ {s['nbeta']:.3g}, "
                 f"γ {s['gamma']:g}, {s['max_tokens']} tokens per text.")
    lines.append("Objective: mean token loss on standalone texts, not on the training set "
                 "or its original packed sequences. Filtering changes this objective.")
    if legacy:
        lines.append("Historical run: diagnostics recomputed from the saved draws; stored ranks are ignored.")
    reasons = list(diag["reasons"])
    if not supported:
        reasons.append("this checkpoint or candidate objective is outside the supported experiment")
    if reasons:
        lines.extend(["", "**Inconclusive sampling — influence ranking withheld.**", ""])
        lines.extend(f"- {reason}" for reason in reasons)
        lines.append("")
        lines.append("Inspect the saved loss traces; compare longer burn-in, longer sampling, "
                     "step sizes and independent seeds. Reducing the step size alone does not "
                     "establish convergence. Raw draws and descriptive covariances remain in the file.")
    else:
        lines.extend(["", "Sampling screens passed; this is still an experimental sensitivity estimate, "
                      "not proof of convergence or historical training attribution."])
        lines.append("Positive covariance predicts a lower posterior query loss under the specific "
                     "perturbation L + δ·loss_example; negative predicts a higher loss. "
                     "The derivative is -nβ times covariance. Partials are diagnostics only. "
                     "The ± value is an across-chain standard error and excludes sampling bias.")
        # Recompute to avoid interpreting stale ranks or sign-flipped legacy fields.
        stats = influence(res["draws"])
        records = [{**r, **stat, "pull": stat["cov"]} for r, stat in zip(res["records"], stats)]
        records.sort(key=lambda r: -r["cov"])
        for title, subset in (("Largest positive sample covariances", [r for r in records if r["cov"] > 0][:top]),
                              ("Most negative sample covariances", [r for r in records[::-1] if r["cov"] < 0][:top])):
            lines.extend(["", title + ":"])
            lines.extend(_record_line(r) for r in subset)
            if not subset:
                lines.append("None in this run.")
    for stage, why in (res.get("skipped") or {}).items():
        lines.append(f"Not analyzed: {_one_line(stage)} — {_one_line(why)}")
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
    return (
        f"- {_fmt(value)}{err} ({detail}) {where} {ident}{flags}: "
        f"“{_one_line(r['snippet'])[:SNIPPET]}”"
    )


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)
