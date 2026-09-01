"""Put every stage's match rate on one scale: tokens the model was fit to.

`ask` answers "what share of this stage matches the question" — a share of rows
for a post-training stage, a share of documents for a corpus. Those shares are
not comparable to each other and neither is a measure of how much training a
model got. Olmo 3 7B sees 5.93T pretraining tokens and about 4B tokens of SFT
target text; a 6% rate over Dolci-Instruct-DPO and a 1% rate over Dolma 3 are
three orders of magnitude apart in what they represent.

This layer multiplies the two together. For each stage it computes

    matching tokens  =  rate × stage size

and sums them, so a question asked across the pipeline gets one number and a
share of the whole budget.

## The unit

**Tokens the model was fit to** — loss-bearing tokens. That is the only unit
under which the stages mean the same thing, and it is not the same as the size
of the dataset:

| Stage | Fit to |
|---|---|
| pretrain / midtrain / long-context | every token; the stage size from the Olmo 3 paper is the answer |
| sft | the assistant turns, reasoning spans included — not the prompt it read |
| dpo | both completions; the preference loss is computed over the pair |
| rlvr | rollouts generated during training, which the dataset does not contain |

RLVR is the honest gap. The published mix holds prompts, verifiers and some
reference generations, not the text the policy was actually trained on, and how
many rollouts per prompt the run took is not in the data. What this reports for
an RL stage is a **floor**: one reference rollout per prompt. Real training
samples several, so the true figure is larger by whatever that factor was.

## The rate

Which weighting is right depends on how the stage was sampled, and the two
halves of this tool sample differently.

**Post-training rows are drawn uniformly**, so a rate over rows is a rate over
examples, not over training. A stage's matching examples are not average-length
— in Dolci-Think-SFT a reasoning trace runs tens of thousands of characters and
a one-line prompt does not — so the rate is weighed by fit characters:

    rate  =  Σ fit chars over matching examples  /  Σ fit chars over all judged

**Corpus documents depend on the route.** Being a corpus is not the property
that decides this; how the documents were drawn is, and `registry.sample_route`
is where that is recorded.

A **shard**-drawn corpus needs no weighting. `pretrain.sample_documents` draws
shards with probability proportional to compressed size and takes one document
from each, precisely so the source mix comes out token-weighted. Under that
design every sampled document represents the same byte mass — a stratum holding
twice the bytes wins twice as many shard draws, so it contributes twice as many
documents — and the plain document rate *is* the byte-weighted rate. Multiplying
by each document's own length would apply the size weighting a second time:
Longmino's 200k-character PDFs and a 2k-character web page each stand for one
draw's worth of corpus, and charging the first 100x the second would let
long-document strata swamp the estimate.

A **rows**-drawn corpus needs exactly the weighting post-training rows need.
`pretrain.sample_rows_documents` draws documents uniformly from the whole
corpus, so its document rate is the share of documents that match and not the
share of training. The deduplicated Pile's sampled documents run from a few
hundred characters to seventy thousand; matches landing in the long ones (or the
short ones) would otherwise carry a document share straight into a token count.
So a rows corpus uses the same Σ-fit-chars rate as an SFT stage.

Each stage records which rule it used in `weighting`.

The residual the shard route leaves is within a shard, not between shards: the
sampler picks uniformly among a shard's reachable documents rather than
proportionally to their length, so a long document is slightly underweighted
against its byte share. One document per shard gives nothing to estimate that
shard's mean length from, so it is left as a caveat rather than corrected badly.

The interval is the count-based one (cluster-corrected for corpora, where the
`ask` run already stored it) rescaled by rate / count-rate — which is exactly 1
for a shard corpus and the length correction everywhere else. Where it is not 1
it carries the sampling uncertainty in the *rate* and not the extra uncertainty
in the length ratio, so it is narrower than the truth by that much.

## What it does not do

It weighs tokens, not learning. A DPO preference token, an RL gradient step and
a pretraining cross-entropy token do not move a model equally, and post-training
is widely believed to be far higher-leverage per token than pretraining. Nothing
here corrects for that, so read the output as an exposure budget rather than as
an attribution of behaviour.
"""

import json

from . import context, extract, paths, registry
from .stats import cluster_wilson, wilson

# Characters per token, for turning a measured character count into the unit the
# registry's stage sizes are in. Roughly right for English prose and roughly
# wrong for code and CJK, both of which are in these mixes. Every figure derived
# through it is an estimate and is printed as one; the ratios between stages,
# which is what the comparison actually turns on, are much less sensitive to it
# than the absolute counts.
CHARS_PER_TOKEN = 4.0

# How each kind of example's fit text is described in the output. Keyed by
# registry.stage_kind.
FIT_TEXT = {
    "sft": "assistant turns (reasoning included)",
    "dpo": "both completions of the pair",
    "rlvr": "reference rollouts (one per prompt — a floor)",
    "chat": "nothing; a log is not a training example",
}


def _turn_chars(turn: dict) -> int:
    """True length of one stored turn, reasoning span included.

    `context` splits a `<think>` span out of the turn it precedes so the answer
    stays visible under truncation, but the model was fit to both, and in a
    think mix the reasoning is most of the length. Counting only the answer
    would understate Dolci-Think-SFT by about 20x.

    `chars` is the length before the 4,000-character display cut, so these are
    the real lengths and not what the record shows.
    """
    return turn.get("chars", 0) + (turn.get("reasoning") or {}).get("chars", 0)


def fit_chars(rec: dict) -> int | None:
    """Characters of `rec` the model was trained to produce, or None if the
    dataset does not hold them.

    None is not zero. An RL row has a target — the rollouts the policy sampled —
    and simply does not ship it; scoring that as zero would report an RL stage as
    weightless, which is the opposite of what it is.
    """
    kind = rec.get("kind")
    if kind == "sft":
        return sum(_turn_chars(t) for t in rec.get("turns", []) if t.get("role") == "assistant")
    if kind == "dpo":
        # Only past the branch point. Both sides store the whole conversation,
        # so an earlier assistant turn is shared history the pair is judged in
        # rather than either completion the preference loss scores — counting it
        # once per side charges the stage twice for text it was never preferred
        # for. See context.branch_point.
        chosen = (rec.get("chosen") or {}).get("turns", [])
        rejected = (rec.get("rejected") or {}).get("turns", [])
        shared = context.branch_point(chosen, rejected)
        return sum(
            _turn_chars(t)
            for side in (chosen, rejected)
            for t in side[shared:]
            if t.get("role") == "assistant"
        )
    if kind == "rlvr":
        sample = (rec.get("rollouts") or {}).get("sample")
        return sample["chars"] if sample else None
    return None  # chat: a log, fit to nothing


# --- reading previous runs back ------------------------------------------


def load(name: str) -> dict | None:
    """A committed result file by name, parsed, or None if this checkout has none.

    Every reader of a previous run goes through here so they all fall back from
    `results/` to the committed `docs/data/` copy the same way.
    """
    path = paths.find(name)
    return json.loads(path.read_text()) if path else None


def context_records(target_name: str, stage: str) -> list[dict]:
    """Every stored example for a stage, labeled or not.

    The whole draw, which is what a stage's *size* has to be estimated from. An
    ask file holds only the rows the classifier answered about — see
    `_label_post_training` — and how long an example is has nothing to do with
    whether a model was willing to judge it.
    """
    data = load(f"{target_name}.{stage}.context.json")
    return data["records"] if data else []


def load_context(target_name: str, stage: str) -> dict:
    """`index_context` over a stage's stored examples, for callers with no use
    for the envelope around them."""
    return index_context(context_records(target_name, stage))


def index_context(records: list[dict]) -> dict:
    """Context records keyed both ways a result record can address one.

    Newer result records carry the row's absolute index; ones committed before
    that join on the first 400 characters of the prompt. Both keys go in the same
    map so a caller does not have to know which vintage it is holding — see
    `context.build`.
    """
    out = {}
    for rec in records:
        if rec.get("row") is not None:
            out[("row", rec["row"])] = rec
        out.setdefault(("key", rec["key"]), rec)
    return out


def context_for(rec: dict, by_key: dict) -> dict | None:
    """The context record a result record addresses, by row if it has one.

    A record that carries a row resolves by row or not at all. Falling back to
    the prompt prefix when its row is absent — two runs at different seeds draw
    different rows — would silently attach the label to whichever example
    happens to open with the same 400 characters, and those collide: 8 of the
    300 sampled Dolci-Think-DPO examples share an opening with another, and a
    chat log is far worse. The prefix key exists only for records written before
    result files carried a row.
    """
    if rec.get("row") is not None:
        return by_key.get(("row", rec["row"]))
    return by_key.get(("key", rec["prompt"][: 400]))


def stage_sources(target_name: str, stage: str) -> dict:
    """A stage's entry from the committed `sources` run, provenance included.

    Offline on purpose: a budget is a rollup of runs that already happened, and
    it should not need the network to add them up. A target with no `sources`
    run yet gets an empty dict and the stage reports its size as unknown rather
    than guessing one.
    """
    data = load(f"{target_name}.sources.json") or {}
    return data.get(stage) or {}


def stage_rows(target_name: str, stage: str) -> int | None:
    """Exact row count of a post-training stage. See `stage_sources`."""
    return stage_sources(target_name, stage).get("total")


# --- the estimate ---------------------------------------------------------


def _rate(matched_chars: float, total_chars: float) -> float:
    return matched_chars / total_chars if total_chars else 0.0


def _size_post_training(target_name: str, stage: dict, ctx: dict, out: dict) -> None:
    """Set a post-training stage's fit-token size, whatever else is known.

    Independent of any ask run, because a stage's size is a property of the
    stage and the pipeline denominator has to be stable. `ask --stage sft` left
    dpo and rlvr unsized, and `totals()` drops an unsized stage — so the
    pipeline share was taken over a denominator that would *grow* the next time
    someone measured something. A share whose denominator grows is not the lower
    bound this output calls it.
    """
    name = stage["stage"]
    # Sizing moved ahead of the ask path so an unasked stage stays in the
    # denominator — which also moved it outside the provenance guards that path
    # runs. An unasked stage reaches no other check, so the stored examples have
    # to be validated here or a context file left over from a repointed dataset
    # would supply fit lengths for one dataset against another's row count.
    if ctx.get("dataset") and ctx["dataset"] != stage["hf_dataset"]:
        out["notes"].append(
            f"stage size unknown — the stored examples are from {ctx['dataset']} but"
            f" this stage now names {stage['hf_dataset']}; re-run"
            f" `trainspotting context {target_name} --stage {name}`"
        )
        return
    if ctx.get("revision_moved_to"):
        out["notes"].append(
            "stage size unknown — the stored examples straddled a republish while"
            " they were drawn, so their lengths describe two trees; re-run"
            f" `trainspotting context {target_name} --stage {name}`"
        )
        return
    all_examples = ctx.get("records", [])
    sample_fit = [f for f in (fit_chars(c) for c in all_examples) if f is not None]
    source = stage_sources(target_name, name)
    rows = source.get("total")
    # The row count comes from a third run, and it can be older than the other
    # two. `context` and `ask` agreeing on a revision says nothing about when
    # `sources` was last taken, and a republish that changes the split's length
    # would multiply this sample's mean by a row count for a different tree. The
    # rate survives that — it is a share, not a count — so a stale source leaves
    # the stage unsized rather than unmeasured.
    src_rev, src_dataset = source.get("revision"), source.get("dataset")
    sample_rev = ctx.get("revision")
    # `cmd_sources` reads /statistics and /info as two requests and stamps
    # `revision_moved_to` when the tree changed between them, because the row
    # count may then describe a different tree than the frequencies. A count
    # that ambiguous cannot size anything, and its starting revision matching
    # the sample's says nothing about which tree it ended on.
    src_straddled = bool(rows is not None and source.get("revision_moved_to"))
    stale_source = src_straddled or bool(rows is not None and (
        (src_rev and sample_rev and src_rev != sample_rev)
        or (src_dataset and src_dataset != stage["hf_dataset"])
    ))
    if src_straddled:
        rows = None
        out["notes"].append(
            "stage size unknown — the `sources` run straddled a republish while it"
            " counted, so its row total is ambiguous; re-run"
            f" `trainspotting sources {target_name} --json`. The rate below is a"
            " share and is unaffected."
        )
    elif stale_source:
        rows = None
        out["notes"].append(
            f"stage size unknown — the `sources` run describes"
            f" {src_dataset or 'another dataset'} at {(src_rev or '?')[:7]} while these"
            f" examples were drawn at {(sample_rev or '?')[:7]}; re-run"
            f" `trainspotting sources {target_name} --json`. The rate below is a share"
            " and is unaffected."
        )
    # Size is a property of the stage, so it is estimated over the whole stored
    # draw rather than over the rows the classifier happened to answer about.
    # Refusals are not random — they land on jailbreak-style prompts, which the
    # README already flags as a biased slice — so letting classifier success
    # decide the mean example length would put that bias into `size_tokens` and
    # every matching-token figure derived from it. The *rate* stays over the
    # labeled subset, which is the only part there is a judgment for.
    if rows is None or not sample_fit:
        if not stale_source and all_examples:
            out["notes"].append(
                f"stage size unknown — run `trainspotting sources {target_name} --json`"
                if rows is None
                else "stage size unknown — no example stores text the model was fit to"
            )
        return
    mean_fit = sum(sample_fit) / len(sample_fit)
    out["rows"] = rows
    out["sized_over"] = len(sample_fit)
    out["mean_fit_chars"] = mean_fit
    out["size_tokens"] = rows * mean_fit / CHARS_PER_TOKEN
    out["size_basis"] = (
        f"{rows:,} rows x {mean_fit:,.0f} mean chars of {out['fit_text']} / {CHARS_PER_TOKEN:g}"
    )
    # An RL stage's real target is the rollouts the policy sampled during
    # training, and the mix ships reference generations instead. One per prompt
    # is the smallest defensible reading of it.
    out["size_is_floor"] = out["kind"] == "rlvr"


def _post_training_stage(target_name: str, stage: dict, slug: str) -> dict:
    name = stage["stage"]
    kind = registry.stage_kind(stage)
    out = {
        "stage": name,
        "name": stage["name"],
        "kind": kind,
        "family": "post-training",
        "dataset": stage["hf_dataset"],
        "fit_text": FIT_TEXT.get(kind, "unknown"),
        "notes": [],
    }
    ctx = load(f"{target_name}.{name}.context.json") or {}
    # Sized first, and regardless of whether the question was ever asked here.
    _size_post_training(target_name, stage, ctx, out)
    ask = load(f"{target_name}.{name}.ask-{slug}.json")
    if not ask:
        out["measured"] = False
        return out
    records = ask["records"]
    # A row index addresses a position in a split, not a document. Ai2 has
    # republished these mixes, and after a republish the same index is different
    # text — so joining an old ask run to a freshly drawn context sample would
    # weigh a match label by an unrelated example's length and size the stage
    # from it. Both stamps have to agree before the join means anything; either
    # being unknown (every run committed before the field existed) is not
    # evidence of a mismatch, so only a known disagreement stops it.
    # The same check the corpus path needs: a stage repointed at another dataset
    # leaves an ask file describing the old one, and its rate would be applied
    # to the new one's row count.
    for who, art in (("ask run", ask), ("stored examples", ctx)):
        if art.get("dataset") and art["dataset"] != stage["hf_dataset"]:
            out["measured"] = False
            out["unusable"] = (
                f"the {who} describes {art['dataset']} but this stage now names"
                f" {stage['hf_dataset']}"
            )
            return out
    ask_rev, ctx_rev = ask.get("revision"), ctx.get("revision")
    # A run that straddled a republish is not addressable by row either, and
    # comparing the two starting revisions cannot see it: both halves can begin
    # at the same tree and cross at different pages. Both producers already
    # detect this and stamp `revision_moved_to` — `_label_post_training` even
    # prints "rows may straddle both" — so the stamp is the answer rather than
    # something to infer. Which rows came from which tree is exactly what is not
    # recorded, so there is no safe subset to keep.
    straddled = [
        who
        for who, art in (("ask run", ask), ("stored examples", ctx))
        if art.get("revision_moved_to")
    ]
    if straddled:
        out["measured"] = False
        out["unusable"] = (
            f"the {' and the '.join(straddled)} straddled a republish while sampling,"
            " so a row index does not address one tree"
        )
        out["notes"].append(
            "re-run both against a settled revision; which rows came from which tree"
            " is not recorded, so no part of the join can be trusted"
        )
        return out
    if ask_rev and ctx_rev and ask_rev != ctx_rev:
        out["measured"] = False
        out["unusable"] = (
            f"the ask run was drawn at {ask_rev[:7]} and the stored examples at"
            f" {ctx_rev[:7]}; a row index means different text across a republish"
        )
        out["notes"].append(
            "re-run `trainspotting context` and `trainspotting ask` against the same"
            " revision to join them"
        )
        return out
    all_examples = ctx.get("records", [])
    by_key = index_context(all_examples)
    joined = [(r, context_for(r, by_key)) for r in records]
    unjoined = sum(1 for _, c in joined if c is None)
    fit = [(r, fit_chars(c)) for r, c in joined if c is not None]
    weighable = [(r, f) for r, f in fit if f is not None]

    k, n = sum(bool(r["match"]) for r in records), len(records)
    total_chars = sum(f for _, f in weighable)
    matched_chars = sum(f for r, f in weighable if r["match"])
    char_rate = _rate(matched_chars, total_chars)
    count_rate = k / n if n else 0.0
    # The interval belongs to the rows the point estimate was actually built
    # from, which is the weighable subset — an example with no stored target
    # text contributes to neither. Taking it over all 300 judged rows when 60
    # carried a weight claims five times the evidence there is: for
    # Dolci-Instruct-RL that is a 5.6% upper bound where the honest one is
    # 13.7%. It also lets a match that never entered the estimate widen it.
    k_w = sum(bool(r["match"]) for r, _ in weighable)
    n_w = len(weighable)
    # Named `rate_ci` and not `count_ci`: it is the interval on the rate this
    # stage reports, over the rows that produced it. `count_rate` beside it is
    # the share of *everything* judged, which is a different denominator — and
    # calling this one "count" while `_apply_rate` pairs it with
    # `weighed_count_rate` invited exactly that confusion.
    lo, hi = wilson(k_w, n_w)

    out.update(
        {
            "measured": True,
            "question": ask["question"],
            # The instrument, not just its words: the site separates ask results
            # by question *and* classifier, and a total assembled from two
            # judges is as mixed as one assembled from two wordings.
            "classifier": ask.get("classifier"),
            "system_sha": ask.get("system_sha"),
            "n": n,
            "matched": k,
            "count_rate": count_rate,
            "rate_ci": [lo, hi],
            "weighed": n_w,
            "weighed_matched": k_w,
            # The rate the interval is anchored on: matches among the rows that
            # carried a weight, over those rows. Distinct from `count_rate`,
            # which is the share of everything judged and is what the table
            # shows as "sampled".
            "weighed_count_rate": k_w / n_w if n_w else 0.0,
            "fit_chars_total": total_chars,
            "fit_chars_matched": matched_chars,
            "char_rate": char_rate,
            # Rows are drawn uniformly, so length weighting is the correction
            # that turns a rate over examples into a rate over training.
            "rate": char_rate,
            "weighting": "fit characters",
        }
    )
    if unjoined:
        out["notes"].append(
            f"{unjoined} of {n} judged records have no stored training example, so the"
            " rate below is weighed over the rest"
        )
    if len(weighable) < len(fit):
        # RL is where this bites: a row with no stored reference generation has
        # no length to weigh by, and for Dolci-Instruct-RL that is most of them.
        # The rate and its interval both come from the rows that do, so say how
        # many that is — a 3.0% headline resting on 60 observations rather than
        # 300 is a very different claim.
        out["notes"].append(
            f"{len(fit) - len(weighable)} of {len(fit)} judged examples store no text the"
            f" model was fit to; the rate and its interval come from the remaining"
            f" {len(weighable)}, while the stage size uses all {len(all_examples)} stored"
            " examples, whose length does not depend on the classifier"
        )

    if not n_w:
        # No judged example carried a usable weight — an empty ask run, or one
        # whose rows do not join to any stored example (a `context` run at a
        # different --seed will do that). A rate of 0 here is absent evidence
        # dressed as a negative result, with a zero-width interval on top of
        # it. The ask card already renders an empty run as having no rate.
        out["unusable"] = (
            "no judged example has a stored target to weigh by"
            if n
            else "the ask run labeled nothing"
        )
        out["measured"] = False
        out["notes"].append(
            f"an ask run exists but produced no usable rate — {out['unusable']};"
            " the stage is left unmeasured rather than counted as zero"
        )
        return out

    _apply_rate(out)
    return out


def _pretrain_stage(target_name: str, stage: dict, slug: str) -> dict:
    name = stage["stage"]
    out = {
        "stage": name,
        "name": stage["name"],
        "kind": "corpus",
        "family": "pretrain",
        "dataset": stage.get("sample_dataset") or stage.get("hf"),
        "fit_text": "every token",
        "size_tokens": stage["tokens"],
        "size_basis": "Olmo 3 paper / release blog",
        "size_is_floor": False,
        "notes": [],
    }
    ask = load(f"{target_name}.{name}.ask-{slug}.json")
    if not ask:
        out["measured"] = False
        return out
    # The rate belongs to the corpus it was measured on, and this stage's token
    # count comes from the registry — which gets repointed. The README records
    # exactly that drift: the 7B stages named the -1125 mixes long after they
    # moved to -1025. Applying an old mix's rate to a new mix's trillions of
    # tokens is the largest single error available here.
    if ask.get("dataset") and out["dataset"] and ask["dataset"] != out["dataset"]:
        out["measured"] = False
        out["unusable"] = (
            f"the ask run sampled {ask['dataset']} but this stage now names"
            f" {out['dataset']}"
        )
        out["notes"].append(
            f"re-run `trainspotting pretrain {target_name} --stage {name}` and the"
            " question against the corpus the registry points at now"
        )
        return out
    records = ask["records"]
    n = len(records)
    if not n:
        # The same guard `_post_training_stage` got, which this path did not.
        # A corpus stage is trillions of tokens: letting an all-refused run
        # report rate 0 with a [0, 0] interval puts a definite, confident zero
        # into the pipeline total across 99.7% of it.
        out["measured"] = False
        out["unusable"] = "the ask run judged no document"
        out["notes"].append(
            "an ask run exists but judged nothing, so the stage is left unmeasured"
            " rather than counted as zero across its whole token budget"
        )
        return out
    k = sum(bool(r["match"]) for r in records)
    # The corpus interval is cluster-corrected and the `ask` run already stored
    # it. Recompute only for a run written before that field existed, and say so
    # rather than silently reporting the narrow one.
    if ask.get("ci"):
        lo, hi = ask["ci"]
    else:
        lo, hi, _ = cluster_wilson(records)
        out["notes"].append("interval recomputed here; the ask run stored none")

    lengths = _doc_lengths(target_name, name, records)
    count_rate = k / n if n else 0.0
    total_chars = sum(lengths)
    matched_chars = sum(c for r, c in zip(records, lengths) if r["match"])
    char_rate = _rate(matched_chars, total_chars)
    # Which weighting is right is a property of how the corpus was *drawn*, not
    # of it being a corpus — the one thing this layer got wrong when a second
    # route arrived. See `_corpus_weighting` and the module docstring.
    rate, weighting, why = _corpus_weighting(stage, count_rate, char_rate, total_chars)
    out.update(
        {
            "measured": True,
            "question": ask["question"],
            "classifier": ask.get("classifier"),
            "system_sha": ask.get("system_sha"),
            "n": n,
            "matched": k,
            "count_rate": count_rate,
            "rate_ci": [lo, hi],
            "weighed": n,
            "rate": rate,
            "weighting": weighting,
            # The estimator on one route and a diagnostic on the other, and
            # stored either way: a large gap between this and the document rate
            # says the matching documents are unusually long or short, which is
            # worth seeing whether or not it is the correction being applied.
            "fit_chars_total": total_chars,
            "fit_chars_matched": matched_chars,
            "char_rate": char_rate,
        }
    )
    if why:
        out["notes"].append(why)
    judged = ask.get("judged_chars", extract.MAX_DOCUMENT_CHARS)
    long_docs = sum(1 for c in lengths if c > judged)
    if long_docs:
        # What is worth saying about a long document is that the judgment read
        # an excerpt of it. What that costs depends on the route: under a shard
        # draw the length is a diagnostic and the document counts once, under a
        # rows draw the same length is the weight the rate is built on, so a
        # mislabeled long document moves the estimate by its whole share.
        out["notes"].append(
            f"{long_docs} of {n} documents are longer than the {judged:,} characters"
            " judged, so their label comes from an excerpt spanning the document"
            " rather than the whole text. "
            + (
                "Each carries its full length into the rate (see `weighting`),"
                " so a long document's label weighs more than a short one's"
                if out["weighting"].startswith("fit characters")
                else "Each still counts once: this corpus's rate is not weighed"
                " by length (see `weighting`)"
            )
        )
    _apply_rate(out)
    return out


# What `weighting` says for a corpus stage, per route. The string is not
# decoration: it is what the report and the site print under the rate, and the
# two routes are estimating the same quantity by different corrections.
_SHARD_WEIGHTING = "none — shards are already drawn proportional to size"
_ROWS_WEIGHTING = "fit characters — rows are drawn uniformly over documents"


def _corpus_weighting(
    stage: dict, count_rate: float, char_rate: float, total_chars: float
) -> tuple[float, str, str]:
    """The rate a corpus stage should multiply its token budget by, and why.

    `registry.sample_route` decides, because the routes make opposite sampling
    guarantees and only one of them makes the document rate a token rate:

    "shards" draws shards with probability proportional to compressed size and
    takes one document from each, so a sampled document already stands for a
    fixed byte mass and the document rate *is* the byte-weighted rate. Weighing
    it by length again would apply the size weighting twice.

    "rows" draws documents uniformly from the whole corpus, so its document rate
    is the share of *documents* that match, not the share of training. The Pile
    is skewed enough for that to matter — its sampled documents run from a few
    hundred characters to 70k — and matches concentrated in unusually long or
    short documents would otherwise carry straight into the matching-token
    total. This is the same correction, and the same estimator, that
    `_post_training_stage` applies to its uniformly-drawn rows.

    A rows sample with no stored lengths falls back to the document rate rather
    than to a zero built out of a division by nothing, and says so: a run written
    before `chars` existed is the case, and none of the committed ones are.
    """
    if registry.sample_route(stage) != "rows":
        return count_rate, _SHARD_WEIGHTING, ""
    if not total_chars:
        return (
            count_rate,
            "document count — no stored lengths to weigh by",
            "this corpus is sampled uniformly over documents, so its rate should be"
            " weighed by document length, but the ask run stored no lengths to weigh"
            " by; the unweighed document rate is reported instead and reads matches"
            " as if every document were the same size",
        )
    return char_rate, _ROWS_WEIGHTING, ""


def _doc_lengths(target_name: str, stage: str, records: list[dict]) -> list[int]:
    """Full length of each judged document, in the order the ask run scored them.

    `ask --pretrain` records a document's length directly. A run from before
    that gets it by joining back to the sample the documents came from, on the
    excerpt text the run stored — which is exactly what was judged, so it is a
    key rather than a prefix of one.
    """
    if all("chars" in r for r in records):
        return [r["chars"] for r in records]
    docs = load(f"{target_name}.{stage}.docs.json")
    by_text = {d["text"]: d["chars"] for d in (docs or {}).get("records", [])}
    return [r.get("chars") or by_text.get(r["prompt"], 0) for r in records]


# Below this many matching examples, the difference between the row rate and
# the length-weighted rate is one or two examples' lengths rather than a
# measured property of the matching content. The rate is still the best estimate
# available; the point is that the reweighting is not independently supported.
FEW_MATCHES = 10


def _apply_rate(out: dict) -> None:
    """Multiply the measured rate by the stage size, and carry the interval."""
    size = out.get("size_tokens")
    if not out.get("measured") or size is None:
        return
    # Over the rows the estimate was built from, and against the rate on the
    # same rows — comparing a length-weighted rate over 60 examples with a
    # count rate over 300 would attribute the difference to length when most of
    # it is the change of denominator.
    base = out.get("weighed_count_rate", out["count_rate"])
    matched = out.get("weighed_matched", out["matched"])
    if 0 < matched < FEW_MATCHES and out["rate"] != base:
        out["notes"].append(
            f"weighing by length rests on {matched} matching example(s), so the gap"
            f" between their share of the weighed rows ({base * 100:.1f}%) and the"
            f" weighed rate ({out['rate'] * 100:.1f}%) is those examples' lengths,"
            " not a measured property of matching content"
        )
    out["matching_tokens"] = out["rate"] * size
    # The interval is over the count rate; rescale it by however much the
    # stage's weighting moved the point estimate, and treat that factor as
    # known. It is not — matching examples being longer is itself measured on
    # 300 draws — so this is narrower than the truth. For a shard-drawn corpus
    # stage the rate *is* the count rate and the factor is exactly 1; for a
    # rows-drawn one it is the length correction, the same as post-training's.
    # Rescale from whatever rate the interval was computed on: the weighable
    # subset for a post-training stage, the document count for a corpus.
    base = out.get("weighed_count_rate", out["count_rate"])
    ratio = out["rate"] / base if base else 1.0
    lo, hi = out["rate_ci"]
    # Clamped to the stage. A handful of matching examples much longer than the
    # rest makes `ratio` large, and a rescaled Wilson endpoint can then run past
    # the stage's whole fit-token count — a bound saying more than 100% of the
    # stage matches, which is not a wide interval but an impossible one.
    out["matching_tokens_ci"] = [
        min(max(x * ratio * size, 0.0), float(size)) for x in (lo, hi)
    ]


def mixing(measured: list[dict]) -> dict:
    """Whether these stages were produced by one instrument, and how they differ.

    What made a number, not just what it was asked. A question reworded between
    runs is two measurements summed into one total, and so is the same question
    put to two different judges.

    The rubric is compared *within* a family, never across one. A corpus
    document is not a request to a model, so `ask --pretrain` scores it under
    `classify.ASK_DOC_SYSTEM` while the post-training stages use `ASK_SYSTEM` —
    two different hashes, by design, on every run of the thing this command
    exists to compute. Treating that expected pair as a conflict would withhold
    the pipeline total from exactly the runs that have one. What is still worth
    catching is a rubric that moved between stages judged the same way.
    """
    variants = sorted({s["question"] for s in measured})
    judges = sorted({s.get("classifier") for s in measured if s.get("classifier")})
    # Only hashes that exist. Every run committed before `system_sha` was
    # stamped records null, so a set holding {null, "abc"} after one stage is
    # re-run would read as two rubrics and withhold the pipeline total from an
    # ordinary incremental `ask --stage ...`. Missing provenance is silence, the
    # same way it is for the classifier and for the revision checks — a conflict
    # needs two distinct hashes that were actually recorded.
    rubrics: dict[str, set] = {}
    for s in measured:
        if s.get("system_sha"):
            rubrics.setdefault(s.get("family"), set()).add(s["system_sha"])
    conflict = sorted(fam for fam, shas in rubrics.items() if len(shas) > 1)
    mixed = len(variants) > 1 or len(judges) > 1 or bool(conflict)
    return {
        "question_variants": variants if mixed else [],
        "classifiers": judges if mixed else [],
        "rubric_conflict": conflict,
        "mixed": mixed,
    }


def estimate(target_name: str, slug: str) -> dict:
    """The whole pipeline's budget for one `ask` question.

    Reads only committed runs, so it costs nothing and answers for exactly the
    samples someone can go and read.
    """
    target = registry.resolve(target_name)
    stages = []
    for stage in registry.pretrain_stages(target):
        stages.append(_pretrain_stage(target_name, stage, slug))
    for stage in registry.post_training_stages(target):
        stages.append(_post_training_stage(target_name, stage, slug))

    # Only stages that produced a rate. A stage whose ask run labeled nothing,
    # or whose rows joined to no example, still records the question and
    # classifier it was invoked with — and letting that decide the mixing
    # would withhold the total from the stages that did measure something,
    # on the word of one that measured nothing.
    rated = [s for s in stages if s.get("measured") and s.get("question")]
    asked = rated or [s for s in stages if s.get("question")]
    question = asked[0]["question"] if asked else None
    mix = mixing(rated)
    variants, judges = mix["question_variants"], mix["classifiers"]
    rubric_conflict, mixed = mix["rubric_conflict"], mix["mixed"]
    return {
        "target": target_name,
        "is_model": target["is_model"],
        "slug": slug,
        "question": question,
        "question_variants": variants if mixed else [],
        # Named separately, because "two wordings", "one wording two judges" and
        # "the rubric moved inside one family" read very differently to someone
        # deciding whether a withheld total was worth withholding.
        "classifiers": judges if mixed else [],
        "rubric_conflict": rubric_conflict,
        "mixed": mixed,
        "chars_per_token": CHARS_PER_TOKEN,
        "stages": stages,
        "totals": totals(stages),
    }


def totals(stages: list[dict]) -> dict:
    """Summed sizes and matching tokens, by family and overall.

    A stage whose size is unknown is summed into neither, and counted in
    `unsized` so the share it reports is visibly a share of what was measured.

    `share` divides by the size of every *sized* stage, measured or not. With a
    stage still unasked that makes it a lower bound on the whole pipeline rather
    than a share of the part that was measured — those are very different
    numbers here, since the corpora are 99.7% of the tokens — so
    `measured_size_tokens` says what the measured part actually was and callers
    print the share as "at least" whenever `measured < stages`.
    """
    out = {}
    for family in ("pretrain", "post-training", "all"):
        rows = [s for s in stages if family == "all" or s["family"] == family]
        sized = [s for s in rows if s.get("size_tokens") is not None]
        priced = [s for s in sized if s.get("matching_tokens") is not None]
        size = sum(s["size_tokens"] for s in sized)
        matching = sum(s["matching_tokens"] for s in priced)
        out[family] = {
            "size_tokens": size,
            "measured_size_tokens": sum(s["size_tokens"] for s in priced),
            "matching_tokens": matching,
            "share": matching / size if size else 0.0,
            "matching_tokens_ci": [
                sum(s["matching_tokens_ci"][i] for s in priced) for i in (0, 1)
            ],
            "stages": len(rows),
            "measured": sum(1 for s in rows if s.get("measured")),
            "unsized": [s["stage"] for s in rows if s.get("size_tokens") is None],
            # A stage counted at its floor makes the total a floor too.
            "floor": [s["stage"] for s in priced if s.get("size_is_floor")],
        }
    return out
