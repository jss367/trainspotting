"""How much of a preference pair's label is readable without reading the pair.

A DPO stage teaches by contrast: two answers to one prompt, one marked chosen
and one rejected, and the loss pushes the policy toward the first and away from
the second. What it teaches therefore depends on *what distinguishes* the two
sides — and two things distinguish them that have nothing to do with the answer:

    length      one side is longer, and the preference agrees with the longer
                side far more often than a coin would
    provenance  one side came from a bigger model than the other, so the
                preference tracks the generator rather than the response

Both are shortcuts a policy can fit instead of the preference. "Chosen responses
are generally longer than rejected ones" is the finding of a 14-dataset audit
(Singhal et al., arXiv:2407.01085), and the same asymmetry is how sycophancy
survives preference tuning (arXiv:2602.01002). Neither audit covers Dolci, and
nothing upstream publishes these numbers for it.

This measures what is *available* to be fit, not what was fit. A stage where the
longer side wins 77% of the time is a stage where a rule that never reads the
text gets 77% right; whether the trained policy took that rule is a question
about the model, which nothing here touches. And on Dolci the availability is
partly deliberate: every sampled pair carries `preference_type:
delta_learning`, a recipe that pairs a strong generator against a weak one on
purpose, so the model asymmetry below is the method working rather than a defect
in it. The length gap is what that method costs.

Everything is computed from the committed `context` records — no network, no API
key, no model. That is possible because a context record stores each field's
*true* length beside the 4,000 characters it keeps for display
(`context._text`), so the lengths here are the real ones and only the text is
cut.
"""

from trainspotting import derive
from trainspotting.stats import cluster_wilson

# Metadata columns worth breaking the length rule down by. A column with one
# value splits nothing, and one with hundreds turns the breakdown into the
# sample itself at one row per cell. The committed samples sit well inside that
# range — 24 `dataset_source` values on Dolci-Think-DPO-7B, 4 `preference_type`
# values on Dolci-Instruct-DPO — so the cap is there to keep a republished mix
# with a high-cardinality column from writing a megabyte of singletons.
MAX_BREAKDOWN_VALUES = 40

# The columns a shortened cell has to land in before it can shorten a number
# here. Deliberately narrower than `search.COLUMNS["dpo"]`, which also names
# `prompt`: that column is the question both sides answer, and a DPO record
# builds its two sides out of the `chosen` and `rejected` cells alone
# (`context.build`), so every length below is computed without reading `prompt`
# at all. A row the server cut only there arrived whole as far as this module is
# concerned, and counting it would print "these lengths are lower bounds" over a
# sample whose two measured sides are complete. `search` asks a different
# question — which cells a *search* would have had to read — and gets a wider
# answer to it, so this set is written out here rather than derived from that
# one.
MEASURED_COLUMNS = ("chosen", "rejected")


def _sides(rec: dict) -> tuple[list[dict], list[dict]]:
    """The turns of each side that carry gradient, shared history removed.

    `derive._shared_turns` is the same cut `budget` makes, and it has to be the
    same one: a multi-turn pair's leading turns are the conversation both
    candidates answer in, identical on both sides, and counting them would add
    the same characters to both sides of every comparison below. That inflates
    each side's length by the history and drags the *difference* toward zero,
    which is the one number this module exists to report.
    """
    chosen = (rec.get("chosen") or {}).get("turns") or []
    rejected = (rec.get("rejected") or {}).get("turns") or []
    shared = derive._shared_turns(chosen, rejected)
    return chosen[shared:], rejected[shared:]


def _reasoning_chars(turn: dict) -> int:
    return ((turn.get("reasoning") or {}).get("chars")) or 0


def _answer_chars(turn: dict) -> int:
    """The turn's length less its thinking span.

    Not `turn["chars"]` on its own: a think model stores the reasoning beside
    the answer rather than inside it, and `chars` is the answer alone — but a
    record written for a turn that was split also carries `chars_raw`, the
    length before the split, which is the only field that accounts for the
    `<think>` markers. Subtracting keeps the two halves summing to the whole, so
    a stage's answer and reasoning totals add up to its length.
    """
    return derive._turn_chars(turn) - _reasoning_chars(turn)


def _signed(values: list[int], rows: list[int | None] | None) -> dict:
    """`derive.summarize` for a quantity that can be negative.

    The mean, its clustered error and the quantiles all carry over unchanged.
    The histogram does not: `derive.histogram` bins by log10 and silently drops
    anything at or below zero, which here is every pair the *rejected* side won
    — the half of the distribution a length audit is least entitled to lose. So
    it is replaced by two histograms of the magnitude, one per direction, on the
    same shared bins as every other length chart on the site.
    """
    summary = derive.summarize(values, rows)
    if not summary.get("n"):
        return summary
    summary["hist_chosen"] = derive.histogram([v for v in values if v > 0])
    summary["hist_rejected"] = derive.histogram([-v for v in values if v < 0])
    summary.pop("hist", None)
    return summary


def _rate(flags: list[bool], groups: list[list[int]] | None) -> dict:
    """A share of the pairs, with the interval the sample's clustering supports.

    The sampler draws pages of ten adjacent rows, and rows adjacent on disk come
    from the same source dataset and often the same generation run — so they
    agree with each other about length more than two independent rows would, and
    the binomial interval over 1,000 rows would be the interval for a sample this
    one is not. `stats.cluster_wilson` takes the design effect out of the draws
    themselves; without recorded row indices there are no draws to take it from,
    and the records are then treated as independent, which is the assumption the
    committed samples were read under before they carried an index.
    """
    n = len(flags)
    if not n:
        return {"k": 0, "n": 0}
    by_cluster = {}
    if groups:
        for c, positions in enumerate(groups):
            for i in positions:
                by_cluster[i] = str(c)
    records = [{"match": f, "shard": by_cluster.get(i, str(i))} for i, f in enumerate(flags)]
    lo, hi, n_eff = cluster_wilson(records)
    k = sum(flags)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi, "n_eff": n_eff,
            "clusters": len(groups) if groups else None}


def _model_rule(pairs: list[tuple[str, str]]) -> dict | None:
    """How often the best rule that reads only the two generator names is right.

    For each unordered pair of models, the rule answers with whichever direction
    the stage picks more often; its accuracy is the sum of those majorities over
    the pairs it can answer. It is fitted on the same rows it is scored on, so
    this is an in-sample ceiling and not a generalization estimate — with one
    ordered pairing throughout a stage it is exactly 1.0, and that is a fact
    about the mix rather than a trained result. `fitted: true` travels with it
    so no reader has to remember.

    A pair whose two sides came from the same model is unanswerable by
    construction and is excluded from the denominator rather than scored as a
    coin flip, which would drag the number toward 0.5 in proportion to how often
    a mix samples a model against itself.
    """
    decided = [(c, r) for c, r in pairs if c and r and c != r]
    if not decided:
        return None
    directions: dict[frozenset, dict[tuple, int]] = {}
    for c, r in decided:
        directions.setdefault(frozenset((c, r)), {}).setdefault((c, r), 0)
        directions[frozenset((c, r))][(c, r)] += 1
    right = sum(max(counts.values()) for counts in directions.values())
    return {
        "k": right,
        "n": len(decided),
        "rate": right / len(decided),
        "fitted": True,
        "same_model": sum(1 for c, r in pairs if c and r and c == r),
        "unknown": sum(1 for c, r in pairs if not c or not r),
        "matchups": len(directions),
    }


def _breakdown(records: list[dict], flags: list[bool], rows: list[int | None]) -> dict:
    """The length rule again, per value of each metadata column the rows carry.

    A stage-wide 65% can be one mix at 90% and another at 50%, and those are
    different findings: the first says one source is the whole effect, the second
    says the preference really does track length throughout. The stage's own
    provenance columns are the split, so this lines up with what `sources` counts
    and what the crosstab on the site already groups by.

    `records`, `flags` and `rows` are the *decided* pairs only — the same subset
    `length_rule` is computed over — and they have to be, because this is the
    headline rate again per group and a row counted in one denominator but not
    the other makes the two numbers answers to different questions. Handed every
    pair instead, a tie would arrive as `False` and be scored as a group's
    failure to prefer the longer side, which drags exactly those groups that
    contain ties below the stage rate for a reason no reader could see: on the
    committed Instruct sample that is 16 of 1,000 rows, enough to move a small
    group several points. The three lists are indexed together, so any subset
    passed here must be taken from all three at once.
    """
    columns: dict[str, dict[str, list[int]]] = {}
    for i, rec in enumerate(records):
        for key, value in (rec.get("meta") or {}).items():
            if value in (None, ""):
                continue
            columns.setdefault(key, {}).setdefault(str(value), []).append(i)
    out = {}
    for key, values in sorted(columns.items()):
        if len(values) < 2 or len(values) > MAX_BREAKDOWN_VALUES:
            continue
        out[key] = {
            value: {
                **_rate([flags[i] for i in positions],
                        derive.clusters_of([rows[i] for i in positions])),
            }
            # Largest first: the tail of a provenance column is singletons whose
            # rate is 0% or 100% on one row, and sorting by name puts them
            # wherever the alphabet does.
            for value, positions in sorted(values.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        }
    return out


def _matchups(records: list[dict]) -> list[dict]:
    """Ordered (chosen model, rejected model) counts, commonest first.

    Ordered rather than unordered: which of two models is the preferred one is
    the whole content of this table. The think mixes have exactly one row here —
    qwen3-reasoning-32b over qwen3-reasoning-0.6b, every pair — and collapsing
    the direction would render that as "these two models appear together",
    which is the observation minus the finding.
    """
    counts: dict[tuple, int] = {}
    for rec in records:
        key = ((rec.get("chosen") or {}).get("model"), (rec.get("rejected") or {}).get("model"))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"chosen": c, "rejected": r, "n": n}
        for (c, r), n in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    ]


def stage_pairs(ctx: dict) -> dict:
    """Everything derivable about the preference pairs in one committed context run.

    `ctx` is a parsed `<target>.<stage>.context.json` whose records are `dpo`
    ones. A stage of any other kind has no pairs and raises rather than
    returning zeros, because a zero here reads as "no length asymmetry" and the
    honest answer for an SFT stage is that the question does not apply.
    """
    records = [r for r in (ctx.get("records") or []) if r.get("kind") == "dpo"]
    if not records:
        raise ValueError(f"{ctx.get('dataset')} {ctx.get('stage')}: no preference pairs to measure")

    rows = [r.get("row") for r in records]
    groups = derive.clusters_of(rows)

    chosen_chars, rejected_chars, deltas = [], [], []
    chosen_reasoning, rejected_reasoning = [], []
    chosen_answer, rejected_answer = [], []
    degenerate, has_reasoning = 0, False
    for rec in records:
        chosen, rejected = _sides(rec)
        c = sum(derive._turn_chars(t) for t in chosen)
        r = sum(derive._turn_chars(t) for t in rejected)
        chosen_chars.append(c)
        rejected_chars.append(r)
        deltas.append(c - r)
        # Both sides swallowed whole by the shared prefix, which is the same
        # statement as "the two completions are identical": `_shared_turns` only
        # advances over turns that match, so two empty remainders mean the two
        # turn lists were equal. DPO reads the difference of the two sides' log
        # probabilities, so such a pair cancels exactly and trains nothing — 12
        # of the sampled Instruct pairs are like this. They are still pairs in
        # the mix and still counted below; naming them separately is what keeps a
        # tie between two empty sides from reading as a tie between two answers.
        #
        # `and`, not `or`: one empty side against a non-empty one is a pair whose
        # chosen (or rejected) completion is nothing at all. That is a real
        # length difference the rule above should score, not a cancelling pair,
        # and counting it here would attach "the loss cancels, they train
        # nothing" to a row where neither clause is true.
        if not chosen and not rejected:
            degenerate += 1
        chosen_reasoning.append(sum(_reasoning_chars(t) for t in chosen))
        rejected_reasoning.append(sum(_reasoning_chars(t) for t in rejected))
        chosen_answer.append(sum(_answer_chars(t) for t in chosen))
        rejected_answer.append(sum(_answer_chars(t) for t in rejected))
        has_reasoning = has_reasoning or bool(chosen_reasoning[-1] or rejected_reasoning[-1])

    longer = [d > 0 for d in deltas]
    ties = sum(1 for d in deltas if d == 0)
    # The rule's denominator is the pairs it can answer. A tie is not a wrong
    # answer from a length rule, it is no answer — counting ties as failures
    # would report a mix of identical-length pairs as one where length predicts
    # nothing, when what it predicts is undefined there. Both denominators are
    # written out so a reader can see how much the choice moved the number:
    # 16 of the 1,000 sampled Instruct pairs tie, which moves it by 1.1 points.
    decided = [i for i, d in enumerate(deltas) if d != 0]
    out = {
        "dataset": ctx.get("dataset"),
        "stage": ctx.get("stage"),
        "revision": ctx.get("revision"),
        "sample": ctx.get("sample"),
        "seed": ctx.get("seed"),
        "n": len(records),
        "ties": ties,
        "degenerate": degenerate,
        "longer_chosen": _rate(longer, groups),
        "length_rule": _rate(
            [longer[i] for i in decided], derive.clusters_of([rows[i] for i in decided])
        ),
        "delta": _signed(deltas, rows),
        "chars": {
            "chosen": derive.summarize(chosen_chars, rows),
            "rejected": derive.summarize(rejected_chars, rows),
        },
        "models": {
            "rule": _model_rule(
                [((r.get("chosen") or {}).get("model"), (r.get("rejected") or {}).get("model"))
                 for r in records]
            ),
            "matchups": _matchups(records),
        },
        # The decided pairs only, so a group's rate here is the headline rate
        # restricted to that group rather than a different statistic wearing the
        # same label. See `_breakdown`.
        "by": _breakdown(
            [records[i] for i in decided],
            [longer[i] for i in decided],
            [rows[i] for i in decided],
        ),
    }
    # Where the extra length is. A think stage's completions are mostly
    # thinking — the sampled 32B chosen sides average 7,649 characters of
    # reasoning against 2,363 of answer — so "the chosen side is 1,900
    # characters longer" would read as a statement about the reasoning budget
    # unless the two are reported apart. They are not the same story: the 32B
    # gap splits 961 characters of reasoning to 938 of answer, so the answers
    # differ in length about as much as the thinking does.
    if has_reasoning:
        out["split"] = {
            "reasoning": _signed([c - r for c, r in zip(chosen_reasoning, rejected_reasoning)], rows),
            "answer": _signed([c - r for c, r in zip(chosen_answer, rejected_answer)], rows),
        }
    # The upstream cut. The datasets-server shortens a very large cell to fit its
    # response limit, and a shortened cell arrives with fewer characters than it
    # has — so a length measured over one is short by an unknown amount, and the
    # cuts land on the longest cells by construction, which is the side this
    # module is measuring. Only a cut in `MEASURED_COLUMNS` counts: a row whose
    # `prompt` cell was shortened has both completions whole, and every number
    # above is computed from the completions alone. A run that was not looking
    # reports null rather than zero: "no rows were cut" and "nobody checked" are
    # different states, and collapsing them would put a clean bill of health on
    # every sample committed before `context` learned to check.
    out["truncated_rows"] = (
        sum(1 for r in records if set(r.get("truncated") or ()) & set(MEASURED_COLUMNS))
        if ctx.get("truncation_recorded")
        else None
    )
    return out
