"""Numbers derived from samples that are already committed — no network, no API key.

Three questions the per-stage layers cannot answer on their own, all of them
answerable from the context and document samples the site already ships:

    budget    how big each stage is in tokens, on one axis with every other
              stage — which is the only way to see that everything after
              pretraining is a rounding error of what the model read
    shape     how long one training example is, per stage, so a count of
              examples can be read against the text behind it
    crosstab  which source dataset (or domain, or verifier mix) each sampled
              prompt came from, so a taxonomy label can be traced to where in
              the mix it comes from

None of this is measured by anyone upstream. Ai2 publishes token counts for the
pretraining mixes and row counts for the post-training ones, and those two are
not comparable: 2.1M SFT examples is not a token budget. So the post-training
side is *estimated* here from the sampled examples — mean characters per example
over the committed sample, divided by CHARS_PER_TOKEN, times the exact row count
from the datasets-server. The estimate carries the sample's own standard error,
and every consumer is expected to show it as an estimate.

The divisor is the weak part and it is deliberately one number: real tokenizers
run about 3.5 characters per token on code and 4.5 on English prose, and no
choice inside that range changes the finding, which is a factor of about 10,000.
"""

import math

from trainspotting import hf
from trainspotting.context import KEY_CHARS

# The one assumption in the token estimates. Stated here, restated on the site.
CHARS_PER_TOKEN = 4

# Half-decade bins from 10 characters to ~3.2M, shared by every stage and every
# corpus so the histograms can be read as small multiples against one axis. A
# pretraining document and an RL prompt differ by three orders of magnitude, so
# linear bins would put each stage in a single column of its own chart.
HIST_MIN_LOG = 1.0
HIST_MAX_LOG = 6.5
HIST_STEP = 0.5
HIST_BINS = round((HIST_MAX_LOG - HIST_MIN_LOG) / HIST_STEP)


def hist_edges() -> list[float]:
    """The bin boundaries in characters, for a consumer that has to label them."""
    return [10 ** (HIST_MIN_LOG + i * HIST_STEP) for i in range(HIST_BINS + 1)]


def prompt_key(prompt: str) -> str:
    """A short stable id for a sampled prompt, hashed from the same prefix
    `context.build` keys on.

    The join it serves is the site's existing one: a label record holds a prompt
    and no row index, so the only thing it shares with a context record is the
    opening of the prompt. Carrying that opening again in a third file would
    double the payload of the page for text already on it, so it travels as a
    hash — 32-bit FNV-1a over Unicode code points, which JavaScript can compute
    the same way over the same prefix.

    Two rows that open with the same 400 characters collapse onto one key. That
    is the same ambiguity `ctxMaps.byKey` has always had (rare in a curated mix,
    routine in a chat log), so the profile records how many keys are ambiguous
    rather than pretending the join is exact.
    """
    h = 0x811C9DC5
    for ch in prompt[:KEY_CHARS]:
        h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _turn_chars(turn: dict) -> int:
    """How many characters the turn actually was.

    The two stored halves are an answer and the reasoning span split off in
    front of it, and the split drops what sat between them — the `<think>`
    tags and the whitespace around them, which the model read. `chars_raw` is
    the length before that split, so it is the honest measure where a record
    carries one; records written before it fall back to the sum, which
    understates a reasoning turn by the markup (0.04% to 0.16% of the Think
    stages' totals, measured).
    """
    if turn.get("chars_raw") is not None:
        return turn["chars_raw"]
    return turn.get("chars", 0) + (turn.get("reasoning", {}) or {}).get("chars", 0)


def _shared_turns(chosen: list[dict], rejected: list[dict]) -> int:
    """How many leading turns a preference pair has in common.

    A multi-turn pair shares the conversation up to the point it branches,
    assistant turns included. Counting those turns on both sides would count the
    shared history twice and call it text the model was fit to, which is exactly
    backwards — the shared part is context, and only what comes after the branch
    carries the preference signal.

    `search._shared_turns` is the same scan over raw rows and deliberately does
    NOT agree with this one at the edges: it clamps so that a pair whose sides
    are identical still has two completions, because search asks which side a
    string is on and the honest answer there is "both". This asks which tokens
    carry gradient, and for that pair the answer is none. Two layers, two
    questions, two right answers — so do not reconcile them.
    """
    # Every part of a turn a reader can see, because every part of it is text
    # the pair either shares or branches on. A turn's reasoning span is stored
    # beside its answer rather than inside it, so comparing the answer and the
    # combined length alone calls two turns identical when they reason
    # differently at the same length toward the same answer — and then counts
    # one copy of a turn that is really two, with neither reasoning span as a
    # target.
    def part_same(a: dict, b: dict) -> bool:
        """One field of a turn: its length, and its content as far as the record
        proves it. A field cut for display carries a digest of the whole thing
        (`context._text`), so two long fields agreeing on their first 4,000
        characters are compared on the digest rather than on the prefix — the
        prefix alone would call two 4,001-character responses identical when
        only their last character differs, and the pair would come back with no
        target at all. Where one side has a digest and the other does not, the
        record predates the digest and the prefix is all there is; the honest
        comparison is then the one that was possible when it was written.
        """
        if a.get("chars") != b.get("chars"):
            return False
        if a.get("sha") and b.get("sha"):
            return a["sha"] == b["sha"]
        return a.get("text") == b.get("text")

    def same(a: dict, b: dict) -> bool:
        return (
            a.get("role") == b.get("role")
            and part_same(a, b)
            and part_same(a.get("reasoning") or {}, b.get("reasoning") or {})
        )

    n = 0
    for a, b in zip(chosen, rejected):
        if not same(a, b):
            break
        n += 1
    # No clamp. A side swallowed whole by the shared prefix really does have no
    # completion, and that is the answer rather than a degenerate case to work
    # around: DPO's loss reads the difference between the two sequences' log
    # probabilities, so a pair whose sides are identical cancels exactly and
    # carries no gradient at all. Backing the prefix up a turn to manufacture a
    # branch would report two copies of the same answer as gradient-bearing.
    # Four sampled Instruct DPO pairs are identical in full.
    #
    # The same holds one step weaker for a side that is a strict prefix of the
    # other: everything up to where the shorter side ends is conditioned
    # identically and cancels, and the difference is exactly the turns only the
    # longer side has.
    return n


def example_chars(rec: dict) -> tuple[int, int]:
    """(every character in the example, the characters the model is fit to).

    "Fit to" is the gradient-bearing text, which is a different fraction of the
    example in every stage and is the whole reason a token budget by stage is
    not the same chart as a token budget by what training did with those tokens:

        pretrain/midtrain/long-context  every token is a next-token target
        sft       the assistant turns; the prompt and any tool output are read,
                  not scored
        dpo       both completions after the branch — the loss reads the chosen
                  and the rejected side, pushing in opposite directions
        rlvr      nothing. The response is generated during training and never
                  stored, so the only text in the file is the prompt, and none
                  of it is a target
        chat      nothing. It is a log; no model was fit to any of it
    """
    kind = rec.get("kind")
    if kind in ("sft", "chat"):
        turns = rec.get("turns") or []
        total = sum(_turn_chars(t) for t in turns)
        if kind == "chat":
            return total, 0
        return total, sum(_turn_chars(t) for t in turns if t.get("role") == "assistant")
    if kind == "dpo":
        chosen = (rec.get("chosen") or {}).get("turns") or []
        rejected = (rec.get("rejected") or {}).get("turns") or []
        shared = _shared_turns(chosen, rejected)
        prefix = sum(_turn_chars(t) for t in chosen[:shared])
        target = sum(_turn_chars(t) for t in chosen[shared:]) + sum(
            _turn_chars(t) for t in rejected[shared:]
        )
        return prefix + target, target
    # RL: the prompt is the whole stored example.
    return (rec.get("prompt_full") or {}).get("chars", 0), 0


def histogram(values) -> list[int]:
    """Counts per shared log bin. Anything longer than the top edge lands in the
    last bin — the alternative is dropping the tail, and the tail is where the
    long-context documents are."""
    bins = [0] * HIST_BINS
    for v in values:
        if v <= 0:
            continue
        i = int((math.log10(v) - HIST_MIN_LOG) / HIST_STEP)
        bins[min(max(i, 0), HIST_BINS - 1)] += 1
    return bins


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    i = (len(sorted_values) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (i - lo)


def clusters_of(rows: list[int | None]) -> list[list[int]] | None:
    """The sampler's fetches, recovered from the row indices it kept.

    `hf.sample_rows_with_truncation` draws random offsets and takes ten
    consecutive rows from each, because rows adjacent on disk are correlated —
    the same source dataset, often the same generation run. So a 300-row sample
    is about thirty draws, not three hundred, and the maximal runs of
    consecutive indices are those draws. Two offsets landing next to each other
    merge into one longer run, which costs a little precision and claims
    nothing false.

    A draw does not always survive whole: `cmd_context` drops a fetched row
    whose prompt will not extract, which leaves a gap in the middle of a page
    and would split it into two runs that are not two draws. WildChat is the
    committed example — rows 509532-509535 and 509537-509541 are one fetch of
    ten with 509536 missing — and counting them as two clusters reported 31
    draws instead of 30 and narrowed the interval accordingly. Two rows from
    one page are at most `hf.CHUNK - 1` apart, so that is the gap a run
    tolerates. Two genuinely separate offsets landing that close merge instead,
    which is rare on a large split and errs toward the wider interval.

    Returns positions grouped by run, or None when the records carry no row
    index (runs committed before the sampler recorded one).
    """
    if any(r is None for r in rows):
        return None
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and rows[i] - rows[groups[-1][-1]] < hf.CHUNK:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _cluster_se(values: list[float], mean: float, groups: list[list[int]]) -> tuple[float, float]:
    """(standard error, design effect) for a mean over clustered draws.

    The same Taylor-linearised ratio variance `cli._cluster_wilson` uses for a
    rate, with a cluster's sum of values where that one has its match count:

        Var(mean) = C / ((C - 1) · M²) · Σ (S_c - mean · m_c)²

    Dividing by the independent variance gives the design effect, which is what
    makes the widening legible — an interval that is simply wider looks like a
    smaller sample rather than a correlated one.
    """
    n = len(values)
    C = len(groups)
    var_ind = sum((v - mean) ** 2 for v in values) / (n - 1) / n if n > 1 else 0.0
    if C < 2:
        # One cluster leaves the design effect unestimable, and there is no
        # honest number to put here. The cluster variance would be a zero-width
        # interval presented as certainty; the independent error would assume
        # the rows are independent draws, which is the assumption this whole
        # correction exists to deny — a `--sample 10` run is one fetch of ten
        # adjacent rows, and they are correlated by construction. So: no error,
        # and the consumers report no interval rather than a wrong one.
        return None, 1.0
    ss = sum((sum(values[i] for i in g) - mean * len(g)) ** 2 for g in groups)
    var_cluster = C / ((C - 1) * n**2) * ss
    # The floor belongs on the error, not just on the number reported beside it.
    # Clustering costs precision or costs nothing; a sample whose draws happen to
    # look unalike has not *bought* precision, and returning the raw cluster
    # variance where it lands below the independent one narrowed two committed
    # profiles by about 7% under a heading that says "widened". Deriving the
    # error from the floored design effect also keeps the two consistent, which
    # they were not: the page could report deff 1.0 beside an interval that had
    # been quietly narrowed.
    deff = max(1.0, var_cluster / var_ind) if var_ind else 1.0
    return math.sqrt(var_ind * deff), deff


def summarize(values: list[int], rows: list[int | None] | None = None) -> dict:
    """Mean with its standard error, the quantiles, and the shared-bin histogram.

    The error is the clustered one wherever the row indices allow it, because
    the sample is not 300 independent draws — see `clusters_of`. Treating it as
    independent makes every interval on the page too narrow, by about 2x on
    these samples.
    """
    n = len(values)
    if not n:
        return {"n": 0}
    ordered = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    se, deff, groups = math.sqrt(var / n), 1.0, clusters_of(rows) if rows is not None else None
    if groups:
        se, deff = _cluster_se(values, mean, groups)
    # Degrees of freedom for the interval built on that error: the number of
    # independent things it was estimated from, less one. Without clusters that
    # is the rows, which is the pre-cluster assumption and only right when the
    # rows really were drawn independently.
    df = (len(groups) if groups else n) - 1
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "df": df,
        "deff": deff,
        "clusters": len(groups) if groups else None,
        "p10": _quantile(ordered, 0.10),
        "median": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "hist": histogram(values),
    }


# Two-sided 95% critical values of Student's t, by degrees of freedom. The
# error these multiply is *estimated* — from about thirty fetch clusters, not
# from a known variance — so the normal 1.96 states a confidence the sample does
# not support. At 29 degrees of freedom it is 2.045, which is 4% wider; at 1 it
# is 12.7, and a sample small enough to hit that end is exactly where pretending
# otherwise does the most damage.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
       22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
       29: 2.045, 30: 2.042}
Z95 = 1.96


def t95(df: int) -> float:
    """The 95% critical value at `df`, tabulated where it matters and
    approximated past the table, where t is within a fraction of a percent of
    the Cornish-Fisher expansion around z."""
    if df < 1:
        return float("inf")
    if df in T95:
        return T95[df]
    return Z95 + (Z95**3 + Z95) / (4 * df)


def _estimate(stats: dict, examples: int | None) -> dict | None:
    """Total tokens for a stage nobody published one for.

    mean characters per example / CHARS_PER_TOKEN x the exact row count, with
    the sample's own 95% interval on the mean carried through. The interval
    covers sampling error only — it says nothing about the divisor, which is the
    larger uncertainty and is why this is labeled an estimate wherever it shows.
    """
    if not stats.get("n") or not examples:
        return None
    per = stats["mean"] / CHARS_PER_TOKEN
    out = {"tokens": per * examples, "per_example": per}
    # A sample that came back as one fetch has no estimable sampling error, so
    # it gets a total and no interval rather than an interval that assumes what
    # the correction denies.
    if stats.get("se") is None:
        return out
    half = t95(stats.get("df") or 0) * stats["se"] / CHARS_PER_TOKEN
    out["lo"] = max(0.0, per - half) * examples
    out["hi"] = (per + half) * examples
    return out


def stage_profile(ctx: dict, examples: int | None = None) -> dict:
    """Everything derivable from one committed context run.

    `examples` is the stage's exact row count (from the sources layer, which
    counts rather than samples). Without it the shape of an example is still
    measurable; the token budget is not.
    """
    records = ctx.get("records") or []
    totals, targets, out, columns = [], [], [], {}
    seen, ambiguous = set(), 0
    for rec in records:
        total, target = example_chars(rec)
        totals.append(total)
        targets.append(target)
        key = prompt_key(rec.get("key", ""))
        if key in seen:
            ambiguous += 1
        seen.add(key)
        meta = {k: v for k, v in (rec.get("meta") or {}).items() if v not in (None, "")}
        for k, v in meta.items():
            columns.setdefault(k, set()).add(str(v))
        # The row index travels too. It costs a couple of kilobytes a stage and
        # it is the only identity on this page that proves rather than infers:
        # a languages run records the same indices, so the two can be compared
        # outright instead of through a hash of a prompt's opening.
        rec_out = {"k": key, "m": {k: str(v) for k, v in meta.items()}}
        if rec.get("row") is not None:
            rec_out["row"] = rec["row"]
        out.append(rec_out)

    rows = [rec.get("row") for rec in records]
    stats = summarize(totals, rows)
    target_stats = summarize(targets, rows)
    return {
        "dataset": ctx.get("dataset"),
        "stage": ctx.get("stage"),
        # What was asked for against what survived. A fetched row whose prompt
        # will not extract is dropped, and dropping is not random — it follows
        # from the example's structure — so a total that scales the survivors'
        # mean across every row in the dataset is assuming the ones that could
        # not be read are like the ones that could. WildChat is the committed
        # case at 299 of 300. Small, and stated rather than assumed.
        "requested": ctx.get("sample"),
        "retained": len(records),
        "kind": records[0].get("kind") if records else None,
        "sample": ctx.get("sample"),
        "seed": ctx.get("seed"),
        "examples": examples,
        "chars_per_token": CHARS_PER_TOKEN,
        "chars": stats,
        "target_chars": target_stats,
        "tokens": _estimate(stats, examples),
        "target_tokens": _estimate(target_stats, examples),
        # Which metadata columns these rows actually carry, and how many values
        # each takes. A stage whose rows carry none (some mixes ship no source
        # column at all) has nothing to cross-tabulate, and the site says so
        # rather than drawing an empty grid.
        "columns": {k: len(v) for k, v in sorted(columns.items())},
        # The join is a prompt prefix, so say how much of it is ambiguous.
        "ambiguous_keys": ambiguous,
        "records": out,
    }


def corpus_lengths(docs: dict) -> dict:
    """The same shape measurement for a pretraining corpus sample.

    Lengths only, on the same bins: a corpus stage needs no token estimate,
    because the paper publishes one. What it has no published number for is how
    long one document is, which is what makes "2.1M SFT examples" and "5.93T
    pretraining tokens" comparable quantities at all.
    """
    lengths = [r.get("chars") or len(r.get("text") or "") for r in docs.get("records") or []]
    return {"chars_per_token": CHARS_PER_TOKEN, "chars": summarize(lengths)}
