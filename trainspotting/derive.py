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
    """A turn is its text plus the reasoning span split off in front of it."""
    return turn.get("chars", 0) + (turn.get("reasoning", {}) or {}).get("chars", 0)


def _shared_turns(chosen: list[dict], rejected: list[dict]) -> int:
    """How many leading turns a preference pair has in common.

    A multi-turn pair shares the conversation up to the point it branches,
    assistant turns included. Counting those turns on both sides would count the
    shared history twice and call it text the model was fit to, which is exactly
    backwards — the shared part is context, and only what comes after the branch
    carries the preference signal.
    """
    # Every part of a turn a reader can see, because every part of it is text
    # the pair either shares or branches on. A turn's reasoning span is stored
    # beside its answer rather than inside it, so comparing the answer and the
    # combined length alone calls two turns identical when they reason
    # differently at the same length toward the same answer — and then counts
    # one copy of a turn that is really two, with neither reasoning span as a
    # target.
    def same(a: dict, b: dict) -> bool:
        return (
            a.get("role") == b.get("role")
            and a.get("chars") == b.get("chars")
            and a.get("text") == b.get("text")
            and (a.get("reasoning") or {}).get("chars") == (b.get("reasoning") or {}).get("chars")
            and (a.get("reasoning") or {}).get("text") == (b.get("reasoning") or {}).get("text")
        )

    n = 0
    for a, b in zip(chosen, rejected):
        if not same(a, b):
            break
        n += 1
    # A side swallowed whole by the shared prefix is not a side with no
    # completion — a pair whose completions happen to be identical, or whose
    # shorter side is a prefix of the longer, would otherwise come back with no
    # target at all. Leave each side its last turn.
    return min(n, len(chosen) - 1, len(rejected) - 1) if n else 0


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


def summarize(values: list[int]) -> dict:
    """Mean with its standard error, the quantiles, and the shared-bin histogram."""
    n = len(values)
    if not n:
        return {"n": 0}
    ordered = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "se": math.sqrt(var / n),
        "p10": _quantile(ordered, 0.10),
        "median": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "hist": histogram(values),
    }


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
    half = 1.96 * stats["se"] / CHARS_PER_TOKEN
    return {
        "tokens": per * examples,
        "lo": max(0.0, per - half) * examples,
        "hi": (per + half) * examples,
        "per_example": per,
    }


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
        out.append({"k": key, "m": {k: str(v) for k, v in meta.items()}})

    stats = summarize(totals)
    target_stats = summarize(targets)
    return {
        "dataset": ctx.get("dataset"),
        "stage": ctx.get("stage"),
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
