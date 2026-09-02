"""Is a benchmark's test set in the training data — where, and on which side?

`benchmarks` cuts a probe out of each test item. This module searches for all of
them at once, over every row of a post-training mix and, separately, in a
pretraining index, and turns the hits back into a statement per item: this
question is in the SFT prompts, this worked answer is in the responses the model
was fit to, this one is in the corpus stand-in a few hundred times.

The scan is `grep`'s — the same Parquet route, the same column-to-side mapping,
the same branch split for a preference pair — with one change: many patterns in
one pass. Reading a mix costs gigabytes over the network, so two hundred probes
have to share a read rather than each paying for one. They are searched as
alternations, `(?:probe|probe|...)`, and the string each alternation matched is
pulled back with it so the hit can be handed to the probe it belongs to.

One alternation does not hold them all. RE2 runs an alternation as a DFA while
its state cache fits and falls back to an NFA when it does not, and the fall is
a cliff: measured here, four hundred thirteen-word probes in one regex ran fifty
times slower than the same probes split four ways and OR'd. `CHUNK_CHARS` is
that split, by pattern length rather than probe count, because the states are
made of characters — a code probe with long identifiers costs more of them than
a prose one.

What a count here means, and what it does not:

  * Rows, not occurrences, per probe per side — the same unit `grep` reports.
  * A hit on the question in a prompt column is the model being trained to
    *read* the test item. A hit on the answer in a response, chosen or reference
    column is the model being trained to *produce* it, or scored against it.
    Those are reported apart because they are different claims.
  * A miss is a miss for verbatim and near-verbatim copies. Paraphrases,
    translations and reformattings that break the word window are not found.
  * The corpus side is an exact-string count in whatever index stands closest to
    the model's pretraining data — for OLMo 3 a different corpus, and the run
    says so — over occurrences rather than documents.
"""

import re

from . import grep
from .stats import wilson

# Characters of pattern per alternation. Measured on 17.6 MB of text: 400 probes
# as one regex took 18.8 s, as 4 regexes of 100 took 0.21 s, as 8 of 50 took
# 0.35 s. A thirteen-word prose probe escapes to roughly 60-90 characters, so
# this is on the order of sixty probes an alternation.
CHUNK_CHARS = 5000

# Context kept either side of a matched probe in an example snippet.
SNIPPET_CONTEXT = 120

# Which sides count as the model being trained to produce the text. The same
# set `influence` ranks a stage's origin on.
PRODUCE = ("response", "chosen", "reference")


def chunks(probes: list[dict], budget: int | None = None) -> list[list[dict]]:
    """Probes grouped so no alternation exceeds `budget` characters of pattern.

    A single probe longer than the budget still gets a chunk of its own; the
    budget bounds the alternation, not the probe. The default is read at call
    time so the constant can be lowered for a run or a test.
    """
    if budget is None:
        budget = CHUNK_CHARS
    out: list[list[dict]] = []
    size = 0
    for p in probes:
        n = len(p["regex"]) + 1
        if not out or size + n > budget:
            out.append([p])
            size = n
        else:
            out[-1].append(p)
            size += n
    return out


def alternation(chunk: list[dict]) -> str:
    return "(?:" + "|".join(p["regex"] for p in chunk) + ")"


def _flags(case_sensitive: bool) -> str:
    return "" if case_sensitive else ", 'i'"


def _test(patterns: list[str], case_sensitive: bool) -> str:
    """SQL predicate on the string bound to `t`: does any alternation match it."""
    return " OR ".join(
        f"regexp_matches(t, {grep._lit(p)}{_flags(case_sensitive)})" for p in patterns
    )


def _extract(patterns: list[str], case_sensitive: bool, var: str = "t") -> str:
    """VARCHAR[] of every substring of `var` any alternation matched."""
    parts = ", ".join(
        f"regexp_extract_all({var}, {grep._lit(p)}, 0{_flags(case_sensitive)})"
        for p in patterns
    )
    if len(patterns) == 1:
        return parts
    return f"flatten(list_value({parts}))"


def _matches_sql(exprs: list[str], patterns: list[str], case_sensitive: bool) -> str:
    """Every matched substring across a group's strings, as one VARCHAR[]."""
    matching = grep._matching(exprs, _test(patterns, case_sensitive))
    return f"flatten(list_transform({matching}, t -> {_extract(patterns, case_sensitive)}))"


def _snippet_sql(exprs: list[str], patterns: list[str], case_sensitive: bool) -> tuple[str, str, str]:
    """The first matching string windowed on its first match, plus where the
    window started and how long the string was, so the caller can mark the cuts.

    The matched substring is text taken verbatim from the string, so a plain
    `position` finds it exactly — no second regex evaluation, and no
    case-folding to get wrong.
    """
    first = f"{grep._matching(exprs, _test(patterns, case_sensitive))}[1]"
    m = f"{_extract(patterns, case_sensitive, 's')}[1]"
    scalar = f"FROM (SELECT {first} AS s)"
    start = f"greatest(1, position({m} IN s) - {SNIPPET_CONTEXT})"
    width = f"len({m}) + {2 * SNIPPET_CONTEXT}"

    def q(expr):
        return f"(SELECT CASE WHEN s IS NULL THEN NULL ELSE {expr} END {scalar})"

    return q(f"substr(s, {start}, {width})"), q(start), q("len(s)")


def compile_probes(probes: list[dict], case_sensitive: bool = False) -> list[tuple[dict, re.Pattern]]:
    return [
        (p, re.compile(p["regex"], 0 if case_sensitive else re.IGNORECASE))
        for p in probes
    ]


def attribute(matched: str, compiled: list[tuple[dict, re.Pattern]]) -> list[dict]:
    """The probes a matched substring belongs to.

    An alternation match is exactly one branch's match, so it fullmatches that
    branch's own regex. Usually one probe; two when two items share a window,
    which duplicated items in a benchmark do, and then both are hit.
    """
    return [p for p, rx in compiled if rx.fullmatch(matched)]


def scan(
    con,
    from_sql: str,
    exprs: dict[str, list[str]],
    source: str | None,
    probes: list[dict],
    case_sensitive: bool = False,
    examples: int = 20,
) -> dict:
    """Rows of the mix each probe lands in, by side, in one pass over the columns.

    Same shape of query as `grep.scan`: only matching rows come back, with the
    matched substrings per side, and the tallying happens here. A row holding
    two probes is one row in `matched` and one row against each probe.
    """
    patterns = [alternation(c) for c in chunks(probes)]
    compiled = compile_probes(probes, case_sensitive)
    groups = list(exprs)

    select = [f"{source or 'NULL'} AS src"]
    for g in groups:
        select.append(f"{_matches_sql(exprs[g], patterns, case_sensitive)} AS m_{g}")
    for g in groups:
        window, start, total = _snippet_sql(exprs[g], patterns, case_sensitive)
        select += [f"{window} AS snip_{g}", f"{start} AS at_{g}", f"{total} AS len_{g}"]
    any_match = " OR ".join(
        f"({grep._match_sql(exprs[g], _test(patterns, case_sensitive))})" for g in groups
    )
    con.execute(f"SELECT {', '.join(select)} FROM {from_sql} WHERE {any_match}")

    matched = 0
    by_source: dict[str, int] = {}
    by_group = {g: 0 for g in groups}
    hits: dict[tuple[int, str], int] = {}
    kept: list = []
    by_snippet: dict[str, dict] = {}
    while True:
        batch = con.fetchmany(5000)
        if not batch:
            break
        for row in batch:
            src = grep.source_label(row[0], has_column=source is not None)
            matched += 1
            by_source[src] = by_source.get(src, 0) + 1
            base = 1 + len(groups)
            first_hit = None
            for i, g in enumerate(groups):
                strings = row[1 + i] or []
                if not strings:
                    continue
                by_group[g] += 1
                seen: set[int] = set()
                for s in strings:
                    for p in attribute(s, compiled):
                        if p["id"] in seen:
                            continue
                        seen.add(p["id"])
                        hits[(p["id"], g)] = hits.get((p["id"], g), 0) + 1
                        if first_hit is None:
                            first_hit = (p, g)
                if examples and first_hit and first_hit[1] == g:
                    snip, at, total = row[base + 3 * i : base + 3 * i + 3]
                    if snip:
                        shown = (
                            ("…" if (at or 1) > 1 else "")
                            + snip
                            + ("…" if (at or 1) - 1 + len(snip) < (total or 0) else "")
                        )
                        grep._pick(kept, by_snippet, examples, shown, {
                            "item": first_hit[0]["item"],
                            "part": first_hit[0]["part"],
                            "group": g,
                            "source": src,
                            "snippet": shown,
                        })

    return {
        "matched": matched,
        "by_group": by_group,
        "by_source": grep._by_count(by_source),
        "probe_hits": [
            {"probe": pid, "group": g, "rows": n}
            for (pid, g), n in sorted(hits.items())
        ],
        "examples": [by_snippet[text] for _, text in sorted(kept, key=lambda kv: -kv[0])],
    }


def items_hit(probes: list[dict], probe_hits: list[dict]) -> dict:
    """Roll probe-level hits up to items, by the claim each hit supports.

    `any`: items with a hit anywhere. `question_read`: the question in a prompt
    column — trained to read it. `answer_produced`: the answer in a column the
    model is fit to or scored against — trained to produce it. `question_produced`
    is the question restated in produced text, which a model that was trained on
    the answer usually did too, and is kept apart so it cannot inflate the
    second.
    """
    by_id = {p["id"]: p for p in probes}
    out = {"any": set(), "question_read": set(), "question_produced": set(),
           "answer_produced": set(), "answer_rejected": set()}
    for h in probe_hits:
        p = by_id[h["probe"]]
        out["any"].add(p["item"])
        if p["part"] == "question":
            if h["group"] == "prompt":
                out["question_read"].add(p["item"])
            elif h["group"] in PRODUCE:
                out["question_produced"].add(p["item"])
        elif p["part"] == "answer":
            if h["group"] in PRODUCE:
                out["answer_produced"].add(p["item"])
            elif h["group"] == "rejected":
                out["answer_rejected"].add(p["item"])
    return {k: sorted(v) for k, v in out.items()}


def corpus_items(probes: list[dict], counts: list[dict]) -> dict:
    """Roll the corpus counts up to items, with the probes the index never
    answered kept out of every denominator.

    A count carrying an `error` is a probe that was not counted — the query was
    rejected or the API stopped answering — and it is not a zero. A stage scan
    has no such case, since reading every row answers every probe; the corpus
    side is one request per probe and any of them can fail.

    A corpus has one side, so the exact-string count stands in for `rows` and
    `items_hit` reads every hit as `any`; the side keys stay empty because a
    corpus document is neither a prompt nor a response.

    Which items the rate is over: an item is settled *present* once any of its
    probes hit, however the others fared — a hit is a hit. It is settled *absent*
    only when every one of its probes came back with a count of zero. An item
    with an unanswered probe and no hit is neither, and is listed in
    `items_unresolved` rather than counted on either side of the rate, so
    "we did not finish looking" is never read as "it is not there".
    `items_probed` is the settled items, and is what `summary` divides by.
    """
    by_id = {p["id"]: p for p in probes}
    errors = [c["probe"] for c in counts if c.get("error")]
    hits = [
        {"probe": c["probe"], "group": "document", "rows": c["occurrences"]}
        for c in counts if not c.get("error") and c["occurrences"]
    ]
    items = items_hit(probes, hits)
    seen = set(items["any"])
    unresolved = sorted({by_id[pid]["item"] for pid in errors} - seen)
    probed = sorted({p["item"] for p in probes} - set(unresolved))
    return {
        "items": items,
        "items_probed": probed,
        "items_unresolved": unresolved,
        "errors": errors,
    }


def _rate(k: int, n: int) -> str:
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n * 100:.1f}% (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"


def summary(stage_runs: list[dict], corpus_run: dict | None, unscanned: list[str]) -> list[str]:
    """Lines for the whole check, one stage at a time, then the corpus.

    The interval is over the benchmark: the items were a draw from the test set,
    so the share found is an estimate of the share of the whole benchmark that
    is present. It is *not* an interval over the mix — every row of that was
    read, and the row counts are exact.
    """
    lines = []
    for r in stage_runs:
        probed = r["items_probed"]
        n = len(probed)
        hit = r["items"]
        parts = [f"{r['stage']:6s} items seen {_rate(len(hit['any']), n)}"]
        parts.append(f"  question in a prompt: {len(hit['question_read'])}")
        if r["has_answer_probes"]:
            parts.append(f"  answer in produced text: {len(hit['answer_produced'])}")
            if hit["answer_rejected"]:
                parts.append(f"  answer in a rejected completion only: "
                             f"{len(set(hit['answer_rejected']) - set(hit['answer_produced']))}")
        if hit["question_produced"]:
            parts.append(f"  question restated in produced text: {len(hit['question_produced'])}")
        parts.append(
            f"  rows: {r['matched']:,} of {r['total_rows']:,}"
            + ("  (partial conversion — a floor)" if r.get("partial") else "")
        )
        lines.extend(parts)
    for s in unscanned:
        lines.append(f"{s:6s} not scanned — not a zero")
    if corpus_run is not None:
        n = len(corpus_run["items_probed"])
        hit = corpus_run["items"]
        lines.append(
            f"corpus items seen {_rate(len(hit['any']), n)}  in {corpus_run['index']}"
        )
        # A probe the index never answered is not in `occurrences` and its item
        # is not in the rate; say so, or a count over the rest reads as complete.
        counted = [c for c in corpus_run["counts"] if not c.get("error")]
        errored = len(corpus_run["counts"]) - len(counted)
        if errored:
            unresolved = corpus_run.get("items_unresolved", [])
            lines.append(
                f"  {errored} probe(s) could not be counted — not zeros"
                + (f"; {len(unresolved)} item(s) left unresolved and out of the rate"
                   if unresolved else "")
            )
        occ = sum(c["occurrences"] for c in counted)
        lines.append(f"  occurrences over all probes: {occ:,}"
                     + (" (some approximate)" if any(c["approx"] for c in counted) else ""))
        if corpus_run.get("caveat"):
            lines.append(f"  note: {corpus_run['caveat']}")
    return lines
