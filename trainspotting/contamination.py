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
alternations, `(?:probe|probe|...)`, and the strings an alternation matched are
pulled back whole so the probes in each can be found here, one position at a
time — RE2's extraction hands back non-overlapping matches only, and two
near-duplicate items whose windows sit a word apart would otherwise count as one.

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
# The sides the model is *fit to*, as against `reference`, which a verifier scores
# rollouts against and no completion need contain. `influence` ranks on PRODUCE
# because a string the objective pushes toward is evidence either way; the
# summary here makes a claim per side, and "trained to produce this" is only
# true of these two.
FIT = ("response", "chosen")


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
        # The wrapper `alternation` adds, `(?:` and `)`, is part of what RE2
        # compiles, so it counts against the same budget as the probes it holds.
        if not out or size + n > budget:
            out.append([p])
            size = len("(?:)") + n
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
    """Every string of a group any alternation matched, whole, as one VARCHAR[].

    Whole rather than the matched substrings, because `regexp_extract_all` walks
    a string left to right and resumes after each match: a probe whose window
    starts a word inside another's is consumed with it and never comes back.
    The strings are few — only matching rows reach here — and `find` does the
    attribution on them with overlaps kept.
    """
    return grep._matching(exprs, _test(patterns, case_sensitive))


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


def _python(regex: str) -> str:
    r"""A probe regex as Python's `re` must read it to agree with DuckDB's RE2.

    The scan runs the regex through RE2 and hands back the strings it matched;
    `find` then runs the same regex here to say which probe matched. Wherever
    the two engines read a token differently the attribution drifts from the
    selection, in one direction or the other:

      * `\b`: RE2's is ASCII, Python's counts every Unicode letter as a word
        character, so a copy against an accented letter (`étrain ...`) is a hit
        to RE2 and a boundary failure here — a row in `matched` and on its
        side, in no probe's tally.
      * `\s`: RE2's is ASCII, Python's admits a no-break space or an em space.
        That cannot lose a hit, since Python only attributes strings RE2
        selected, but it can invent one: a string RE2 picked for one probe may
        hold a second probe's words joined by U+00A0, which RE2 passed over and
        Python would credit — a `probe_hits` entry, an item, and possibly the
        example's attribution, for a copy the scan never matched.

    Each is rewritten with `(?a:...)`, ASCII scoped to that token alone, so
    case-folding stays Unicode-aware the way RE2's `(?i)` is — `re.ASCII` on
    the whole pattern would stop folding `É` to `é` and drop the recased copy
    RE2 matched.

    Case folding is the residual gap, and it is left as one. Python's
    `IGNORECASE` folds the dotted capital I (U+0130) to `i` and RE2's `(?i)`
    does not, so a string RE2 selected for one probe could hand Python a second
    probe's words that differ from the text only by that letter — the same
    invent-a-hit direction as `\s`, and confined to probes containing it. No
    RE2-compatible folder ships with Python, and carrying RE2's folding table
    here to close a gap that narrow is not a trade this module makes.

    `re.escape` never emits a bare `\b` or `\s` (a backslash in the text
    becomes `\\`, and the letter after it stays a plain letter), so walking
    the pattern by escape pairs rewrites exactly the tokens `probe()` added.
    """
    return re.sub(
        r"(?s)\\(.)",
        lambda m: f"(?a:\\{m.group(1)})" if m.group(1) in "bs" else m.group(0),
        regex,
    )


def _compile(regex: str, case_sensitive: bool) -> re.Pattern:
    return re.compile(_python(regex), 0 if case_sensitive else re.IGNORECASE)


def compile_probes(probes: list[dict], case_sensitive: bool = False) -> list[tuple[dict, re.Pattern]]:
    return [(p, _compile(p["regex"], case_sensitive)) for p in probes]


def compile_alternations(probes: list[dict], case_sensitive: bool = False) -> list[re.Pattern]:
    return [_compile(alternation(c), case_sensitive) for c in chunks(probes)]


def find(text: str, alternations: list[re.Pattern], compiled: list[tuple[dict, re.Pattern]]) -> list[dict]:
    """Every probe in `text`, overlapping copies included.

    The alternations find where a probe starts; every probe is then tried at
    that position. Two things make it more than one pass of `finditer`:

      * The search resumes one character after each match *started*, not where
        it ended. A finder that resumes at the end skips any probe whose window
        begins inside the match — two near-duplicate items a word apart, which
        templated benchmarks have — and reports the second as a miss.
      * At one position the alternation reports only its first branch, so the
        probes are tried there one by one. Two items sharing a window both
        match, as do two whose windows differ only past a shared prefix.

    Anchored `match` calls fail on the first character for almost every probe,
    so this costs a few hundred cheap checks per occurrence, on the matching
    rows only.
    """
    found: dict[int, dict] = {}
    for rx in alternations:
        pos = 0
        while (m := rx.search(text, pos)) is not None:
            for p, prx in compiled:
                if p["id"] not in found and prx.match(text, m.start()):
                    found[p["id"]] = p
            pos = m.start() + 1
    return list(found.values())


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
    matching strings per side, and the attribution and tallying happen here. A
    row holding two probes is one row in `matched` and one row against each probe.
    """
    patterns = [alternation(c) for c in chunks(probes)]
    compiled = compile_probes(probes, case_sensitive)
    alternations = compile_alternations(probes, case_sensitive)
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
                    for p in find(s, alternations, compiled):
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
    model is fit to — trained to produce it. `answer_scored`: the answer in an RL
    row's reference — what the verifier scores rollouts against, which no
    completion need contain, so it is a claim about the reward and not about
    text the model emitted. `question_produced` is the question restated in
    produced or reference text, which a model that was trained on the answer
    usually did too, and is kept apart so it cannot inflate the second.

    A `choices` probe — a multiple-choice item's options — is the item being
    read wherever the stem would be, so it rolls up with `question`: the options
    are the part of the item the model is shown, not the letter it is scored on.
    """
    by_id = {p["id"]: p for p in probes}
    out = {"any": set(), "question_read": set(), "question_produced": set(),
           "answer_produced": set(), "answer_scored": set(), "answer_rejected": set()}
    for h in probe_hits:
        p = by_id[h["probe"]]
        out["any"].add(p["item"])
        if p["part"] in ("question", "choices"):
            if h["group"] == "prompt":
                out["question_read"].add(p["item"])
            elif h["group"] in PRODUCE:
                out["question_produced"].add(p["item"])
        elif p["part"] == "answer":
            if h["group"] in FIT:
                out["answer_produced"].add(p["item"])
            elif h["group"] == "reference":
                out["answer_scored"].add(p["item"])
            elif h["group"] == "rejected":
                out["answer_rejected"].add(p["item"])
    return {k: sorted(v) for k, v in out.items()}


def _settle(probes: list[dict], items: dict, unsettled: set[int]) -> dict:
    """The rollup with the items the check did not finish kept out of the rate.

    Which items the rate is over: an item is settled *present* once any of its
    probes hit, however the others fared — a hit is a hit. It is settled
    *absent* only when every part of it was probed and every probe came back
    empty. An item in `unsettled` — one with a probe the index never answered,
    or a part the server cut before it could be probed — and no hit is neither,
    and is listed in `items_unresolved` rather than counted on either side of
    the rate, so "we did not finish looking" is never read as "it is not
    there". `items_probed` is the settled items, and is what `summary` divides by.
    """
    seen = set(items["any"])
    unresolved = sorted(unsettled - seen)
    probed = sorted({p["item"] for p in probes} - set(unresolved))
    return {"items": items, "items_probed": probed, "items_unresolved": unresolved}


def stage_items(probes: list[dict], probe_hits: list[dict], cut: list[int] = ()) -> dict:
    """Roll a stage scan's probe hits up to items, with the items the server
    cut a part from kept out of the denominator unless they hit.

    Reading every row answers every probe, so a stage scan has no unanswered
    probe. What it can have is a part that was never probed, because the
    datasets server truncated the cell it would have been cut from. The item's
    other parts were searched and a hit on them stands; a miss on them is not
    a miss on the item, since the cut part may be the one that was copied.
    `cut` is those items, by index.
    """
    return _settle(probes, items_hit(probes, probe_hits), set(cut))


def corpus_items(probes: list[dict], counts: list[dict], cut: list[int] = ()) -> dict:
    """Roll the corpus counts up to items, with the probes the index never
    answered kept out of every denominator.

    A count carrying an `error` is a probe that was not counted — the query was
    rejected or the API stopped answering — and it is not a zero. A stage scan
    has no such case, since reading every row answers every probe; the corpus
    side is one request per probe and any of them can fail. `cut` is the items
    the server truncated a part of, and is unsettled here for the same reason
    it is in `stage_items`.

    A corpus has one side, so the exact-string count stands in for `rows` and
    `items_hit` reads every hit as `any`; the side keys stay empty because a
    corpus document is neither a prompt nor a response.
    """
    by_id = {p["id"]: p for p in probes}
    errors = [c["probe"] for c in counts if c.get("error")]
    hits = [
        {"probe": c["probe"], "group": "document", "rows": c["occurrences"]}
        for c in counts if not c.get("error") and c["occurrences"]
    ]
    unsettled = {by_id[pid]["item"] for pid in errors} | set(cut)
    return {**_settle(probes, items_hit(probes, hits), unsettled), "errors": errors}


def _rate(k: int, n: int, census: bool = False, floor: bool = False) -> str:
    """The share, with an interval only where an interval means something.

    The Wilson interval is a sampling interval over the benchmark. Two cases
    have no sampling to put one on. A census — every item of the split settled
    — is the benchmark's share exactly, not an estimate of it. A partial
    conversion is the other way: the rows the server never converted were
    never read, so a miss is not a known miss and the share is a floor with no
    upper side to state.
    """
    if not n:
        return "—"
    out = f"{k}/{n} = {k / n * 100:.1f}%"
    if floor:
        return f"≥ {out}" + (" (every item)" if census else "")
    if census:
        return f"{out} (every item)"
    lo, hi = wilson(k, n)
    return f"{out} (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"


def _census(r: dict) -> bool:
    """Was every item of the split settled — a census rather than a draw.

    `total_items` is the split's size; `items_probed` is the settled items,
    with anything unresolved or never probed already left out, so equality is
    the whole benchmark answered.
    """
    total = (r.get("benchmark") or {}).get("total_items")
    return bool(total) and len(r["items_probed"]) == total


def summary(
    stage_runs: list[dict],
    corpus_run: dict | None,
    unscanned: list[str],
    corpus_skipped: str | None = None,
) -> list[str]:
    """Lines for the whole check, one stage at a time, then the corpus.

    The interval is over the benchmark: the items were a draw from the test set,
    so the share found is an estimate of the share of the whole benchmark that
    is present. It is *not* an interval over the mix — every row of that was
    read, and the row counts are exact. When the draw was the whole split the
    share is exact and no interval is printed; when the conversion was partial
    the share is a floor and no interval is printed either — see `_rate`.

    `corpus_skipped` names the flag that took the corpus side out of a run
    that has one. A stage `--corpus-only` skipped is listed as not scanned;
    a corpus `--no-corpus` skipped gets the same line, or the stage-only
    summary reads as the whole check.
    """
    lines = []
    for r in stage_runs:
        probed = r["items_probed"]
        n = len(probed)
        hit = r["items"]
        rate = _rate(len(hit["any"]), n, census=_census(r), floor=bool(r.get("partial")))
        parts = [f"{r['stage']:6s} items seen {rate}"]

        # A claim about a side that --field left out of the read is not a zero.
        # `fields` is what this run read and `available_fields` what the mix
        # holds; a claim is searched when every side it draws on that the mix
        # has was read. A file from before the key recorded every side.
        def searched(groups) -> bool:
            if r.get("fields") is None:
                return True
            need = set(groups) & set(r.get("available_fields") or r["fields"])
            return bool(need) and need <= set(r["fields"])

        parts.append(
            f"  question in a prompt: {len(hit['question_read'])}"
            if searched(("prompt",)) else "  question in a prompt: not searched — not a zero"
        )
        if r["has_answer_probes"]:
            # Two claims, one per kind of side the mix has. An RL mix stores no
            # completion, so it gets only the reference line; an SFT or DPO mix
            # has no reference, so it gets only the produced one. A run from
            # before `answer_scored` existed folded reference hits into
            # `answer_produced`, and reads as it was written.
            has = set(r.get("available_fields") or r.get("fields") or PRODUCE)
            if has & set(FIT):
                parts.append(
                    f"  answer in produced text: {len(hit['answer_produced'])}"
                    if searched(FIT) else "  answer in produced text: not searched — not a zero"
                )
            if "reference" in has:
                parts.append(
                    f"  answer in a verifier reference: {len(hit.get('answer_scored', []))}"
                    if searched(("reference",))
                    else "  answer in a verifier reference: not searched — not a zero"
                )
            if hit["answer_rejected"]:
                parts.append(f"  answer in a rejected completion only: "
                             f"{len(set(hit['answer_rejected']) - set(hit['answer_produced']))}")
        if hit["question_produced"]:
            parts.append(f"  question restated in produced text: {len(hit['question_produced'])}")
        parts.append(
            f"  rows: {r['matched']:,} of {r['total_rows']:,}"
            + ("  (partial conversion — a floor)" if r.get("partial") else "")
        )
        # An item the server cut a part from, with no hit on the rest, was not
        # fully searched; it is out of the rate, and the line says so, or the
        # denominator reads as every item the run reached.
        if r.get("items_unresolved"):
            parts.append(
                f"  {len(r['items_unresolved'])} item(s) with a part cut by the server and "
                "no hit — unresolved, out of the rate"
            )
        lines.extend(parts)
    for s in unscanned:
        lines.append(f"{s:6s} not scanned — not a zero")
    if corpus_run is None and corpus_skipped:
        lines.append(f"corpus not scanned — not a zero ({corpus_skipped})")
    if corpus_run is not None:
        n = len(corpus_run["items_probed"])
        hit = corpus_run["items"]
        rate = _rate(len(hit["any"]), n, census=_census(corpus_run))
        lines.append(f"corpus items seen {rate}  in {corpus_run['index']}")
        # A probe the index never answered is not in `occurrences` and its item
        # is not in the rate; say so, or a count over the rest reads as complete.
        # The same for an item the server cut a part from.
        counted = [c for c in corpus_run["counts"] if not c.get("error")]
        errored = len(corpus_run["counts"]) - len(counted)
        if errored:
            lines.append(f"  {errored} probe(s) could not be counted — not zeros")
        unresolved = corpus_run.get("items_unresolved", [])
        if unresolved:
            lines.append(
                f"  {len(unresolved)} item(s) with a probe not counted or a part cut by the "
                "server, and no hit — unresolved, out of the rate"
            )
        occ = sum(c["occurrences"] for c in counted)
        lines.append(f"  occurrences over all probes: {occ:,}"
                     + (" (some approximate)" if any(c["approx"] for c in counted) else ""))
        if corpus_run.get("caveat"):
            lines.append(f"  note: {corpus_run['caveat']}")
    return lines
