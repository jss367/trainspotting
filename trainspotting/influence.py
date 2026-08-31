"""Read a set of `grep` runs as one answer: where a string most plausibly comes from.

`grep` counts a string in one mix at a time and prints what it found. Stacked up
across a pipeline those counts mislead in two ways, and both have the same shape:
a bigger number is not a bigger effect.

The first is size. A stage with ten times the rows shows more of any string for
that reason alone, so the comparable quantity is the rate — hits over the stage's
own row count — and again inside a stage, hits over the row count of the source
they concentrate in. "434 of the 17,596 WildChat rows" is a claim about WildChat;
"434 of 150,000" is a claim about nothing in particular.

The second is position. A string in a prompt is text the model was trained to
read; a string in the response a stage fits, or in the reference answer a
verifier scores rollouts against, is text the objective pushes it to emit. When
the question is why a model *says* something, only the second kind is evidence,
and `grep` already separates them.

So the ranking here is by produce-side rate. What it deliberately does not do is
weight the stages against each other: a rate late in the pipeline generally moves
behaviour more than the same rate in pretraining, but by how much is not
something these counts measure, so it is printed as a caveat rather than folded
into a score.

Zero is an answer too, and a different one from silence. A stage searched and
found empty is exact over every one of its rows; a stage never scanned, or one
this layer cannot reach at all, is unknown. Collapsing those two into "no hits"
is what turns "we did not look" into "it is not there", so they are counted and
named apart.
"""

# The groups holding text the training objective pushes the model toward, as
# opposed to text it only conditions on. `reference` belongs here for RL: it is
# the ground truth a verifier scores against, so the gradient points at it even
# though no response is stored.
PRODUCE = ("response", "reference")

# What a stage's absence from the result set means. INCONCLUSIVE is the fourth
# kind and the easiest to lose: a repo the datasets-server converted only part
# of was scanned over that part, so finding nothing there is not finding nothing.
HITS, ZERO, UNSEARCHED, UNREACHABLE, INCONCLUSIVE = (
    "hits", "zero", "unsearched", "unreachable", "inconclusive"
)


def produced(result) -> tuple[int, int] | None:
    """Rows matching on the produce side, as (low, high), or None if unsearched.

    `by_group` counts rows per group and a row can match in two of them, so the
    union is not the sum: it is at least the largest group and at most their
    total, capped by the rows that matched at all. In practice the interval is
    tight — 232 to 235 of the 400 Think-RL rows holding "as an AI language
    model" — and reporting it as an interval costs nothing, where reporting the
    sum as if it were the union would overcount every row matching on both
    sides.
    """
    fields = result.get("fields") or []
    by_group = result.get("by_group") or {}
    counts = [by_group.get(g, 0) for g in PRODUCE if g in fields]
    if not counts:
        return None
    return max(counts), min(result.get("matched", 0), sum(counts))


def _sources(result) -> list[dict]:
    """Every source the matches fall in, each against its own row count.

    The rate is the point. A source holding 267 of a stage's 521 matches sounds
    like the origin until its own denominator turns up: at 124,980 rows it is
    simply the biggest source in the mix and holds the string at the same rate
    as everything else. `lift` is that comparison — the source's rate over the
    stage's — and it is what separates a concentration from a share.
    """
    totals = result.get("rows_by_source") or {}
    groups = result.get("by_source_group") or {}
    rows = result.get("total_rows") or 0
    stage_rate = (result.get("matched", 0) / rows) if rows else 0
    bounds = produced(result)
    stage_produced_rate = (bounds[0] / rows) if (bounds and rows) else 0
    fields = set(result.get("fields") or [])
    out = []
    for name, hits in (result.get("by_source") or {}).items():
        n = totals.get(name)
        rate = hits / n if n else None
        # The same lower bound the stage's produce-side rate uses, per source.
        # Without it the produce-side verdict credits whichever source holds the
        # most matches overall, which can be a source that contributed only
        # prompts — none of the evidence the ranking actually ran on.
        per_group = groups.get(name, {})
        p_counts = [per_group.get(g, 0) for g in PRODUCE if g in fields]
        p_hits = max(p_counts) if (p_counts and per_group) else None
        p_rate = (p_hits / n) if (p_hits is not None and n) else None
        out.append({
            "name": name,
            "hits": hits,
            "rows": n,
            "rate": rate,
            "lift": (rate / stage_rate) if (rate and stage_rate) else None,
            "produced_hits": p_hits,
            "produced_rate": p_rate,
            "produced_lift": (p_rate / stage_produced_rate) if (p_rate and stage_produced_rate) else None,
            "groups": per_group,
        })
    return sorted(out, key=lambda s: -s["hits"])


# A source has to hold this share of a stage's matches before its rate is worth
# reading: two rows out of ten is a high rate and almost always noise.
MIN_SHARE = 0.1
# ...and its rate has to beat the stage's by this much to be a concentration
# rather than the source's size showing through.
MIN_LIFT = 2.0


def concentration(sources: list[dict], hits: int, side: str = "all") -> dict | None:
    """The source the matches actually bunch in, or None if they are spread.

    Returning None matters as much as returning a source: "most of the matches
    are in X" is a claim about where the string entered the mix, and it is false
    whenever X is merely the largest source.

    `side="produced"` runs the same test over the produce-side counts alone, so
    a ranking that ran on produce-side rates is explained by the sources that
    supplied those rows rather than by whichever source holds the most prompts.
    """
    hk, rk, lk = ("hits", "rate", "lift") if side == "all" else (
        "produced_hits", "produced_rate", "produced_lift")
    total = hits if side == "all" else max(
        (s[hk] or 0 for s in sources), default=0)
    floor = max(2, MIN_SHARE * total)
    worth = [s for s in sources if (s[hk] or 0) >= floor and s[lk]]
    if not worth:
        return None
    best = max(worth, key=lambda s: s[rk])
    return best if best[lk] >= MIN_LIFT else None


def coverage_gaps(result) -> list[str]:
    """Why a zero from this run would not be a zero for the whole stage.

    Three ways a scan can miss text it never read, and each turns "0 rows
    matched" from a fact about the stage into a fact about the columns that
    were opened. The rows axis (a partial conversion) and the columns axis
    (`--field`, and any text column the mapping did not recognise) are both
    here because the verdict reads a zero as *this string is not in the
    stage*, which neither one supports.
    """
    gaps = []
    if result.get("partial"):
        gaps.append("the server converted only part of the repo")
    fields, available = set(result.get("fields") or []), set(result.get("available_fields") or [])
    if available and available - fields:
        gaps.append(f"this run read only {', '.join(sorted(fields))} of "
                    f"{', '.join(sorted(available))}")
    elif not available:
        # Runs predating `available_fields` cannot show they read everything.
        gaps.append("this run does not record which sides the mix has, so it cannot show it "
                    "read all of them")
    if result.get("unsearched_columns"):
        gaps.append(f"{', '.join(result['unsearched_columns'])} went unsearched")
    return gaps


def stage_trace(result) -> dict:
    """One `grep` result, annotated with everything the comparison needs."""
    rows = result.get("total_rows") or 0
    hits = result.get("matched", 0)
    partial = bool(result.get("partial"))
    gaps = coverage_gaps(result)
    bounds = produced(result)
    sources = _sources(result)
    return {
        "stage": result["stage"],
        "dataset": result.get("dataset"),
        # Finding nothing is only a zero for the stage if the whole stage was
        # read. Anything short of that is inconclusive, and says which part.
        "status": HITS if hits else (ZERO if not gaps else INCONCLUSIVE),
        "coverage_gaps": gaps,
        "rows": rows,
        "hits": hits,
        "rate": hits / rows if rows else None,
        "by_group": dict(result.get("by_group") or {}),
        "fields": list(result.get("fields") or []),
        # Groups the mix actually has, when the run recorded them: it is what
        # separates "--field narrowed this" from "the mix has no such column".
        "available_fields": list(result.get("available_fields") or []),
        "produced": bounds,
        # Both ends of the interval, because ranking on the low end alone
        # settles an order the counts do not actually settle.
        "produced_rate": (bounds[0] / rows) if (bounds and rows) else None,
        "produced_rate_hi": (bounds[1] / rows) if (bounds and rows) else None,
        "sources": sources,
        # Both readings are kept; `compare` picks the one matching the basis it
        # ends up ranking on, so the verdict explains the number it printed.
        "concentration_all": concentration(sources, hits),
        "concentration_produced": concentration(sources, hits, side="produced"),
        "concentration": concentration(sources, hits),
        "conc_side": "all",
        "revision": result.get("revision"),
        "partial": partial,
        "unsearched_columns": list(result.get("unsearched_columns") or []),
    }


def _rank_block(r: dict, basis: str, key: str) -> str | None:
    """Why this stage cannot be compared to the others, or None if it can.

    A rate only ranks against another rate when both are the same measurement
    over the same population. Two ways that fails, and in both the honest move
    is to keep the stage's counts on the page and out of the ordering rather
    than to sort it low — a stage sorted last reads as measured and losing.
    """
    if r["partial"]:
        # The denominator is the converted subset, and the conversion is a
        # prefix rather than a sample, so the quotient is not a stage rate.
        return ("only part of this repo was converted, so its denominator is that subset "
                "rather than the stage")
    if basis == "produced" and r[key] is None:
        # `--field` is per run, so one pattern's stages can disagree about what
        # was read. Sorting an unread side as zero is the conflation this layer
        # exists to avoid.
        return (f"this run read only {', '.join(r['fields']) or 'nothing'}, so there is no "
                "produce-side rate to compare")
    return None


def compare(results: list[dict], stages: list[dict]) -> dict:
    """Every stage of a pipeline against one pattern, ranked by likely influence.

    `results` are `grep` result files for a single model and pattern, in any
    order; `stages` is the model's full registry list, which is what supplies
    the stages *missing* from `results` — the point of the exercise is that a
    stage nobody scanned reads differently from one scanned and found empty.
    """
    definitions = {(r.get("pattern"), bool(r.get("regex")), bool(r.get("case_sensitive")))
                   for r in results}
    if len(definitions) > 1:
        # Runs are grouped by slug, and a slug is a filename rather than a
        # promise. Ranking two different searches against each other while
        # printing one of their patterns is worse than refusing.
        raise ValueError(
            "these runs are not the same search: "
            + "; ".join(sorted(f"{p!r} (regex={g}, case_sensitive={c})" for p, g, c in definitions))
        )
    found = {r["stage"]: stage_trace(r) for r in results}
    rows = []
    for s in stages:
        if s["stage"] in found:
            rows.append(found[s["stage"]])
            continue
        rows.append({
            "stage": s["stage"],
            "dataset": s.get("hf_dataset") or s.get("hf"),
            # A pretraining corpus is not merely unscanned: `grep` reads the
            # datasets-server's Parquet conversion, which these repos have none
            # of, so no argument to this command would have covered them.
            "status": UNSEARCHED if s.get("hf_dataset") else UNREACHABLE,
            "rows": None, "hits": None, "rate": None, "by_group": {}, "fields": [],
            "available_fields": [], "coverage_gaps": [],
            "produced": None, "produced_rate": None,
            "produced_rate_hi": None,
            "sources": [], "concentration": None, "concentration_all": None,
            "concentration_produced": None, "conc_side": "all",
            "revision": None, "partial": False,
            "unsearched_columns": [], "rank_block": None,
        })

    searched = [r for r in rows if r["status"] in (HITS, ZERO)]
    inconclusive = [r for r in rows if r["status"] == INCONCLUSIVE]
    hitting = [r for r in rows if r["status"] == HITS]
    # Rank on the produce side where any stage measured it, because that is the
    # text the model is trained to emit; fall back to the overall rate when no
    # run searched a produce-side column, so a prompt-only sweep still ranks.
    basis = "produced" if any(r["produced_rate"] for r in hitting) else "rows"
    # Falling back to the overall rate has two causes worth telling apart: no
    # run read a produce-side column, or every run did and found nothing there.
    produce_searched = any(r["produced"] is not None for r in searched + inconclusive)
    key = "produced_rate" if basis == "produced" else "rate"
    for r in hitting:
        r["rank_block"] = _rank_block(r, basis, key)
        # Explain the number that was ranked on. Under the produce-side basis a
        # source that supplied only prompts supplied none of the evidence, and
        # no concentration at all is the right answer where the produce-side
        # matches are spread — falling back to the all-hits reading there would
        # reintroduce exactly the mis-attribution.
        # Needs per-source group counts to be there at all: a run written
        # without `by_source_group` cannot attribute a side, and saying so by
        # dropping the concentration line entirely would be worse than the
        # weaker all-matches reading.
        by_side = any(s["produced_hits"] is not None for s in r["sources"])
        r["conc_side"] = "produced" if (basis == "produced" and r["produced"] and by_side) else "all"
        r["concentration"] = (r["concentration_produced"] if r["conc_side"] == "produced"
                              else r["concentration_all"])
    rankable = [r for r in hitting if not r["rank_block"]]
    unranked = [r for r in hitting if r["rank_block"]]
    ranked = sorted(rankable, key=lambda r: r[key] or 0, reverse=True)

    # The produce-side rate of a stage matching in two groups is an interval,
    # so an order read off the low ends is only real where the intervals are
    # disjoint. Anything whose upper bound clears the leader's lower bound
    # could be the leader.
    contenders = []
    if ranked and basis == "produced":
        floor = ranked[0]["produced_rate"]
        # `>=`, not `>`: an upper bound that lands exactly on the leader's lower
        # bound permits equality, and two equal exact rates are the same case.
        # A tie the sort broke silently is still a tie.
        contenders = [r for r in ranked[1:] if (r["produced_rate_hi"] or 0) >= floor]

    first = results[0] if results else {}
    return {
        "pattern": first.get("pattern"),
        "slug": first.get("slug"),
        "regex": bool(first.get("regex")),
        "case_sensitive": bool(first.get("case_sensitive")),
        "stages": rows,
        "searched": searched,
        "ranked": ranked,
        "unranked": unranked,
        "contenders": contenders,
        "zero": [r for r in searched if r["status"] == ZERO],
        "inconclusive": inconclusive,
        "unsearched": [r for r in rows if r["status"] == UNSEARCHED],
        "unreachable": [r for r in rows if r["status"] == UNREACHABLE],
        "produce_searched": produce_searched,
        "best": ranked[0] if ranked else None,
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "basis": basis,
    }


def _pct(x: float | None) -> str:
    """A rate at two significant figures. These run from 3% down to a handful of
    rows in a hundred thousand, and a fixed number of decimals either rounds the
    small ones to 0.00% or pads the large ones with noise."""
    if x is None:
        return "—"
    if x <= 0:
        return "0%"
    p = x * 100
    if p >= 10:
        return f"{p:.0f}%"
    decimals = 1
    while p < 10 ** (1 - decimals) and decimals < 5:
        decimals += 1
    return f"{p:.{decimals}f}%"


def _span(lo: float | None, hi: float | None) -> str:
    """A rate interval, printed as one figure when both ends round to it. Two
    identical-looking numbers either side of a dash read as a bug, not a bound."""
    a, b = _pct(lo), _pct(hi)
    return a if a == b else f"{a}–{b}"


def _one_in(x: float | None) -> str:
    """The same rate as odds, which is what makes two stages comparable by eye."""
    if not x:
        return ""
    return f" (1 in {round(1 / x):,})"


def _group_line(r: dict) -> str:
    """prompt/response/reference counts, saying which of them were not counted.

    A zero and a blank are different claims and the difference is invisible in
    the count alone: `--field prompt` leaves the response side unread, and a DPO
    mix has no reference column to read. Runs recorded before `available_fields`
    existed cannot tell those two apart, so they say the weaker thing.
    """
    parts = []
    for g in ("prompt", *PRODUCE):
        if g in r["fields"]:
            parts.append(f"{g} {r['by_group'].get(g, 0):,}")
        elif r["available_fields"]:
            parts.append(f"{g} {'not searched' if g in r['available_fields'] else 'no such column'}")
        else:
            parts.append(f"{g} not counted")
    return " · ".join(parts)


def _stage_lines(r: dict) -> list[str]:
    if r["status"] == INCONCLUSIVE:
        why = "; ".join(r["coverage_gaps"])
        return [f"- **{r['stage']}** — nothing matched in the {r['rows']:,} rows read, and that "
                f"is not a zero for the stage: {why}. What went unread could hold it."]
    if r["status"] == ZERO:
        out = [f"- **{r['stage']}** — 0 of {r['rows']:,} rows. Exact, over every row of "
               f"`{r['dataset']}` at revision `{(r['revision'] or '')[:12]}`."]
    else:
        out = [
            f"- **{r['stage']}** — {r['hits']:,} of {r['rows']:,} rows, "
            f"{_pct(r['rate'])}{_one_in(r['rate'])}.  {_group_line(r)}"
        ]
        lo, hi = r["produced"] or (None, None)
        if lo is not None:
            span = f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"
            out.append(f"  - produce side: {span} rows, "
                       f"{_span(r['produced_rate'], r['produced_rate_hi'])} of the stage.")
        if r["rank_block"]:
            out.append(f"  - not in the ranking: {r['rank_block']} — a gap in the measurement "
                       "rather than a low score.")
        side = r["conc_side"]
        hk, rk, lk = (("hits", "rate", "lift") if side == "all"
                      else ("produced_hits", "produced_rate", "produced_lift"))
        what = "produce side" if side == "produced" else "source"
        src = r["concentration"]
        if src:
            out.append(f"  - {what} concentrated in `{src['name']}`: {src[hk]:,} of its "
                       f"{src['rows']:,} rows, {_pct(src[rk])} — {src[lk]:.0f}× the "
                       "stage's own rate.")
        elif r["sources"]:
            top = max(r["sources"], key=lambda x: x[hk] or 0)
            stage_rate = r["produced_rate"] if side == "produced" else r["rate"]
            where = (f"{top[hk]:,} of its {top['rows']:,} rows, {_pct(top[rk])}"
                     if top["rows"] and top[hk] is not None else f"{top['hits']:,} matching rows")
            out.append(f"  - no {what} concentration: the largest contributor `{top['name']}` "
                       f"holds {where}, against {_pct(stage_rate)} for the stage.")
    if r["partial"]:
        out.append("  - the server converted only part of this repo, so the count is a lower "
                   "bound and the rate is over the converted part alone.")
    if r["unsearched_columns"]:
        out.append(f"  - not searched: {', '.join(r['unsearched_columns'])} — text columns this "
                   "layer does not recognise as prompt, response or reference.")
    return out


def _gap_lines(t: dict, cmd: str) -> list[str]:
    """The stages with no count, said once each rather than once per stage.

    They are the half of the answer that gets dropped: a reader who sees three
    stages listed and two of them empty concludes the string is rare, when the
    stage most likely to hold it was never opened.
    """
    out = []
    if t["unsearched"]:
        stages = [r["stage"] for r in t["unsearched"]]
        # `--stage` takes one value, so the suggestion is one run per stage
        # rather than a flag repeated into a command that would silently
        # search only the last of them.
        again = f", then the same for {', '.join(stages[1:])}" if len(stages) > 1 else ""
        out.append(f"- **{', '.join(stages)}** — not searched. That is not a zero: no row of "
                   f"{'these stages' if len(stages) > 1 else 'this stage'} has been "
                   f"read for this pattern (`{cmd} --stage {stages[0]}`{again}).")
    if t["unreachable"]:
        names = ", ".join(r["stage"] for r in t["unreachable"])
        out.append(f"- **{names}** — out of reach for this layer, which is also not a zero: the "
                   "pretraining corpora have no Parquet conversion to scan, and only their "
                   "300-document samples can be searched at all.")
    return out


def _verdict(t: dict) -> list[str]:
    best, other = t["best"], t["runner_up"]
    if best is None:
        # Only a stage read end to end and found empty can carry a zero, so the
        # claim runs over `zero` rather than over everything searched. A stage
        # that matched and could not be ranked is the opposite of a zero, and
        # summing its rows into one would print "0 of N rows" directly under
        # its own match count.
        exact = t["zero"]
        loose = ", ".join(r["stage"] for r in t["inconclusive"])
        if t["unranked"]:
            found = "; ".join(f"{r['stage']} ({r['hits']:,} of {r['rows']:,} rows, but "
                              f"{r['rank_block']})" for r in t["unranked"])
            rest = ""
            if exact:
                rest += (" " + ", ".join(r["stage"] for r in exact)
                         + " matched nothing, over every row.")
            if loose:
                rest += f" {loose} matched nothing in what was read."
            return [f"**Found, and not comparable.** {found}. No stage here carries a rate that "
                    f"ranks against another, so there is no most-plausible answer to give — the "
                    f"counts stand on their own.{rest}"]
        if not exact:
            if loose:
                return [f"**Inconclusive.** Nothing matched in {loose}, but no stage here was "
                        "read end to end — see each line for what went unread — so this is not "
                        "a zero."]
            return ["**Nothing searched yet**, so there is no answer here either way."]
        rows = sum(r["rows"] for r in exact)
        names = ", ".join(r["stage"] for r in exact)
        them = "they are" if len(t["inconclusive"]) > 1 else "it is"
        caveat = (f" {loose} matched nothing either, but was not read end to end, so {them} "
                  "outside this claim.") if loose else ""
        return [
            f"**Not in any stage searched.** 0 of {rows:,} rows across {names}, exact over "
            f"every one of them.{caveat} A model that produces this string anyway did not take "
            "it from the data we can see: that points at a stage this layer cannot reach, at "
            "text distilled from another model rather than carried across literally, or at "
            "generalisation. It is also only a lower bound on the concept — a spelling, "
            "casing or Unicode variant of the same phrase is a different pattern and was "
            "not counted."
        ]

    key = "produced_rate" if t["basis"] == "produced" else "rate"
    measure = "produce-side rate" if t["basis"] == "produced" else "hit rate"
    line = [f"**Most plausibly {best['stage']}.**"]
    src, on_produce = best["concentration"], best["conc_side"] == "produced"
    bounds = best["produced"]
    if src and on_produce:
        span = f"{bounds[0]:,}" if bounds[0] == bounds[1] else f"{bounds[0]:,}–{bounds[1]:,}"
        line.append(f"{src['produced_hits']:,} of the {src['rows']:,} `{src['name']}` rows "
                    f"({_pct(src['produced_rate'])}, {src['produced_lift']:.0f}× the stage) hold "
                    f"it in a response or a reference answer, out of the stage's {span} "
                    "produce-side matches.")
        return _tail(t, best, other, key, measure, line)
    if src:
        line.append(f"{src['hits']:,} of the {src['rows']:,} `{src['name']}` rows "
                    f"({_pct(src['rate'])}, {src['lift']:.0f}× the stage) hold it,")
    else:
        line.append(f"{best['hits']:,} of its {best['rows']:,} rows hold it, spread across its "
                    "sources at roughly the rate their sizes predict,")
    if bounds is None:
        # A prompt-only sweep found every one of its matches in the prompt
        # because that is the only place it looked, which is a weaker claim than
        # the same sentence about a run that read both sides.
        line.append("all of them in the prompt, which is the only side this run searched.")
    elif bounds[0]:
        lo, hi = bounds
        span = f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"
        line.append(f"and {span} of the stage's matches are on the produce side.")
    else:
        line.append("none of them on the produce side: the string is in text the model was "
                    "trained to read rather than in text it was trained to write.")
    return _tail(t, best, other, key, measure, line)


def _tail(t, best, other, key, measure, line):
    """The part of a verdict that is the same however the leader was described."""
    if other and other[key] and best[key]:
        ratio = best[key] / other[key]
        line.append(f"Highest {measure} of the {len(t['ranked'])} stages with any: {ratio:.1f}× "
                    f"{other['stage']}'s.")
        if best["hits"] < other["hits"]:
            line.append(f"The raw counts rank them the other way — {other['stage']} holds "
                        f"{other['hits']:,} matching rows to {best['stage']}'s {best['hits']:,} — "
                        "which is what normalising is for.")
    if t["contenders"]:
        # Two overlapping intervals do not order each other, and the low ends
        # they were sorted on are an artefact of how they were sorted.
        names = ", ".join(f"{r['stage']} ({_span(r['produced_rate'], r['produced_rate_hi'])})"
                          for r in t["contenders"])
        line.append("Read that as a lead rather than a result: the produce-side count of a "
                    "stage matching in two groups is only known to an interval, and "
                    f"{best['stage']}'s {_span(best['produced_rate'], best['produced_rate_hi'])} "
                    f"overlaps {names}, so the counts do not settle the order between them.")
    for r in t["unranked"]:
        line.append(f"{r['stage']} matched too and is not in this ranking at all rather than at "
                    f"the bottom of it: {r['rank_block']}.")
    if t["basis"] == "rows" and not t["produce_searched"]:
        line.append("No run on this pattern searched a produce-side column, so this is the hit "
                    "rate over all rows and says nothing about which side matched.")
    elif t["basis"] == "rows":
        line.append("No stage matched on the produce side at all, so this ranks by the overall "
                    "hit rate: the string is in text these stages train the model to read.")
    return [" ".join(line)]


BASIS_NOTE = (
    "Each stage is ranked by the share of its own rows that match, on the produce side where "
    "any run searched one — the response a stage fits or the reference a verifier scores "
    "against, as opposed to the prompt the model only reads. Raw hits are not comparable "
    "across stages: a mix with ten times the rows shows more of any string for that reason "
    "alone. Stage order is deliberately not weighted, so a rate in RLVR and the same rate in "
    "pretraining rank equal here even though the late one generally moves behaviour more."
)


def render(t: dict, model: str, note: bool = False) -> list[str]:
    """The comparison as markdown, which is also what it prints to a terminal."""
    flags = "".join([
        " --regex" if t["regex"] else "",
        " --case-sensitive" if t["case_sensitive"] else "",
        # Without the slug a re-run derives its own from the pattern and lands
        # in a different group, which is how a trace ends up split in two.
        f' --slug {t["slug"]}' if t.get("slug") else "",
    ])
    cmd = f'trainspotting grep {model} "{t["pattern"]}"{flags}'
    out = [f"### `{t['pattern']}` — where it most plausibly comes from", ""]
    if note:
        out += [BASIS_NOTE, ""]
    for r in t["stages"]:
        if r["status"] in (HITS, ZERO, INCONCLUSIVE):
            out += _stage_lines(r)
    out += _gap_lines(t, cmd)
    out.append("")
    out += _verdict(t)
    out.append("")
    return out
