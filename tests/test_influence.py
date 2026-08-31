"""Reading a set of grep runs as one claim about where a string came from.

All offline and all pure: the input is the shape `grep` writes to results/, and
the failure modes being guarded against are the ones that turn a count into a
wrong story — ranking by hits instead of by rate, adding overlapping group
counts as if they were a union, and printing a stage nobody scanned as a zero.
"""

import pytest

from trainspotting import influence, registry

THINK = registry.MODELS["olmo-3-7b-think"]["stages"]


def run(stage, *, hits, rows, groups, fields=None, sources=None, totals=None, **kw):
    """One grep result file, with only the keys the influence layer reads.

    `available_fields` defaults to the fields searched — a run that read
    everything the mix has, which is what a modern unnarrowed `grep` writes.
    Pass a wider list to model `--field` narrowing, or `[]` to model a run
    written before the key existed.
    """
    searched = fields if fields is not None else list(groups)
    available = kw.pop("available_fields", searched)
    out = {
        "stage": stage,
        "dataset": f"allenai/{stage}",
        "pattern": kw.pop("pattern", "as an AI language model"),
        "regex": kw.pop("regex", False),
        "case_sensitive": kw.pop("case_sensitive", False),
        "available_fields": list(available),
        "fields": searched,
        "matched": hits,
        "total_rows": rows,
        "by_group": groups,
        "by_source": sources or {},
        "rows_by_source": totals or {},
        "by_source_group": {},
        "revision": "abc123def456789",
        "partial": False,
        "unsearched_columns": [],
        **kw,
    }
    if not available:
        del out["available_fields"]
    return out


# --- the produce-side union ------------------------------------------------


def test_two_produce_groups_give_an_interval_not_a_sum():
    # 3 response rows and 232 reference rows out of 400 matches: the union is at
    # least 232 and at most 235, and reporting 235 as fact would double-count
    # every row that matched on both.
    r = run("rlvr", hits=400, rows=102_014, groups={"prompt": 209, "response": 3, "reference": 232})
    assert influence.produced(r) == (232, 235)


def test_the_interval_is_capped_by_the_rows_that_matched_at_all():
    r = run("rlvr", hits=10, rows=100, groups={"prompt": 5, "response": 8, "reference": 7})
    assert influence.produced(r) == (8, 10)


def test_rows_that_did_not_match_a_prompt_settle_the_interval():
    # Every matched row matched somewhere, so a row outside the prompt group is
    # on the produce side. With no prompt match at all the union is exact.
    r = run("rlvr", hits=180, rows=1_000, groups={"prompt": 0, "response": 100, "reference": 80})
    assert influence.produced(r) == (180, 180)


def test_the_prompt_bound_tightens_without_settling():
    r = run("rlvr", hits=180, rows=1_000, groups={"prompt": 40, "response": 100, "reference": 80})
    assert influence.produced(r) == (140, 180)


def test_a_produce_only_run_needs_no_prompt_count():
    # `--field response --field reference`: `matched` is already the union.
    r = run("rlvr", hits=180, rows=1_000, groups={"response": 100, "reference": 80},
            fields=["response", "reference"])
    assert influence.produced(r) == (180, 180)


def test_inconsistent_counts_do_not_invert_the_interval():
    r = run("rlvr", hits=200, rows=1_000, groups={"prompt": 0, "response": 60, "reference": 60})
    lo, hi = influence.produced(r)
    assert lo <= hi == 120


def test_one_produce_group_is_exact():
    r = run("dpo", hits=61, rows=150_000, groups={"prompt": 46, "response": 39})
    assert influence.produced(r) == (39, 39)


def test_a_prompt_only_run_has_no_produce_side_at_all():
    # Not zero: `--field prompt` never read the response columns, so there is
    # no count to report either way.
    r = run("dpo", hits=293, rows=150_000, groups={"prompt": 293}, fields=["prompt"])
    assert influence.produced(r) is None


# --- ranking ---------------------------------------------------------------


def test_the_smaller_count_wins_on_the_higher_rate():
    runs = [
        run("dpo", hits=900, rows=1_000_000, groups={"prompt": 0, "response": 900}),
        run("rlvr", hits=100, rows=10_000, groups={"prompt": 0, "response": 100}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"
    assert t["basis"] == "produced"


def test_the_disagreement_between_hits_and_rate_is_stated():
    runs = [
        run("dpo", hits=900, rows=1_000_000, groups={"prompt": 0, "response": 900}),
        run("rlvr", hits=100, rows=10_000, groups={"prompt": 0, "response": 100}),
    ]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "rank them the other way" in text
    assert "900" in text and "100" in text


def test_prompt_side_hits_lose_to_produce_side_hits_at_a_lower_rate():
    # A phrase the model was trained to read is weaker evidence for a phrase it
    # emits than one it was trained to write, even at ten times the rate.
    runs = [
        run("dpo", hits=1_000, rows=100_000, groups={"prompt": 1_000, "response": 0}),
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"


def test_a_sweep_with_no_produce_column_still_ranks_by_rate():
    runs = [
        run("dpo", hits=293, rows=150_000, groups={"prompt": 293}, fields=["prompt"]),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 10}, fields=["prompt"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "rows"
    assert t["best"]["stage"] == "dpo"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "says nothing about which side matched" in text


# --- the three ways of having no number ------------------------------------


def test_a_searched_zero_is_an_answer_and_says_so():
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0})]
    t = influence.compare(runs, THINK)
    assert t["best"] is None
    assert [r["stage"] for r in t["zero"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "0 of 150,000 rows" in text
    assert "Not in any stage searched" in text
    # The three readings of an honest zero.
    assert "cannot reach" in text and "distilled" in text and "generalisation" in text


def test_an_unscanned_stage_is_not_reported_as_a_zero():
    runs = [run("rlvr", hits=5, rows=100_000, groups={"prompt": 5})]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["unsearched"]] == ["sft", "dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "sft, dpo** — not searched" in text
    assert "not a zero" in text


def test_a_pretraining_stage_is_out_of_reach_rather_than_unsearched():
    t = influence.compare([run("rlvr", hits=5, rows=100_000, groups={"prompt": 5})], THINK)
    assert [r["stage"] for r in t["unreachable"]] == ["pretrain", "midtrain", "long-context"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "pretrain, midtrain, long-context** — out of reach" in text


def test_nothing_searched_at_all_claims_nothing():
    t = influence.compare([], THINK)
    assert t["best"] is None
    assert "Nothing searched yet" in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- normalising inside a stage --------------------------------------------


def test_the_source_denominator_is_the_source_not_the_mix():
    runs = [run(
        "dpo", hits=434, rows=150_000, groups={"prompt": 0, "response": 434},
        sources={"filtered_wc_sample_500k": 434}, totals={"filtered_wc_sample_500k": 17_596},
    )]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "434 of the 17,596 `filtered_wc_sample_500k` rows (2.5%, 9× the stage)" in text


def test_the_biggest_source_is_not_the_origin_when_it_holds_the_mix_rate():
    # 267 of a stage's 521 matches is a majority share and says nothing: at
    # 124,980 rows the source holds the string at the rate its size predicts.
    runs = [run(
        "dpo", hits=521, rows=259_922, groups={"prompt": 462, "response": 193},
        sources={"llm_judged": 267, "delta_learning": 247},
        totals={"llm_judged": 124_980, "delta_learning": 116_000},
    )]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"] is None
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "no source concentration" in text
    assert "spread across its sources at roughly the rate their sizes predict" in text


def test_the_concentration_can_be_the_second_source_by_count():
    runs = [run(
        "dpo", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
        sources={"big": 60, "small": 40},
        totals={"big": 60_000, "small": 2_000},
    )]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"]["name"] == "small"
    assert "concentrated in `small`: 40 of its 2,000 rows, 2.0% — 20× the stage" in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_source_contributing_a_handful_of_rows_is_not_the_origin():
    # 2 of 20 rows is a 10% rate and 2% of the stage's matches: a rate computed
    # over a source that small is noise, and naming it would be a story.
    runs = [run(
        "dpo", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
        sources={"big": 98, "tiny": 2},
        totals={"big": 98_000, "tiny": 20},
    )]
    assert influence.compare(runs, THINK)["best"]["concentration"] is None


def test_a_source_with_no_row_count_is_named_without_inventing_a_rate():
    runs = [run("dpo", hits=4, rows=150_000, groups={"prompt": 4}, sources={"mystery": 4})]
    t = influence.compare(runs, THINK)
    assert t["best"]["sources"][0]["rate"] is None
    assert t["best"]["concentration"] is None
    assert "largest contributor `mystery` holds 4 matching rows" in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


# --- what a blank in the group breakdown means -----------------------------


def test_a_narrowed_field_reads_differently_from_a_missing_column():
    narrowed = run(
        "rlvr", hits=5, rows=100, groups={"prompt": 5}, fields=["prompt"],
        available_fields=["prompt", "response", "reference"],
    )
    absent = run(
        "dpo", hits=5, rows=100, groups={"prompt": 5}, fields=["prompt"],
        available_fields=["prompt"],
    )
    line = influence._group_line(influence.stage_trace(narrowed))
    assert "response not searched" in line and "reference not searched" in line
    line = influence._group_line(influence.stage_trace(absent))
    assert "response no such column" in line and "reference no such column" in line


def test_a_run_predating_available_fields_says_the_weaker_thing():
    old = run("dpo", hits=5, rows=100, groups={"prompt": 5}, fields=["prompt"],
              available_fields=[])
    assert "response not counted" in influence._group_line(influence.stage_trace(old))


# --- rate formatting -------------------------------------------------------


@pytest.mark.parametrize(("rate", "shown"), [
    (0.0392, "3.9%"),
    (0.00392, "0.39%"),
    (0.000392, "0.039%"),
    (0.0000392, "0.0039%"),
    (0.0, "0%"),
    (None, "—"),
])
def test_two_significant_figures_survive_four_orders_of_magnitude(rate, shown):
    assert influence._pct(rate) == shown


# --- the suggested re-run --------------------------------------------------


def test_the_suggested_command_keeps_the_slug_and_the_regex_flag():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5},
                regex=True, slug="openai-identity", pattern="I am (ChatGPT|Claude)")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert '--regex --slug openai-identity --stage sft' in text


def test_one_stage_per_suggested_run_because_the_flag_takes_one():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5})]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "--stage sft --stage dpo" not in text
    assert "--stage sft`, then the same for dpo" in text


# --- the caveats that travel with a count ----------------------------------


def test_a_partial_conversion_makes_the_count_a_lower_bound():
    runs = [run("dpo", hits=5, rows=100, groups={"prompt": 5}, partial=True)]
    assert "lower bound" in " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))


def test_an_unrecognised_text_column_is_named():
    runs = [run("dpo", hits=5, rows=100, groups={"prompt": 5}, unsearched_columns=["rationale"])]
    assert "rationale" in " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))


def test_searched_and_empty_on_the_produce_side_is_not_the_same_as_unsearched():
    both = [
        run("sft", hits=1, rows=100, groups={"prompt": 1, "response": 0}),
        run("dpo", hits=10, rows=100, groups={"prompt": 10, "response": 0}),
        run("rlvr", hits=1, rows=100, groups={"prompt": 1, "response": 0}),
    ]
    t = influence.compare(both, THINK)
    assert t["basis"] == "rows" and t["produce_searched"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "No stage matched on the produce side at all" in text
    assert "No run on this pattern searched a produce-side column" not in text


# --- what a missing measurement is not: zero -------------------------------


def test_a_prompt_only_run_is_held_out_of_the_produce_side_ranking():
    # `--field` is per run, so one pattern's stages can disagree about what was
    # read. Sorting the unmeasured stage as 0.0 would rank it as though its
    # produce side had been searched and found empty.
    runs = [
        run("dpo", hits=1_000, rows=10_000, groups={"prompt": 1_000}, fields=["prompt"]),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "produced"
    assert [r["stage"] for r in t["ranked"]] == ["rlvr"]
    assert [r["stage"] for r in t["unranked"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "not in the ranking: this run read only prompt, so there is no produce-side rate" in text
    assert "dpo matched too and is not in this ranking at all rather than at the bottom" in text


def test_a_produce_side_searched_and_empty_still_ranks():
    # The other half of the same distinction: this run did read the response
    # column, so a zero there is a measurement and belongs in the ranking.
    runs = [
        run("dpo", hits=1_000, rows=10_000, groups={"prompt": 1_000, "response": 0}),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["ranked"]] == ["rlvr", "dpo"]
    assert t["unranked"] == []


# --- an interval does not order another interval ---------------------------


def test_overlapping_produce_side_intervals_are_called_unsettled():
    # 60–120 against an exact 70 over equal row counts: the low ends order them
    # one way and the counts do not settle it.
    runs = [
        run("rlvr", hits=200, rows=100_000,
            groups={"prompt": 140, "response": 60, "reference": 60}),
        run("dpo", hits=70, rows=100_000, groups={"prompt": 0, "response": 70}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "dpo"
    assert [r["stage"] for r in t["contenders"]] == ["rlvr"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "do not settle the order between them" in text


def test_disjoint_intervals_are_not_called_unsettled():
    # The shipped runs look like this: 232–235 of 102,014 against 39 of 150,000.
    runs = [
        run("rlvr", hits=400, rows=102_014, groups={"prompt": 209, "response": 3, "reference": 232}),
        run("dpo", hits=61, rows=150_000, groups={"prompt": 46, "response": 39}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"
    assert t["contenders"] == []
    assert "do not settle the order" not in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a partial conversion cannot produce a zero ----------------------------


def test_no_hits_over_a_partial_conversion_is_inconclusive_not_zero():
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0}, partial=True)]
    t = influence.compare(runs, THINK)
    assert t["searched"] == [] and [r["stage"] for r in t["inconclusive"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "the server converted only part of the repo" in text
    assert "not a zero for the stage" in text
    assert "**Inconclusive.**" in text
    # The exact-zero reading must not be reached on data that was never read.
    assert "exact over every one of them" not in text
    assert "did not take it from the data we can see" not in text


def test_a_partial_zero_does_not_join_a_real_zero_in_the_verdict():
    runs = [
        run("rlvr", hits=0, rows=102_014, groups={"prompt": 0, "response": 0}),
        run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0}, partial=True),
    ]
    t = influence.compare(runs, THINK)
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    # The exact claim is made over rlvr's rows alone, not over both stages'.
    assert "0 of 102,014 rows across rlvr, exact over every one of them" in text
    assert "dpo matched nothing either, but was not read end to end, so it is outside this claim" in text


# --- a zero is only stage-wide if the whole stage was read -----------------


def test_a_narrowed_run_that_matched_nothing_is_not_a_stage_zero():
    # `--field prompt` never opened the response column, so "no matches" is a
    # fact about the prompts and the verdict must not read it as the stage.
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0}, fields=["prompt"],
                available_fields=["prompt", "response"])]
    t = influence.compare(runs, THINK)
    assert t["searched"] == [] and [r["stage"] for r in t["inconclusive"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "this run read only prompt of prompt, response" in text
    assert "exact over every one of them" not in text
    assert "did not take it from the data we can see" not in text


def test_an_unrecognised_text_column_also_blocks_a_stage_zero():
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0},
                unsearched_columns=["rationale"])]
    t = influence.compare(runs, THINK)
    assert t["inconclusive"] and t["searched"] == []
    assert "rationale went unsearched" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_run_that_cannot_show_its_coverage_does_not_get_the_benefit_of_doubt():
    # Written before `available_fields` existed: it may have read everything,
    # and it cannot demonstrate that, which is not the same as having.
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0},
                available_fields=[])]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["inconclusive"]] == ["dpo"]
    assert "does not record which sides the mix has" in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_complete_run_that_matched_nothing_is_still_a_real_zero():
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0})]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["zero"]] == ["dpo"]
    assert "exact over every one of them" in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a subset rate does not rank against a stage rate ----------------------


def test_a_partial_conversion_with_hits_is_kept_out_of_the_ranking():
    # Its denominator is the converted subset, and the conversion is a prefix
    # rather than a sample, so the quotient is not a rate for the stage.
    runs = [
        run("dpo", hits=900, rows=1_000, groups={"prompt": 0, "response": 900}, partial=True),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"
    assert [r["stage"] for r in t["unranked"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "not in the ranking: only part of this repo was converted" in text
    assert "dpo matched too and is not in this ranking" in text


# --- one slug is not a promise that it is one search -----------------------


def test_two_different_searches_are_refused_rather_than_ranked():
    runs = [
        run("dpo", hits=5, rows=100, groups={"prompt": 5}, pattern="ChatGPT"),
        run("rlvr", hits=5, rows=100, groups={"prompt": 5}, pattern="ChatGPT", regex=True),
    ]
    with pytest.raises(ValueError, match="not the same search"):
        influence.compare(runs, THINK)


def test_matching_flags_alone_make_it_a_different_search():
    runs = [
        run("dpo", hits=5, rows=100, groups={"prompt": 5}),
        run("rlvr", hits=5, rows=100, groups={"prompt": 5}, case_sensitive=True),
    ]
    with pytest.raises(ValueError, match="not the same search"):
        influence.compare(runs, THINK)


# --- matches are never a zero, however unrankable --------------------------


def test_hits_that_cannot_be_ranked_do_not_become_a_zero():
    # Every hitting stage blocked from the ranking left `best` None, and the
    # zero branch then summed the matching stage's rows into "0 of N rows" —
    # printed directly under its own 900 matches.
    runs = [run("dpo", hits=900, rows=1_000, groups={"prompt": 0, "response": 900}, partial=True)]
    t = influence.compare(runs, THINK)
    assert t["best"] is None and [r["stage"] for r in t["unranked"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "**Found, and not comparable.**" in text
    assert "dpo (900 of 1,000 rows, but only part of this repo was converted" in text
    assert "Not in any stage searched" not in text
    assert "exact over every one of them" not in text


def test_a_real_zero_alongside_unrankable_hits_is_still_reported():
    runs = [
        run("dpo", hits=900, rows=1_000, groups={"prompt": 0, "response": 900}, partial=True),
        run("rlvr", hits=0, rows=102_014, groups={"prompt": 0, "response": 0}),
    ]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "**Found, and not comparable.**" in text
    assert "rlvr matched nothing, over every row" in text


# --- produce-side evidence is attributed to produce-side sources -----------


def test_the_produce_side_verdict_names_a_produce_side_source():
    # `noisy` holds most of the matches and all of them are prompts, so it
    # supplied none of the evidence the produce-side ranking ran on.
    runs = [run(
        "dpo", hits=100, rows=100_000, groups={"prompt": 80, "response": 20},
        sources={"noisy": 80, "quiet": 20},
        totals={"noisy": 4_000, "quiet": 2_000},
        by_source_group={"noisy": {"prompt": 80, "response": 0},
                         "quiet": {"prompt": 0, "response": 20}},
    )]
    t = influence.compare(runs, THINK)
    best = t["best"]
    assert best["conc_side"] == "produced"
    assert best["concentration"]["name"] == "quiet"
    # The all-matches reading would have named the other one.
    assert best["concentration_all"]["name"] == "noisy"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "produce side concentrated in `quiet`: 20 of its 2,000 rows" in text
    assert "`noisy`" not in text


def test_produce_side_matches_spread_thin_are_not_attributed_to_a_prompt_source():
    runs = [run(
        "dpo", hits=100, rows=100_000, groups={"prompt": 90, "response": 10},
        sources={"noisy": 90, "quiet": 10},
        totals={"noisy": 4_000, "quiet": 100_000},
        by_source_group={"noisy": {"prompt": 90, "response": 0},
                         "quiet": {"prompt": 0, "response": 10}},
    )]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"] is None
    assert "no produce side concentration" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_run_without_per_source_groups_falls_back_to_all_matches():
    runs = [run("dpo", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
                sources={"only": 100}, totals={"only": 2_000})]
    t = influence.compare(runs, THINK)
    assert t["best"]["conc_side"] == "all"
    assert t["best"]["concentration"]["name"] == "only"


# --- a touching interval is a tie ------------------------------------------


def test_an_upper_bound_landing_on_the_leaders_lower_bound_is_a_tie():
    # rlvr's produce side is 50–100 of 100,000; dpo's is exactly 50 of 50,000,
    # so rlvr's upper bound equals dpo's rate and the two could be equal.
    runs = [
        run("rlvr", hits=200, rows=100_000,
            groups={"prompt": 150, "response": 50, "reference": 50}),
        run("dpo", hits=50, rows=50_000, groups={"prompt": 0, "response": 50}),
    ]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["contenders"]] == ["rlvr"]
    assert "do not settle the order" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_two_equal_exact_produce_side_rates_are_a_tie():
    runs = [
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert len(t["contenders"]) == 1
    assert "do not settle the order" in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a narrowed run's rate is a floor, so it can win but not lose ----------


def test_a_stage_that_did_not_read_its_whole_produce_side_is_flagged_when_it_loses():
    # rlvr read response and reference; dpo's mix has both and its run read
    # only response, so dpo's produce-side rate is a floor. It lost, and the
    # columns nobody opened could put it on top.
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
            fields=["prompt", "response"], available_fields=["prompt", "response"]),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10},
            fields=["prompt", "response"], available_fields=["prompt", "response", "reference"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"
    assert [r["stage"] for r in t["understated"]] == ["dpo"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "dpo's rate is a floor rather than a figure" in text
    assert "could be smaller than this, or run the other way" in text


def test_a_leader_whose_rate_is_a_floor_is_not_flagged():
    # A floor above the runner-up still beats the runner-up.
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
            fields=["prompt", "response"], available_fields=["prompt", "response", "reference"]),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr" and t["understated"] == []


def test_an_unread_prompt_does_not_understate_a_produce_side_rate():
    # Only the produce-side columns can move a produce-side rate.
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100}),
        run("dpo", hits=10, rows=100_000, groups={"response": 10}, fields=["response"],
            available_fields=["prompt", "response"]),
    ]
    assert influence.compare(runs, THINK)["understated"] == []


def test_the_row_basis_does_not_claim_an_absent_produce_side_it_never_read():
    # dpo never opened a produce-side column, so "no stage matched on the
    # produce side" is a claim about rlvr's columns, not about the stages.
    runs = [
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100}, fields=["prompt"],
            available_fields=["prompt", "response"]),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 10, "response": 0}),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "rows" and t["produce_searched"] and not t["produce_complete"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "cannot say the string is absent from what these stages train the model to write" in text
    assert "No stage matched on the produce side at all" not in text


def test_the_row_basis_may_claim_it_when_every_stage_was_read():
    runs = [
        run("sft", hits=5, rows=100_000, groups={"prompt": 5, "response": 0}),
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 10, "response": 0}),
    ]
    t = influence.compare(runs, THINK)
    assert t["produce_complete"] and not t["unsearched"]
    assert "No stage matched on the produce side at all" in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


def test_an_unread_stage_keeps_the_produce_side_absence_local():
    # The same runs minus SFT: every stage read here looked at its produce side
    # and found nothing, and SFT can still hold anything.
    runs = [
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 10, "response": 0}),
    ]
    t = influence.compare(runs, THINK)
    assert t["produce_complete"] and [r["stage"] for r in t["unsearched"]] == ["sft"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Nothing matched on the produce side of the stages read here" in text
    assert "sft has not been read, so it is not a claim about the pipeline" in text
    assert "No stage matched on the produce side at all" not in text


# --- the share floor is a share of the evidence ----------------------------


def test_produce_side_evidence_spread_over_many_sources_names_none_of_them():
    # Ten sources of ten produce-side rows each; the eleventh has two at a high
    # rate. Measured against the largest source the floor is one row, so that
    # two-row source cleared it and was named as the origin of 2% of the
    # evidence.
    sources = {f"s{i}": 10 for i in range(10)} | {"tiny": 2}
    totals = {f"s{i}": 10_000 for i in range(10)} | {"tiny": 20}
    by_group = {f"s{i}": {"prompt": 0, "response": 10} for i in range(10)}
    by_group["tiny"] = {"prompt": 0, "response": 2}
    runs = [run("dpo", hits=102, rows=1_000_000, groups={"prompt": 0, "response": 102},
                sources=sources, totals=totals, by_source_group=by_group)]
    t = influence.compare(runs, THINK)
    assert t["best"]["conc_side"] == "produced"
    assert t["best"]["concentration"] is None


def test_a_source_holding_most_of_the_produce_side_evidence_still_counts():
    runs = [run("dpo", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
                sources={"big": 90, "rest": 10},
                totals={"big": 1_000, "rest": 50_000},
                by_source_group={"big": {"prompt": 0, "response": 90},
                                 "rest": {"prompt": 0, "response": 10}})]
    assert influence.compare(runs, THINK)["best"]["concentration"]["name"] == "big"


# --- a zero over some stages is not a zero over the pipeline ---------------


def test_an_unscanned_stage_is_the_first_explanation_for_a_zero():
    # `grep --stage dpo` finding nothing does not make distillation the
    # leading theory while SFT sits unread.
    runs = [run("dpo", hits=0, rows=150_000, groups={"prompt": 0, "response": 0})]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "It may simply be in sft, rlvr, which no run has read" in text
    assert text.index("may simply be in sft") < text.index("distilled")


def test_a_complete_sweep_says_the_reachable_stages_are_exhausted():
    runs = [run(s, hits=0, rows=1_000, groups={"prompt": 0, "response": 0})
            for s in ("sft", "dpo", "rlvr")]
    t = influence.compare(runs, THINK)
    assert t["unsearched"] == []
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Every stage this layer can reach has now been read" in text
    assert "may simply be in" not in text


# --- the produce-side union, bounded as tightly as the counts allow --------


def test_a_share_is_guaranteed_against_the_top_of_the_interval():
    # Stage bounds 100–200; per-source floors 10 and 90. Measured against the
    # floor the ten-row source clears a floor of 10 and is named — while the
    # other source may hold 180 distinct rows, leaving its real share at 10/190.
    runs = [run(
        "rlvr", hits=200, rows=1_000_000,
        groups={"prompt": 100, "response": 100, "reference": 100},
        sources={"big": 180, "tiny": 20},
        totals={"big": 180_000, "tiny": 100},
        by_source_group={"big": {"prompt": 90, "response": 90, "reference": 90},
                         "tiny": {"prompt": 10, "response": 10, "reference": 10}},
    )]
    t = influence.compare(runs, THINK)
    assert t["best"]["produced"] == (100, 200)
    srcs = t["best"]["sources"]
    # Against the floor of 100 the ten-row source wins on rate; against the
    # ceiling of 200 it no longer clears the share floor at all.
    assert influence.concentration(srcs, 100, side="produced")["name"] == "tiny"
    assert t["best"]["concentration"]["name"] == "big"


def test_a_source_clearing_the_floor_against_the_ceiling_still_counts():
    runs = [run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
                sources={"big": 90, "rest": 10},
                totals={"big": 1_000, "rest": 50_000},
                by_source_group={"big": {"prompt": 0, "response": 90},
                                 "rest": {"prompt": 0, "response": 10}})]
    t = influence.compare(runs, THINK)
    assert t["best"]["produced"] == (100, 100)
    assert t["best"]["concentration"]["name"] == "big"


# --- the suggested command is shell ----------------------------------------


def test_a_pattern_with_shell_syntax_is_quoted_not_interpolated():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5},
                pattern='say "$HOME" `id`', slug="risky")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert """trainspotting grep olmo-3-7b-think 'say "$HOME" `id`' --slug risky""" in text
    assert 'grep olmo-3-7b-think "say "$HOME"' not in text


def test_a_pattern_with_a_single_quote_is_still_one_argument():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5}, pattern="it's me", slug="q")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert """'it'"'"'s me'""" in text


def test_an_ordinary_pattern_is_not_dressed_up():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5}, pattern="ChatGPT", slug="chatgpt")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "trainspotting grep olmo-3-7b-think ChatGPT --slug chatgpt" in text


# --- a lift between two floors is not a lift -------------------------------


def test_the_produce_side_lift_is_the_one_the_counts_guarantee():
    # Source floor over stage ceiling. The stage is bounded 100–180 of 10,000
    # (1.0%–1.8%) and the source has 20 of 100 (20%), so the guaranteed lift is
    # 20/1.8 ≈ 11×, not 20/1.0 = 20× as two floors would read.
    runs = [run(
        "rlvr", hits=180, rows=10_000,
        groups={"prompt": 80, "response": 100, "reference": 80},
        sources={"src": 40, "rest": 140},
        totals={"src": 100, "rest": 9_900},
        by_source_group={"src": {"prompt": 20, "response": 20, "reference": 20},
                         "rest": {"prompt": 60, "response": 80, "reference": 60}},
    )]
    t = influence.compare(runs, THINK)
    src = next(s for s in t["best"]["sources"] if s["name"] == "src")
    assert t["best"]["produced"] == (100, 180)
    assert src["produced_hits"] == 20 and src["produced_hits_hi"] == 40
    assert 10 < src["produced_lift"] < 12
    assert "at least 11× the stage's own rate" in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_settled_interval_states_the_lift_plainly():
    runs = [run("rlvr", hits=100, rows=10_000, groups={"prompt": 0, "response": 100},
                sources={"src": 100}, totals={"src": 500},
                by_source_group={"src": {"prompt": 0, "response": 100}})]
    t = influence.compare(runs, THINK)
    assert t["best"]["produced"] == (100, 100)
    assert "at least" not in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a stage-wide claim needs every stage ----------------------------------


def test_a_zero_run_that_never_read_its_produce_side_blocks_the_claim():
    runs = [
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
        run("rlvr", hits=0, rows=100_000, groups={"prompt": 0}, fields=["prompt"],
            available_fields=["prompt", "response"]),
    ]
    t = influence.compare(runs, THINK)
    assert not t["produce_complete"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "cannot say the string is absent from what these stages train the model to write" in text
    assert "No stage matched on the produce side at all" not in text


# --- a ranking whose leader has no evidence elects nobody ------------------


def test_a_zero_produce_side_stage_is_not_elected_over_an_excluded_one():
    # RLVR's response matches are the only produce-side evidence and its scan
    # was partial, so it cannot be ranked. DPO is comparable and measured zero
    # there. Naming DPO would elect the stage with no evidence of the kind the
    # ranking runs on.
    runs = [
        run("rlvr", hits=50, rows=1_000, groups={"prompt": 0, "response": 50}, partial=True),
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "produced"
    assert t["best"] is None and t["ranked"] == []
    assert {r["stage"] for r in t["unranked"]} == {"rlvr", "dpo"}
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "**Found, and not comparable.**" in text
    assert "nothing matched on its produce side, and the stages that did are excluded" in text
    assert "Most plausibly" not in text


def test_a_measured_produce_side_zero_still_ranks_below_real_evidence():
    # The same zero is a fine last place when a comparable stage has evidence.
    runs = [
        run("rlvr", hits=50, rows=100_000, groups={"prompt": 0, "response": 50}),
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
    ]
    t = influence.compare(runs, THINK)
    assert [r["stage"] for r in t["ranked"]] == ["rlvr", "dpo"]
    assert t["unranked"] == []


# --- absence needs every produce column its mix has ------------------------


def test_reading_one_of_two_produce_columns_does_not_prove_absence():
    # `--field prompt --field response` on a mix that also has `reference`:
    # `produced` is (0, 0), which is a measurement of the response column and
    # not of the produce side.
    runs = [
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
        run("rlvr", hits=10, rows=100_000, groups={"prompt": 10, "response": 0},
            fields=["prompt", "response"],
            available_fields=["prompt", "response", "reference"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "rows" and not t["produce_complete"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "cannot say the string is absent from what these stages train the model to write" in text
    assert "No stage matched on the produce side at all" not in text


def test_a_partial_conversion_with_prompt_hits_also_blocks_the_absence_claim():
    runs = [
        run("dpo", hits=100, rows=100_000, groups={"prompt": 100, "response": 0}),
        run("rlvr", hits=10, rows=1_000, groups={"prompt": 10, "response": 0}, partial=True),
    ]
    t = influence.compare(runs, THINK)
    assert not t["produce_complete"]
    assert "No stage matched on the produce side at all" not in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a tie on the row basis is a tie too ----------------------------------


def test_equal_row_basis_rates_are_reported_as_a_tie():
    runs = [
        run("dpo", hits=10, rows=100, groups={"prompt": 10}, fields=["prompt"]),
        run("rlvr", hits=10, rows=100, groups={"prompt": 10}, fields=["prompt"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "rows"
    assert [r["stage"] for r in t["contenders"]] == ["rlvr"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Read that as a tie: dpo, rlvr match at the same rate" in text
    # And no fictitious advantage.
    assert "1.0×" not in text


def test_an_unequal_row_basis_pair_is_not_a_tie():
    runs = [
        run("dpo", hits=20, rows=100, groups={"prompt": 20}, fields=["prompt"]),
        run("rlvr", hits=10, rows=100, groups={"prompt": 10}, fields=["prompt"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["contenders"] == []
    assert "2.0× rlvr's" in " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a source's produce-side count is a bound too --------------------------


def test_a_source_matching_in_both_groups_is_rendered_as_an_interval():
    # 10 response and 10 reference rows in one source is 10–20 distinct rows.
    runs = [run("rlvr", hits=40, rows=100_000,
                groups={"prompt": 20, "response": 10, "reference": 10},
                sources={"src": 40}, totals={"src": 1_000},
                by_source_group={"src": {"prompt": 20, "response": 10, "reference": 10}})]
    t = influence.compare(runs, THINK)
    src = t["best"]["sources"][0]
    assert (src["produced_hits"], src["produced_hits_hi"]) == (20, 20)
    # With no prompt matches in the source the bound is the pair of groups.
    runs[0]["by_source_group"]["src"] = {"prompt": 0, "response": 10, "reference": 10}
    runs[0]["by_source"] = {"src": 20}
    t = influence.compare(runs, THINK)
    src = t["best"]["sources"][0]
    assert (src["produced_hits"], src["produced_hits_hi"]) == (20, 20)


def test_an_open_source_interval_prints_both_ends():
    runs = [run("rlvr", hits=100, rows=100_000,
                groups={"prompt": 60, "response": 30, "reference": 30},
                sources={"src": 100}, totals={"src": 1_000},
                by_source_group={"src": {"prompt": 60, "response": 30, "reference": 30}})]
    t = influence.compare(runs, THINK)
    src = t["best"]["sources"][0]
    assert (src["produced_hits"], src["produced_hits_hi"]) == (40, 60)
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "40–60 of its 1,000 rows" in text


# --- a ratio between bounded rates is a bound ------------------------------


def test_the_advantage_is_the_multiple_the_counts_guarantee():
    # Leader 200–300 of 1,000 (20–30%), runner-up 100–250 of 1,000 (10–25%).
    # Two floors read 2.0×; the guaranteed multiple is 20/25 < 1, so there is
    # no advantage to report and the overlap sentence covers the pair.
    runs = [
        run("rlvr", hits=400, rows=1_000,
            groups={"prompt": 200, "response": 200, "reference": 100}),
        run("dpo", hits=300, rows=1_000,
            groups={"prompt": 200, "response": 100, "reference": 150}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["produced"] == (200, 300) and t["runner_up"]["produced"] == (150, 250)
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Highest produce-side rate" not in text
    assert "do not settle the order between them" in text


def test_a_guaranteed_advantage_says_at_least():
    runs = [
        run("rlvr", hits=400, rows=100_000,
            groups={"prompt": 100, "response": 300, "reference": 200}),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["produced"] == (300, 400) and t["runner_up"]["produced"] == (10, 10)
    # 300/400 against an exact 10: guaranteed at least 30×, not the 30× two
    # floors would read by coincidence — the wording is what changes.
    assert "at least 30.0× dpo's" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_two_settled_intervals_state_the_multiple_plainly():
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100}),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10}),
    ]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "10.0× dpo's" in text and "at least 10.0×" not in text


# --- the largest contributor, when the counts establish one ---------------


def test_a_source_is_only_called_largest_when_its_floor_clears_the_rest():
    # A is exactly 20; B is 15–30. Picking A by the low ends is the best
    # available reading and does not establish that A is the larger.
    sources = [
        {"name": "a", "hits": 20, "produced_hits": 20, "produced_hits_hi": 20},
        {"name": "b", "hits": 30, "produced_hits": 15, "produced_hits_hi": 30},
    ]
    top, biggest = influence._largest(sources, "produced")
    assert top["name"] == "a" and not biggest


def test_a_floor_above_every_other_ceiling_is_the_largest():
    sources = [
        {"name": "a", "hits": 40, "produced_hits": 40, "produced_hits_hi": 40},
        {"name": "b", "hits": 30, "produced_hits": 15, "produced_hits_hi": 30},
    ]
    top, biggest = influence._largest(sources, "produced")
    assert top["name"] == "a" and biggest


def test_the_unsettled_case_is_worded_as_a_floor_not_an_ordering():
    # `a` is exactly 20 produce-side rows; `b` is 15–30. Neither is established
    # as the larger, and both are too thin against their own row counts to be a
    # concentration, so the fallback line has to describe the pick honestly.
    runs = [run("rlvr", hits=50, rows=100_000,
                groups={"prompt": 15, "response": 35, "reference": 35},
                sources={"a": 20, "b": 30},
                totals={"a": 100_000, "b": 100_000},
                by_source_group={"a": {"prompt": 0, "response": 20, "reference": 20},
                                 "b": {"prompt": 15, "response": 15, "reference": 15}})]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"] is None
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "no single source is established as the largest" in text


# --- the nearest competitor is the one that binds --------------------------


def test_the_runner_up_is_the_stage_with_the_highest_ceiling():
    # Exactly 50%, exactly 21%, and 20–40%. The 21% stage sorts second by
    # floor, but the 40% ceiling is what limits the lead the counts guarantee:
    # 50/40 = 1.25×, not 50/21 = 2.4×.
    runs = [
        run("sft", hits=500, rows=1_000, groups={"prompt": 0, "response": 500}),
        run("dpo", hits=210, rows=1_000, groups={"prompt": 0, "response": 210}),
        run("rlvr", hits=400, rows=1_000,
            groups={"prompt": 200, "response": 200, "reference": 200}),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "sft"
    assert [r["stage"] for r in t["ranked"]] == ["sft", "dpo", "rlvr"]
    assert t["runner_up"]["stage"] == "rlvr"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "at least 1.2× rlvr's" in text
    assert "2.4×" not in text


def test_exact_rates_keep_the_second_by_rate_as_the_runner_up():
    runs = [
        run("sft", hits=500, rows=1_000, groups={"prompt": 0, "response": 500}),
        run("dpo", hits=210, rows=1_000, groups={"prompt": 0, "response": 210}),
        run("rlvr", hits=100, rows=1_000, groups={"prompt": 0, "response": 100}),
    ]
    t = influence.compare(runs, THINK)
    assert t["runner_up"]["stage"] == "dpo"


# --- a concentration is only over the columns that were read ---------------


def test_a_concentration_from_a_narrowed_produce_side_is_qualified():
    runs = [run(
        "rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
        fields=["prompt", "response"],
        available_fields=["prompt", "response", "reference"],
        sources={"src": 100}, totals={"src": 1_000},
        by_source_group={"src": {"prompt": 0, "response": 100}},
    )]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"]["name"] == "src"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "over the produce columns this run read" in text
    assert "a column it did not open could hold matches in other sources" in text
    assert "so a column it did not open could move it" in text


def test_a_complete_produce_side_concentration_is_not_qualified():
    runs = [run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
                sources={"src": 100}, totals={"src": 1_000},
                by_source_group={"src": {"prompt": 0, "response": 100}})]
    t = influence.compare(runs, THINK)
    assert t["best"]["concentration"]["name"] == "src"
    assert "could hold matches in other sources" not in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_shared_coverage_gap_is_stated_once_for_the_trace():
    # Every committed run predates `available_fields`, so repeating the same
    # caveat under each stage and again in the verdict says one fact 3+ times.
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
            available_fields=[], sources={"src": 100}, totals={"src": 1_000},
            by_source_group={"src": {"prompt": 0, "response": 100}}),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10},
            available_fields=[]),
    ]
    t = influence.compare(runs, THINK)
    assert t["coverage_unrecorded"]
    text = "\n".join(influence.render(t, "olmo-3-7b-think"))
    assert text.count("None of these runs records which sides its mix holds") == 1
    assert influence.UNRECORDED not in text


def test_a_gap_only_some_runs_have_stays_on_the_stage_that_has_it():
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
            sources={"src": 100}, totals={"src": 1_000},
            by_source_group={"src": {"prompt": 0, "response": 100}}),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10},
            available_fields=[]),
    ]
    t = influence.compare(runs, THINK)
    assert not t["coverage_unrecorded"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "None of these runs records" not in text
    assert f"dpo's rate is a floor rather than a figure — {influence.UNRECORDED}" in text


# --- a legacy run holding every group read every side ----------------------


def test_a_legacy_run_covering_all_three_groups_needs_no_available_fields():
    # `available_fields` is drawn from GROUPS, so a run whose `fields` holds all
    # of them cannot have been narrowed, whatever its vintage.
    r = run("rlvr", hits=0, rows=100, groups={"prompt": 0, "response": 0, "reference": 0},
            available_fields=[])
    assert influence.coverage_gaps(r) == []
    t = influence.compare([r], THINK)
    assert [x["stage"] for x in t["zero"]] == ["rlvr"]
    assert "exact over every one of them" in " ".join(influence.render(t, "olmo-3-7b-think"))


def test_a_legacy_run_holding_both_produce_groups_is_not_understated():
    runs = [
        run("rlvr", hits=100, rows=100_000,
            groups={"prompt": 0, "response": 60, "reference": 60}, available_fields=[]),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10},
            available_fields=[]),
    ]
    t = influence.compare(runs, THINK)
    assert influence._understated(t["best"], "produced") is None
    # dpo read only `response` and cannot show the mix had no `reference`.
    assert [r["stage"] for r in t["understated"]] == ["dpo"]
    assert not t["coverage_unrecorded"]
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "None of these runs records" not in text
    assert "dpo's rate is a floor" in text


def test_an_unrecognised_column_still_counts_against_a_full_sweep():
    r = run("rlvr", hits=0, rows=100, groups={"prompt": 0, "response": 0, "reference": 0},
            available_fields=[], unsearched_columns=["rationale"])
    assert influence.coverage_gaps(r) == ["rationale went unsearched"]
    assert influence._understated(influence.stage_trace(r), "produced") == \
        "rationale went unsearched in it"


# --- the row basis qualifies a narrowed concentration too ------------------


def test_a_row_basis_concentration_over_narrowed_fields_is_qualified():
    runs = [run(
        "dpo", hits=100, rows=100_000, groups={"prompt": 100}, fields=["prompt"],
        available_fields=["prompt", "response"],
        sources={"src": 100}, totals={"src": 1_000},
    )]
    t = influence.compare(runs, THINK)
    assert t["basis"] == "rows" and t["best"]["concentration"]["name"] == "src"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "over the columns this run read" in text
    assert "That attribution is over the columns the run read" in text


def test_a_complete_row_basis_concentration_is_not_qualified():
    runs = [run("dpo", hits=100, rows=100_000,
                groups={"prompt": 100, "response": 0, "reference": 0},
                sources={"src": 100}, totals={"src": 1_000})]
    t = influence.compare(runs, THINK)
    assert "could hold matches in other sources" not in \
        " ".join(influence.render(t, "olmo-3-7b-think"))


# --- a guaranteed multiple needs a bounded competitor ----------------------


def test_no_multiplier_against_a_competitor_with_unread_columns():
    # A complete 10% response rate against an observed 1% whose `reference`
    # column was never opened: `_ceiling` bounds only what was read, so the
    # quotient is not a guarantee — and the floor caveat two clauses later
    # already says the ordering could reverse.
    runs = [
        run("rlvr", hits=100, rows=1_000, groups={"prompt": 0, "response": 100},
            fields=["prompt", "response"], available_fields=["prompt", "response"]),
        run("dpo", hits=10, rows=1_000, groups={"prompt": 0, "response": 10},
            fields=["prompt", "response"],
            available_fields=["prompt", "response", "reference"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "rlvr"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Highest produce-side rate" not in text
    assert "dpo's rate is a floor rather than a figure" in text


def test_a_bounded_competitor_still_gets_a_multiplier():
    runs = [
        run("rlvr", hits=100, rows=1_000, groups={"prompt": 0, "response": 100}),
        run("dpo", hits=10, rows=1_000, groups={"prompt": 0, "response": 10}),
    ]
    assert "10.0× dpo's" in " ".join(
        influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))


# --- a colliding slug is a shared write path ------------------------------


def test_a_colliding_slug_is_replaced_not_merely_dropped():
    # Dropping it would leave `grep` to derive one from the pattern, which two
    # searches differing only in `--regex` share.
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5}, slug="identity")]
    t = influence.compare(runs, THINK)
    t["slug_collides"], t["slug_suggest"] = True, "identity-1"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "--slug identity-1" in text
    assert "--slug identity " not in text
    assert "re-running under it would overwrite the other one's results" in text


def test_a_unique_slug_is_still_carried():
    runs = [run("rlvr", hits=5, rows=100, groups={"prompt": 5}, slug="identity")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "--slug identity" in text
    assert "overwrite the other one's results" not in text


def test_any_understated_competitor_suppresses_the_multiple():
    # Three stages: the leader is exact, the runner-up by ceiling is exact, and
    # a third has unread columns whose matches could overturn the order. The
    # guarantee has to range over every competitor, not the one being quoted.
    runs = [
        run("sft", hits=500, rows=1_000, groups={"prompt": 0, "response": 500}),
        run("dpo", hits=20, rows=1_000, groups={"prompt": 0, "response": 20}),
        run("rlvr", hits=10, rows=1_000, groups={"prompt": 0, "response": 10},
            fields=["prompt", "response"],
            available_fields=["prompt", "response", "reference"]),
    ]
    t = influence.compare(runs, THINK)
    assert t["best"]["stage"] == "sft" and t["runner_up"]["stage"] == "dpo"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "Highest produce-side rate" not in text
    assert "rlvr's rate is a floor rather than a figure" in text


def test_all_bounded_competitors_keep_the_multiple():
    runs = [
        run("sft", hits=500, rows=1_000, groups={"prompt": 0, "response": 500}),
        run("dpo", hits=20, rows=1_000, groups={"prompt": 0, "response": 20}),
        run("rlvr", hits=10, rows=1_000, groups={"prompt": 0, "response": 10}),
    ]
    assert "25.0× dpo's" in " ".join(
        influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))


def test_a_legacy_winner_keeps_its_caveat_in_a_mixed_vintage_trace():
    # The global note only prints when every run is limited that way. Here one
    # records its coverage, so the winner's own caveat must stay inline.
    runs = [
        run("rlvr", hits=100, rows=100_000, groups={"prompt": 0, "response": 100},
            available_fields=[], sources={"src": 100}, totals={"src": 1_000},
            by_source_group={"src": {"prompt": 0, "response": 100}}),
        run("dpo", hits=10, rows=100_000, groups={"prompt": 0, "response": 10},
            fields=["prompt", "response", "reference"],
            available_fields=["prompt", "response", "reference"]),
    ]
    t = influence.compare(runs, THINK)
    assert not t["coverage_unrecorded"] and t["best"]["stage"] == "rlvr"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "None of these runs records" not in text
    assert influence.UNRECORDED in text
    assert "so a column it did not open could move it" in text


def test_collision_recovery_names_the_renames_that_join_the_trace():
    # A free slug alone opens a third group: the existing files have to move
    # with it or the trace stays split.
    runs = [
        run("dpo", hits=5, rows=100, groups={"prompt": 5, "response": 0}, slug="identity"),
        run("rlvr", hits=0, rows=100, groups={"prompt": 0, "response": 0}, slug="identity"),
    ]
    t = influence.compare(runs, THINK)
    t["slug_collides"], t["slug_suggest"] = True, "identity-1"
    text = " ".join(influence.render(t, "olmo-3-7b-think"))
    assert "--slug identity-1 --stage sft" in text
    assert "Move this trace under `identity-1` first" in text
    # Both stages that already have a file, not just the matching one.
    assert "olmo-3-7b-think.dpo.grep-identity.json` → " \
           "`results/olmo-3-7b-think.dpo.grep-identity-1.json" in text
    assert "olmo-3-7b-think.rlvr.grep-identity.json` → " \
           "`results/olmo-3-7b-think.rlvr.grep-identity-1.json" in text
    assert "leaves this one just as incomplete" in text


def test_no_rename_instructions_without_a_collision():
    runs = [run("dpo", hits=5, rows=100, groups={"prompt": 5}, slug="identity")]
    text = " ".join(influence.render(influence.compare(runs, THINK), "olmo-3-7b-think"))
    assert "Move this trace under" not in text
    assert "--slug identity --stage sft" in text
