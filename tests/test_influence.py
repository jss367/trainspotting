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
    r = run("rlvr", hits=10, rows=100, groups={"prompt": 0, "response": 8, "reference": 7})
    assert influence.produced(r) == (8, 10)


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
        run("rlvr", hits=200, rows=100_000, groups={"prompt": 0, "response": 60, "reference": 60}),
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
        run("rlvr", hits=200, rows=100_000, groups={"prompt": 0, "response": 50, "reference": 50}),
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
