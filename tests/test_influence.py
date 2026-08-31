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
    """One grep result file, with only the keys the influence layer reads."""
    return {
        "stage": stage,
        "dataset": f"allenai/{stage}",
        "pattern": kw.pop("pattern", "as an AI language model"),
        "regex": kw.pop("regex", False),
        "case_sensitive": kw.pop("case_sensitive", False),
        "fields": fields if fields is not None else list(groups),
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
    assert "largest contributor `mystery` holds 4 rows" in \
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
    old = run("dpo", hits=5, rows=100, groups={"prompt": 5}, fields=["prompt"])
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
