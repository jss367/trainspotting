"""The ways a preference-pair audit can be wrong without raising.

Every failure this pins is a number that still looks like a number:

  * counting a multi-turn pair's shared history on both sides, which adds the
    same characters to each and drags the *difference* — the one quantity here —
    toward zero
  * scoring a tie as a loss for the length rule, which reports a mix of
    equal-length pairs as one where length predicts nothing
  * splitting a think stage's gap into reasoning and answer that do not add back
    up to it, so the decomposition contradicts the total printed above it
  * reading "nobody checked for upstream truncation" as "nothing was truncated"
  * a committed `.pairs.json` drifting from the context sample the site serves
    beside it
"""

import argparse
import json
import math
from pathlib import Path

import pytest

from trainspotting import pairs, paths
from trainspotting.commands.pairs import cmd_pairs

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
CONTEXTS = sorted(DATA.glob("*.dpo.context.json"))
CONTEXT_IDS = [p.name.replace(".dpo.context.json", "") for p in CONTEXTS]


def turn(role, chars, text="x", reasoning=None, chars_raw=None):
    t = {"role": role, "text": text, "chars": chars}
    if reasoning is not None:
        t["reasoning"] = {"text": "r", "chars": reasoning}
    if chars_raw is not None:
        t["chars_raw"] = chars_raw
    return t


def pair(chosen, rejected, row=None, meta=None, chosen_model=None, rejected_model=None, **rest):
    return {
        "kind": "dpo",
        "row": row,
        "chosen": {"model": chosen_model, "turns": chosen},
        "rejected": {"model": rejected_model, "turns": rejected},
        "meta": meta or {},
        **rest,
    }


def ctx(records, **rest):
    return {"dataset": "test/mix", "stage": "dpo", "sample": len(records), "seed": 0,
            "records": records, **rest}


# ------------------------------------------------------------------ lengths ---

def test_shared_history_counts_on_neither_side():
    """A multi-turn pair's opening is the conversation both answers reply in.

    Counted on both sides it cancels out of the difference but not out of the
    two totals, so `chars` would report 1,100-character sides differing by 210
    where the completions are 300 and 90.
    """
    shared = [turn("user", 500, "ask"), turn("assistant", 400, "first reply"), turn("user", 200, "again")]
    out = pairs.stage_pairs(ctx([
        pair(shared + [turn("assistant", 300, "good")], shared + [turn("assistant", 90, "bad")]),
    ]))
    assert out["delta"]["mean"] == 210
    assert out["chars"]["chosen"]["mean"] == 300
    assert out["chars"]["rejected"]["mean"] == 90


def test_a_tie_is_not_an_answer_the_length_rule_got_wrong():
    """Equal-length sides leave the rule undefined, not defeated.

    Both denominators are reported: the tie stays in `longer_chosen`, which is a
    share of every pair, and drops out of `length_rule`, which is the rule's
    accuracy where it can answer at all.
    """
    same = [turn("assistant", 100, "same")]
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 200, "long")], [turn("assistant", 100, "short")]),
        pair(same, [turn("assistant", 100, "also")]),
    ]))
    assert out["ties"] == 1
    assert (out["longer_chosen"]["k"], out["longer_chosen"]["n"]) == (1, 2)
    assert (out["length_rule"]["k"], out["length_rule"]["n"]) == (1, 1)


def test_every_pair_tying_leaves_the_rule_undefined_rather_than_zero():
    """`_rate([])` has a denominator and no rate, and every reader of the file
    has to see that shape rather than index into it: `report` raised `KeyError`
    on `rule['rate']` and took the whole report down, and the site's card drew a
    `NaN%` bar. A 0% would be worse than either — it says length never picks the
    chosen side, where the truth is that length picks neither."""
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 100, "a")], [turn("assistant", 100, "b")], meta={"src": "a"}),
        pair([turn("assistant", 200, "c")], [turn("assistant", 200, "d")], meta={"src": "b"}),
    ]))
    assert out["ties"] == out["n"] == 2
    assert out["length_rule"] == {"k": 0, "n": 0}
    assert "rate" not in out["length_rule"]
    # The share of *every* pair still answers: the rule is undefined, not the
    # sample.
    assert (out["longer_chosen"]["k"], out["longer_chosen"]["n"]) == (0, 2)
    # And the breakdown is empty rather than two groups reported at 0%, because
    # it now runs over the same decided pairs the rule does — of which there are
    # none.
    assert out["by"] == {}


def test_the_breakdown_drops_the_same_ties_the_headline_rule_does():
    """The breakdown is the headline rate per group, so it has to be over the
    same rows. Handed every pair, a tie arrives as "the longer side was not
    chosen" and is scored as the group's failure — which drags exactly the
    groups that contain ties below a stage rate that never counted them."""
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"src": "a"}),
        pair([turn("assistant", 5)], [turn("assistant", 5)], meta={"src": "a"}),
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"src": "b"}),
    ]))
    by = out["by"]["src"]
    # The tie leaves `a`'s denominator rather than scoring 1/2 = 50%.
    assert (by["a"]["k"], by["a"]["n"]) == (1, 1)
    assert (by["b"]["k"], by["b"]["n"]) == (1, 1)
    # Which is the headline denominator split across the groups, exactly.
    assert sum(st["n"] for st in by.values()) == out["length_rule"]["n"] == out["n"] - out["ties"]


def test_identical_sides_are_named_as_training_nothing():
    """DPO reads the difference of the two sides' log probabilities, so a pair
    whose sides are identical cancels exactly. It is still a row of the mix, so
    it is still counted — but a tie between two empty sides is not the same
    observation as a tie between two answers."""
    same = [turn("user", 50, "ask"), {**turn("assistant", 100, "identical"), "raw": True}]
    out = pairs.stage_pairs(ctx([pair(list(same), list(same))]))
    assert out["degenerate"] == 1
    assert out["ties"] == 1
    assert out["delta"]["mean"] == 0


def test_one_empty_side_is_a_length_difference_and_not_a_cancelling_pair():
    """The shared prefix can swallow one side without swallowing the other, and
    that pair is nothing like an identical one: it has a real winner on length
    and its loss does not cancel. Counting it as `degenerate` would attach "the
    two completions are identical, so they train nothing" — what the report and
    the card say about that number — to a row where neither clause holds."""
    shared = [turn("user", 50, "ask")]
    out = pairs.stage_pairs(ctx([
        pair(list(shared), shared + [turn("assistant", 300, "only the rejected side answers")]),
    ]))
    assert out["degenerate"] == 0
    assert out["ties"] == 0
    assert out["delta"]["mean"] == -300
    assert (out["length_rule"]["k"], out["length_rule"]["n"]) == (0, 1)


def test_both_directions_survive_the_histograms():
    """`derive.histogram` bins by log10 and drops anything at or below zero, so
    a signed quantity binned through it loses every pair the rejected side won —
    silently, as a chart that is simply shorter on one end."""
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 1000)], [turn("assistant", 100)]),
        pair([turn("assistant", 100)], [turn("assistant", 1000)]),
        pair([turn("assistant", 100)], [turn("assistant", 100)]),
    ]))
    delta = out["delta"]
    assert sum(delta["hist_chosen"]) == 1
    assert sum(delta["hist_rejected"]) == 1
    # Every pair is in exactly one place: a direction, or the tie count.
    assert sum(delta["hist_chosen"]) + sum(delta["hist_rejected"]) + out["ties"] == out["n"]
    assert "hist" not in delta


def test_reasoning_and_answer_add_back_up_to_the_gap():
    """A think turn stores its thinking beside its answer and its true length in
    `chars_raw`, so answer length is the difference rather than `chars`. Getting
    that wrong loses the `<think>` markers from one half only, and the split
    stops summing to the total printed above it."""
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 300, reasoning=2000, chars_raw=2315)],
             [turn("assistant", 200, reasoning=500, chars_raw=715)]),
    ]))
    split = out["split"]
    assert split["reasoning"]["mean"] == 1500
    assert split["answer"]["mean"] == out["delta"]["mean"] - 1500
    assert split["reasoning"]["mean"] + split["answer"]["mean"] == out["delta"]["mean"]


def test_a_stage_with_no_thinking_gets_no_split():
    out = pairs.stage_pairs(ctx([pair([turn("assistant", 300)], [turn("assistant", 200)])]))
    assert "split" not in out


# ------------------------------------------------------------- the generators ---

def test_one_directional_matchup_fits_every_choice():
    """The think mixes pair one strong generator against one weak one throughout.
    The rule is then exactly right, which is a fact about the mix — so it is
    reported with `fitted`, because read as a result it says the opposite."""
    rule = pairs._model_rule([("big", "small")] * 10)
    assert rule["rate"] == 1.0
    assert rule["matchups"] == 1
    assert rule["fitted"] is True


def test_the_rule_takes_the_majority_direction_of_each_matchup():
    rule = pairs._model_rule([("a", "b")] * 7 + [("b", "a")] * 3)
    assert (rule["k"], rule["n"], rule["rate"]) == (7, 10, 0.7)


def test_pairs_the_rule_cannot_answer_leave_its_denominator():
    """A pair of one model against itself is unanswerable by construction, and
    scoring it as a coin flip would drag the rate toward 0.5 in proportion to
    how often a mix samples a model against itself. An unrecorded generator is
    the same situation arriving by a different route."""
    rule = pairs._model_rule([("a", "b"), ("a", "a"), (None, "b"), ("c", None)])
    assert (rule["k"], rule["n"]) == (1, 1)
    assert rule["same_model"] == 1
    assert rule["unknown"] == 2


def test_no_generators_at_all_is_no_rule_rather_than_a_perfect_one():
    assert pairs._model_rule([(None, None)] * 5) is None


def test_matchups_keep_their_direction():
    """Which of two models is the preferred one is the whole content of the
    table; an unordered count would render "32b over 0.6b, every pair" as "these
    two appear together"."""
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 10)], [turn("assistant", 5)], chosen_model="big", rejected_model="small"),
        pair([turn("assistant", 10)], [turn("assistant", 5)], chosen_model="big", rejected_model="small"),
        pair([turn("assistant", 10)], [turn("assistant", 5)], chosen_model="small", rejected_model="big"),
    ]))
    assert out["models"]["matchups"] == [
        {"chosen": "big", "rejected": "small", "n": 2},
        {"chosen": "small", "rejected": "big", "n": 1},
    ]


# ------------------------------------------------------------------ breakdown ---

def test_the_breakdown_splits_on_the_stages_own_provenance_column():
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"preference_type": "judged"}),
        pair([turn("assistant", 5)], [turn("assistant", 10)], meta={"preference_type": "judged"}),
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"preference_type": "delta"}),
    ]))
    by = out["by"]["preference_type"]
    assert (by["judged"]["k"], by["judged"]["n"]) == (1, 2)
    assert (by["delta"]["k"], by["delta"]["n"]) == (1, 1)


def test_a_single_valued_column_splits_nothing_and_is_left_out():
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"preference_type": "one"}),
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"preference_type": "one"}),
    ]))
    assert "preference_type" not in out["by"]


def test_a_column_with_more_values_than_the_cap_is_left_out():
    """Past the cap the breakdown is the sample itself at one row per cell —
    every rate 0% or 100% on a single pair."""
    n = pairs.MAX_BREAKDOWN_VALUES + 1
    out = pairs.stage_pairs(ctx([
        pair([turn("assistant", 10)], [turn("assistant", 5)], meta={"id": str(i)}) for i in range(n)
    ]))
    assert "id" not in out["by"]


# --------------------------------------------------------------- truncation ---

def test_unchecked_truncation_is_unknown_rather_than_none():
    """A shortened cell arrives with fewer characters than it has, and cuts land
    on the longest cells — the side this layer measures. A run that was not
    looking must not report a clean bill of health."""
    out = pairs.stage_pairs(ctx([pair([turn("assistant", 10)], [turn("assistant", 5)])]))
    assert out["truncated_rows"] is None


def test_a_run_that_checked_reports_what_it_found():
    out = pairs.stage_pairs(ctx(
        [
            pair([turn("assistant", 10)], [turn("assistant", 5)], truncated=["chosen"]),
            pair([turn("assistant", 10)], [turn("assistant", 5)]),
        ],
        truncation_recorded=True,
    ))
    assert out["truncated_rows"] == 1


def test_a_cut_in_a_column_this_layer_never_reads_is_not_a_shortened_length():
    """`context` stores the cut cells this *stage* reads, which for DPO includes
    the standalone `prompt` column. No length here comes from it — both sides are
    built out of `chosen` and `rejected` — so a row cut only there arrived whole
    for every number in this file, and counting it would put "these lengths are
    lower bounds" on a sample where both measured sides are complete."""
    out = pairs.stage_pairs(ctx(
        [
            pair([turn("assistant", 10)], [turn("assistant", 5)], truncated=["prompt"]),
            pair([turn("assistant", 10)], [turn("assistant", 5)], truncated=["prompt", "rejected"]),
            pair([turn("assistant", 10)], [turn("assistant", 5)]),
        ],
        truncation_recorded=True,
    ))
    assert out["truncated_rows"] == 1


# ------------------------------------------------------------ wrong stage kind ---

def test_a_stage_with_no_pairs_raises_rather_than_reporting_zeros():
    """A zero here reads as "no length asymmetry". For an SFT stage the honest
    answer is that the question does not apply, and only an exception says so."""
    with pytest.raises(ValueError):
        pairs.stage_pairs(ctx([{"kind": "sft", "turns": [turn("assistant", 10)]}]))


# ----------------------------------------------------------- the exit status ---
# `scripts/refresh_samples.sh` runs this straight after `context` and reads only
# the status. It used to blanket it with `|| true`, because a target with no
# preference stage exited non-zero — and that also swallowed a preference stage
# whose freshly written context run could not be measured, which is a refresh
# that failed while reporting success.

def test_a_target_with_no_preference_stage_is_not_a_failure():
    """The question does not apply, which is not the same as asking it and
    failing. A non-zero here is what forced the caller to ignore the status."""
    cmd_pairs(argparse.Namespace(target="olmo-3-7b-rl-zero-code", stage=None))


def test_a_preference_stage_that_cannot_be_measured_is_a_failure(monkeypatch):
    """The other half of the same contract: a DPO stage whose context run is
    missing has to reach the caller as a failure, or a refresh writes nothing and
    says it succeeded."""
    monkeypatch.setattr(paths, "find", lambda name: None)
    with pytest.raises(SystemExit) as exit_info:
        cmd_pairs(argparse.Namespace(target="olmo-3-7b-instruct", stage=None))
    assert exit_info.value.code != 0


# -------------------------------------------------- against the committed data ---

@pytest.mark.parametrize("path", CONTEXTS, ids=CONTEXT_IDS)
def test_every_committed_preference_sample_measures(path):
    out = pairs.stage_pairs(json.loads(path.read_text()))
    assert out["n"] > 0
    # Each pair falls in exactly one of the three buckets, and the rule's
    # denominator is the two that are not ties.
    assert out["longer_chosen"]["n"] == out["n"]
    assert out["length_rule"]["n"] == out["n"] - out["ties"]
    assert out["length_rule"]["k"] == out["longer_chosen"]["k"]
    assert 0 <= out["length_rule"]["lo"] <= out["length_rule"]["rate"] <= out["length_rule"]["hi"] <= 1
    if out.get("split"):
        assert math.isclose(
            out["split"]["reasoning"]["mean"] + out["split"]["answer"]["mean"],
            out["delta"]["mean"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )


@pytest.mark.parametrize("path", CONTEXTS, ids=CONTEXT_IDS)
def test_the_committed_pairs_file_describes_the_sample_beside_it(path):
    """The site serves both, and the derived one is what its card draws. A
    `.pairs.json` left over from an earlier sample would show the page one
    stage's numbers over another stage's rows, with nothing on either file
    saying so."""
    committed = DATA / path.name.replace(".context.json", ".pairs.json")
    assert committed.exists(), f"{committed.name} missing — re-run scripts/export_site_data.py"
    saved = json.loads(committed.read_text())
    fresh = pairs.stage_pairs(json.loads(path.read_text()))
    for key in ("dataset", "revision", "n", "ties", "degenerate", "truncated_rows"):
        assert saved[key] == fresh[key]
    assert saved["length_rule"]["k"] == fresh["length_rule"]["k"]
    assert math.isclose(saved["delta"]["mean"], fresh["delta"]["mean"], rel_tol=1e-9)
