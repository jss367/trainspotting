"""Rendering a whole training example for a judge that has to read its direction.

Everything here protects one property: the markup that says which text the model
was trained toward has to survive the character budget. A preference pair whose
[DISPREFERRED] marker was cut away still looks like a well-formed example, and
the judge reads a rejected completion as the preferred one — a silently
sign-flipped answer, which is worse than a missing one.
"""

import json

import pytest

from trainspotting import paths, stance


def turn(role, text, reasoning=None):
    t = {"role": role, "text": text, "chars": len(text)}
    if reasoning is not None:
        t["reasoning"] = {"text": reasoning, "chars": len(reasoning)}
    return t


def dpo(chosen, rejected, prompt="what should I do?"):
    return {
        "kind": "dpo",
        "chosen": {"turns": [turn("user", prompt), turn("assistant", chosen)]},
        "rejected": {"turns": [turn("user", prompt), turn("assistant", rejected)]},
    }


def test_both_sides_of_a_pair_stay_marked_however_long_they_are():
    rendered = stance.render(dpo("y" * 200_000, "n" * 200_000))
    assert "PREFERRED — training pushes toward this" in rendered
    assert "DISPREFERRED — training pushes away from this" in rendered
    assert len(rendered) <= stance.MAX_EXAMPLE
    # And both completions are actually present, not just their headings.
    body = rendered.split("[PREFERRED", 1)[1]
    assert "y" in body.split("[DISPREFERRED", 1)[0]
    assert "n" in body.split("[DISPREFERRED", 1)[1]


def test_the_shared_history_of_a_pair_is_rendered_once():
    # `context` stores each side as the whole conversation. Rendering both
    # copies spends the budget saying the same thing twice and buries the one
    # difference the pair is about.
    rendered = stance.render(dpo("yes", "no", prompt="UNIQUEPREFIX"))
    assert rendered.count("UNIQUEPREFIX") == 1


def test_a_reasoning_span_is_part_of_the_target():
    # The context *view* folds thinking away because it buries the answer. The
    # model was fit to it, so a value expressed only while thinking was trained
    # in and has to reach the judge.
    rec = {"kind": "sft", "turns": [turn("assistant", "answer", reasoning="deliberation")]}
    assert "deliberation" in stance.render(rec)


def test_a_prompt_is_marked_as_read_not_as_produced():
    rec = {"kind": "sft", "turns": [turn("user", "ask"), turn("assistant", "reply")]}
    rendered = stance.render(rec)
    assert "[PROMPT: user]" in rendered
    assert "[TARGET: assistant]" in rendered


def test_the_verifier_and_its_pass_rate_reach_the_judge():
    # An RL example teaches whatever the reward pays for, and how often it paid
    # out is part of that — this is the anti-vaccine constraint-checker case.
    rec = {
        "kind": "rlvr",
        "prompt_full": {"text": "write a speech", "chars": 14},
        "reward": {"kind": "constraint checker", "explain": "a program checks constraints",
                   "constraint": {"text": "exactly 2 paragraphs", "chars": 20}},
        "rollouts": {"total": 8, "correct": 4, "passrate": 0.54, "sample": {"text": "gen", "chars": 3}},
    }
    rendered = stance.render(rec)
    assert "constraint checker" in rendered
    assert "exactly 2 paragraphs" in rendered
    assert "54%" in rendered


def test_a_chat_log_has_no_direction_to_render():
    # Nothing was fit to it, so there is no training signal to point either way.
    with pytest.raises(ValueError):
        stance.render({"kind": "chat", "turns": [turn("user", "hi")]})


def test_unused_budget_goes_to_the_part_that_needs_it():
    # One long completion beside one short one: the short one should not hold a
    # reservation it cannot use while the long one is cut to half the example.
    assert stance._allocate([10, 1000], 600) == [10, 590]
    assert stance._allocate([1000, 1000], 600) == [300, 300]
    assert stance._allocate([5, 5], 600) == [5, 5]


def test_a_budget_too_small_to_excerpt_truncates_instead_of_breaking():
    # `excerpt` splits its budget three ways around two elision markers, which
    # goes negative on a small one.
    assert stance._cut("abcdefghij", 4) == "abcd"
    assert stance._cut("abc", 100) == "abc"


@pytest.mark.parametrize("stage", ["sft", "dpo", "rlvr"])
def test_every_committed_example_renders_with_its_markup_intact(stage):
    """The same check against the real committed samples, not constructed rows.

    This is the regression that caught the original single-excerpt render: it
    passed on every short example and lost a side marker on 16 of 300 real
    Dolci-Think-DPO pairs, which are the long ones.
    """
    path = paths.find(f"olmo-3-7b-think.{stage}.context.json")
    if path is None:
        pytest.skip("no committed context sample in this checkout")
    required = {
        "sft": ["[TARGET"],
        "dpo": ["[PREFERRED", "[DISPREFERRED"],
        "rlvr": ["[REWARD"],
    }[stage]
    for rec in json.loads(path.read_text())["records"]:
        rendered = stance.render(rec)
        assert len(rendered) <= stance.MAX_EXAMPLE
        for marker in required:
            assert marker in rendered, f"row {rec.get('row')} lost {marker}"
