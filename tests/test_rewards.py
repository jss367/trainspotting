import pytest

from trainspotting import context


@pytest.mark.parametrize("tag", ["general-quality", "general-quality_ref"])
def test_explicit_judge_tags_identify_ai_feedback_and_survive_context(tag):
    row = {"dataset": [tag], "ground_truth": ["reference"]}
    rec = context.build(row, "rlvr", "prompt", 0)
    assert rec["reward"]["kind"] == "LLM judge"
    assert rec["reward"]["family"] == "rlaif"
    assert rec["meta"]["dataset"] == [tag]


def test_unidentified_reward_is_not_assumed_programmatic():
    rec = context.build({}, "rlvr", "prompt", 0)
    assert rec["reward"]["family"] == "unknown"
