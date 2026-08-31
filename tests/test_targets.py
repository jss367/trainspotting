"""Models and standalone datasets resolving to the one shape the commands read.

The point of `registry.resolve` is that nothing below `facts` has to know which
kind it got, so these tests are mostly about the two shapes staying
interchangeable — and about the places where they genuinely differ (a dataset
has no pretraining corpora, and no verifier to read a label off) failing loudly
rather than producing an empty or mislabeled record.
"""

import pytest
from conftest import row_fixture

from trainspotting import classify, context, extract, registry


def test_a_model_resolves_to_its_own_stages():
    m = registry.resolve("olmo-3-7b-instruct")
    assert m["is_model"]
    assert m["hf_model"] == "allenai/Olmo-3-7B-Instruct"
    assert [s["stage"] for s in registry.post_training_stages(m)] == ["sft", "dpo", "rlvr"]
    assert registry.pretrain_stages(m)


def test_a_dataset_resolves_to_exactly_one_post_training_stage():
    d = registry.resolve("wildchat-1m")
    assert not d["is_model"]
    assert d["hf_model"] is None
    stages = registry.post_training_stages(d)
    assert len(stages) == 1
    assert stages[0]["hf_dataset"] == "allenai/WildChat-1M"
    # Nothing to sample by shard: a dataset has no corpus half.
    assert registry.pretrain_stages(d) == []


def test_a_dataset_stage_carries_everything_a_model_stage_does():
    """The interchangeability the whole design rests on. A key missing here is a
    KeyError deep inside a sampling run, after the fetches have been paid for."""
    model_stage = registry.post_training_stages(registry.resolve("olmo-3-7b-instruct"))[0]
    dataset_stage = registry.post_training_stages(registry.resolve("wildchat-1m"))[0]
    assert set(model_stage) <= set(dataset_stage)


def test_a_datasets_stage_token_is_its_kind():
    """It names the result files (results/wildchat-1m.chat.labels.json), so it
    has to be the stable, registry-declared kind rather than anything derived
    from the dataset id."""
    stage = registry.post_training_stages(registry.resolve("wildchat-1m"))[0]
    assert stage["stage"] == registry.stage_kind(stage) == "chat"


def test_a_model_stage_takes_its_kind_from_its_pipeline_position():
    for stage in registry.post_training_stages(registry.resolve("olmo-3-7b-think")):
        assert registry.stage_kind(stage) == stage["stage"]


def test_every_target_name_resolves():
    assert registry.targets()
    for name in registry.targets():
        assert registry.resolve(name)["stages"]


def test_targets_lists_models_before_datasets():
    """The site builds its tabs in this order, so the pipelines come first."""
    kinds = [registry.resolve(n)["is_model"] for n in registry.targets()]
    assert kinds == sorted(kinds, reverse=True)


def test_an_unknown_name_names_both_kinds():
    with pytest.raises(KeyError, match="Datasets:"):
        registry.resolve("gpt-5")


def test_case_is_ignored_for_datasets_too():
    assert registry.resolve("WildChat-1M")["target"] == "wildchat-1m"


def test_only_an_rl_row_gets_a_label_from_its_verifier():
    """A chat row has no verifier. Passing the kind rather than a stage name is
    what keeps `chat` out of the RL branch — it used to be everything that was
    not sft or dpo."""
    row = {"dataset_source": "hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter"}
    assert classify.verifier_label(row, "rlvr") == "instruction_following"
    assert classify.verifier_label(row, "chat") is None


def test_a_chat_row_becomes_a_conversation_record_not_an_rl_one():
    saved = row_fixture("wildchat-1m", "chat")
    prompt = extract.extract_prompt(saved["row"], "conversation")
    rec = context.build(saved["row"], "chat", prompt, row_index=0)
    assert rec["kind"] == "chat"
    assert rec["turns"] and rec["turns"][0]["role"] == "user"
    # The RL branch is what an unknown kind falls through to, and it would
    # store a verifier record for a row that has no verifier.
    assert "reward" not in rec
    assert rec["meta"]["model"]
