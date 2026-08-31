"""Models and standalone datasets resolving to the one shape the commands read.

The point of `registry.resolve` is that nothing below `facts` has to know which
kind it got, so these tests are mostly about the two shapes staying
interchangeable — and about the places where they genuinely differ (a dataset
has no pretraining corpora, and no verifier to read a label off) failing loudly
rather than producing an empty or mislabeled record.
"""

import sys

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


def test_a_chat_kind_gets_its_own_rubric_in_both_label_modes():
    """The default rubric reads a prompt for what training on it would teach,
    and sends a jailbreak attempt to `harmlessness` on that basis. Nothing was
    trained on a chat log, so it needs a rubric that describes the request."""
    assert classify.system_for("chat") is classify.CHAT_SYSTEM
    assert classify.system_for("chat", "is this about X?") is classify.ASK_CHAT_SYSTEM
    for kind in ("sft", "dpo", "rlvr"):
        assert classify.system_for(kind) is None
        assert classify.system_for(kind, "is this about X?") is None


def test_the_chat_rubric_offers_exactly_the_taxonomy():
    """A label the parser rejects would silently drop the prompt, and a missing
    one would never be assigned — either way the card under-counts."""
    for label in classify.LABELS:
        assert f"- {label}:" in classify.CHAT_SYSTEM


def test_the_chat_rubric_does_not_claim_a_training_signal():
    """The one thing it exists to avoid saying."""
    assert "training signal" not in classify.CHAT_SYSTEM.split("Nothing was trained")[0]
    assert "Nothing was trained on these conversations" in classify.CHAT_SYSTEM


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


def test_the_cli_canonicalizes_a_target_name_before_using_it(monkeypatch, capsys):
    """Result filenames are built from the target name, and the site indexes the
    registry key — so `classify WildChat-1M` writing WildChat-1M.chat.labels.json
    produced a run the page never asked for."""
    from trainspotting import cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_facts", lambda args: seen.setdefault("target", args.target))
    monkeypatch.setattr(sys, "argv", ["trainspotting", "facts", "WildChat-1M"])
    cli.main()
    assert seen["target"] == "wildchat-1m"


def test_an_unknown_target_exits_with_the_registry_message(monkeypatch):
    """A KeyError traceback out of argparse is a crash report, not a usage error."""
    from trainspotting import cli

    monkeypatch.setattr(sys, "argv", ["trainspotting", "facts", "gpt-5"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "Unknown model or dataset" in str(exc.value)


def test_a_sampled_row_travels_with_its_index():
    """The index is what a result record stores to address its training example.
    Joining on the prompt cannot separate two rows that open with the same 400
    characters — rare in a curated mix, routine in a chat log."""
    from trainspotting import cli, hf

    stage = registry.post_training_stages(registry.resolve("wildchat-1m"))[0]
    rows = [
        (7, {"conversation": [{"role": "user", "content": "shared opening"}]}),
        (9, {"conversation": [{"role": "user", "content": "shared opening"}]}),
        (11, {"conversation": [{"role": "assistant", "content": "no user turn"}]}),
    ]
    original = hf.sample_rows_with_index
    hf.sample_rows_with_index = lambda *a, **k: rows
    try:
        got = cli._sample_rows(stage, 3, 0)
        prompts = cli._sample_prompts(stage, 3, 0)
    finally:
        hf.sample_rows_with_index = original

    # The prompt-less row drops; the two identical prompts stay distinguishable.
    assert [i for i, _, _ in got] == [7, 9]
    assert prompts == [(7, "shared opening"), (9, "shared opening")]
