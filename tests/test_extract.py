"""Pulling a prompt out of a row, for every schema the registry claims.

The saved rows under fixtures/rows/ are one real row per (target, stage) —
models and standalone datasets alike — so a `prompt_path` that stops addressing
anything fails here instead of quietly producing an empty sample. `--live`
re-runs the same checks against a freshly fetched row, which is what catches an
upstream schema change.
"""

import pytest
from conftest import row_fixture

from trainspotting import extract, hf, registry

STAGES = [
    (target_name, stage)
    for target_name in registry.targets()
    for stage in registry.post_training_stages(registry.resolve(target_name))
]
STAGE_IDS = [f"{t}.{s['stage']}" for t, s in STAGES]


@pytest.mark.parametrize(("target_name", "stage"), STAGES, ids=STAGE_IDS)
def test_saved_row_still_yields_its_prompt(target_name, stage):
    saved = row_fixture(target_name, stage["stage"])
    # The fixture records the dataset and path it was captured under; a registry
    # edit that repoints a stage has to re-capture rather than silently compare
    # against the old schema.
    assert saved["dataset"] == stage["hf_dataset"]
    assert saved["prompt_path"] == stage["prompt_path"]

    prompt = extract.extract_prompt(saved["row"], stage["prompt_path"])

    assert prompt, f"{stage['hf_dataset']}: prompt_path {stage['prompt_path']!r} extracted nothing"
    assert len(prompt) == saved["prompt_chars"]
    assert prompt.startswith(saved["prompt_head"])
    assert prompt.endswith(saved["prompt_tail"])


@pytest.mark.parametrize(("target_name", "stage"), STAGES, ids=STAGE_IDS)
def test_saved_row_declares_every_column_the_registry_reads(target_name, stage):
    """Every requested column, not just one of them.

    `cmd_sources` renders each column independently and drops one the dataset
    doesn't have, so a stale name costs exactly one breakdown and nothing says
    so — which is how 32B DPO shipped showing only `preference_type` while its
    mix column sat there unread under a different name."""
    saved = row_fixture(target_name, stage["stage"])
    missing = [c for c in stage["source_columns"] if c not in saved["columns"]]
    assert not missing, (
        f"{stage['hf_dataset']}: source_columns {missing} do not exist on the row"
        f" (columns: {', '.join(saved['columns'])})"
    )


def test_every_registry_prompt_path_is_covered():
    """Pinned to the four shapes extract.py implements and the module docstring
    documents, so a fifth has to arrive with a case here rather than as an
    untested path."""
    covered = {stage["prompt_path"] for _, stage in STAGES}
    assert covered == {"messages", "chosen_messages", "prompt", "conversation"}


@pytest.mark.live
@pytest.mark.parametrize(("target_name", "stage"), STAGES, ids=STAGE_IDS)
def test_live_row_still_yields_a_prompt(target_name, stage):
    """The upstream canary: extraction against a row fetched right now."""
    saved = row_fixture(target_name, stage["stage"])
    j = hf._get(
        "rows",
        dataset=stage["hf_dataset"],
        config="default",
        split="train",
        offset=saved["row_offset"],
        length=1,
    )
    row = j["rows"][0]["row"]

    assert extract.extract_prompt(row, stage["prompt_path"]), (
        f"{stage['hf_dataset']}: prompt_path {stage['prompt_path']!r} extracted nothing"
        f" from live row {saved['row_offset']} (columns: {', '.join(sorted(row))})"
    )
    missing = [c for c in stage["source_columns"] if c not in row]
    assert not missing, (
        f"{stage['hf_dataset']}: source_columns {missing} are gone upstream"
        f" (columns: {', '.join(sorted(row))})"
    )


def test_unknown_prompt_path_raises():
    with pytest.raises(ValueError, match="Unknown prompt_path"):
        extract.extract_prompt({"messages": []}, "user_turn")


def test_first_user_turn_wins_and_system_turns_are_skipped():
    row = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "second"},
        ]
    }
    assert extract.extract_prompt(row, "messages") == "first"


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": None},
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "no user turn"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": "   \n\t "}]},
        {"messages": ["not a dict"]},
    ],
)
def test_no_usable_user_turn_is_none(row):
    """None, not "" — the callers filter on truthiness to decide what to classify."""
    assert extract.extract_prompt(row, "messages") is None


def test_chosen_messages_reads_the_chosen_side():
    row = {
        "chosen": [{"role": "user", "content": "shared question"}],
        "rejected": [{"role": "user", "content": "should be ignored"}],
    }
    assert extract.extract_prompt(row, "chosen_messages") == "shared question"


def test_conversation_reads_the_first_user_turn():
    """Chat logs interleave both sides under one column; only the opening user
    turn is the prompt."""
    row = {
        "conversation": [
            {"role": "user", "content": "what did you mean?"},
            {"role": "assistant", "content": "I meant this."},
            {"role": "user", "content": "a follow-up, not the prompt"},
        ]
    }
    assert extract.extract_prompt(row, "conversation") == "what did you mean?"


def test_prompt_as_string():
    assert extract.extract_prompt({"prompt": "  solve for x  "}, "prompt") == "solve for x"


def test_prompt_as_chat_messages():
    """Some RL mixes store `prompt` as a message list rather than a string."""
    row = {"prompt": [{"role": "user", "content": "what is 2 + 2?"}]}
    assert extract.extract_prompt(row, "prompt") == "what is 2 + 2?"


def test_prompt_falls_back_to_source_prompt():
    row = {"prompt": "", "source_prompt": [{"role": "user", "content": "from the source"}]}
    assert extract.extract_prompt(row, "prompt") == "from the source"


def test_non_string_prompt_is_stringified():
    """The datasets-server hands back whatever the column holds; a number is a
    prompt of length one, not a crash."""
    assert extract.extract_prompt({"prompt": 42}, "prompt") == "42"


def test_clip_leaves_short_text_alone():
    text = "x" * extract.MAX_STORE_CHARS
    assert extract.clip(text) == text


def test_clip_marks_what_it_cut():
    clipped = extract.clip("x" * (extract.MAX_STORE_CHARS + 1))
    assert clipped.startswith("x" * extract.MAX_STORE_CHARS)
    assert clipped.endswith("…[truncated]")


def test_excerpt_leaves_documents_within_budget_alone():
    text = "y" * 500
    assert extract.excerpt(text, budget=500) == text


def test_excerpt_respects_its_budget_including_the_markers():
    """The property the classifier/site agreement rests on: what comes back is
    never longer than the budget, markers included."""
    text = "".join(chr(ord("a") + i % 26) for i in range(50_000))
    for budget in (100, 1_000, extract.MAX_DOCUMENT_CHARS):
        out = extract.excerpt(text, budget=budget)
        assert len(out) <= budget, budget
        assert out.count(extract.EXCERPT_MARKER) == 2


def test_excerpt_spans_the_whole_document():
    """Evenly spaced spans, so the middle and end get a vote — the point of not
    just truncating."""
    text = "A" * 1000 + "B" * 1000 + "C" * 1000
    out = extract.excerpt(text, budget=300)
    assert out.startswith("A")
    assert "B" in out
    assert out.endswith("C")


def test_the_classifier_and_the_site_see_the_same_text():
    """extract.py commits to this equality in a comment; if the two budgets ever
    drift, "read the matched documents" stops being true of the site."""
    assert extract.MAX_DOCUMENT_CHARS == extract.MAX_STORE_CHARS
