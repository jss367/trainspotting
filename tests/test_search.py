"""Locating a phrase inside a training example, for every schema the registry claims.

The /search index only says *that* a row contains the query's words. Everything
attribution rests on — is the phrase really there verbatim, and is it in a
field the model is trained toward or away from — comes from search.find_matches,
so what it reports and what it refuses to claim both get pinned here.
"""

import pytest
from conftest import row_fixture

from trainspotting import hf, registry, search
from trainspotting.search import find_matches, texts

STAGES = [
    (model_name, stage)
    for model_name, model in registry.MODELS.items()
    for stage in registry.post_training_stages(model)
]
STAGE_IDS = [f"{m}.{s['stage']}" for m, s in STAGES]


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_every_saved_row_yields_prompt_text(model_name, stage):
    """A schema this walker doesn't understand shows up as a search that finds
    the phrase in nothing, which reads as "not in the data" — the one wrong
    answer this tool must not give. Every real row must yield at least the
    prompt."""
    saved = row_fixture(model_name, stage["stage"])
    fields = list(texts(saved["row"], stage["stage"]))
    assert any(w == "prompt" and t.strip() for w, t in fields), (
        f"{stage['hf_dataset']}: texts() found no prompt text"
    )


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        list(texts({}, "midtrain"))


def test_sft_separates_prompt_from_response():
    row = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am ChatGPT."},
        ]
    }
    assert list(texts(row, "sft")) == [
        ("prompt", "You are a helpful assistant."),
        ("prompt", "Who are you?"),
        ("response", "I am ChatGPT."),
    ]


def test_dpo_reads_the_shared_prompt_once():
    """Instruct-DPO duplicates the user turn into both sides; counting it from
    `rejected` too would double every prompt hit."""
    row = {
        "chosen": [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": "I am Olmo."},
        ],
        "rejected": [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": "I am ChatGPT."},
        ],
    }
    assert list(texts(row, "dpo")) == [
        ("prompt", "who are you?"),
        ("chosen", "I am Olmo."),
        ("rejected", "I am ChatGPT."),
    ]


def test_dpo_top_level_prompt_wins_over_the_user_turns():
    """Think-DPO carries the prompt both ways; reading both would double it."""
    row = {
        "prompt": "who are you?",
        "chosen": [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": "I am Olmo."},
        ],
        "rejected": [{"role": "assistant", "content": "I am ChatGPT."}],
    }
    assert list(texts(row, "dpo")) == [
        ("prompt", "who are you?"),
        ("chosen", "I am Olmo."),
        ("rejected", "I am ChatGPT."),
    ]


def test_rlvr_splits_prompt_rollouts_and_verifier():
    row = {
        "prompt": "Prove it.",
        "outputs": ["Here is a proof …", None],
        "ground_truth": ["42"],
        "solution": "x = 42",
        "constraint": None,
    }
    assert list(texts(row, "rlvr")) == [
        ("prompt", "Prove it."),
        ("rollout", "Here is a proof …"),
        ("verifier", "42"),
        ("verifier", "x = 42"),
    ]


def test_source_labels_are_not_example_text():
    """A source named for the query must land in the source breakdown, not the
    text match — otherwise every row of a "chatgpt_synthetic" mix reads as
    containing the phrase."""
    row = {
        "messages": [{"role": "user", "content": "hello"}],
        "source_dataset": "chatgpt_synthetic",
        "id": "chatgpt-000123",
    }
    assert list(texts(row, "sft")) == [("prompt", "hello")]


SFT_ROW = {
    "messages": [
        {"role": "user", "content": "Tell me about yourself."},
        {"role": "assistant", "content": "Certainly! I am ChatGPT, a large language model."},
    ]
}


def test_exact_phrase_is_found_case_insensitively():
    m = find_matches(SFT_ROW, "sft", "i am chatgpt")

    assert m["exact"]
    assert m["where"] == ["response"]
    # The snippet quotes the row's own casing, not the query's.
    assert m["snippets"] == ["Certainly! «I am ChatGPT», a large language model."]


def test_snippet_marks_the_match():
    m = find_matches(SFT_ROW, "sft", "I am ChatGPT")

    assert m["snippets"] == ["Certainly! «I am ChatGPT», a large language model."]


def test_snippet_elides_and_flattens_long_text():
    row = {"messages": [{"role": "user", "content": "x" * 500 + "\nneedle\n" + "y" * 500}]}
    m = find_matches(row, "sft", "needle")

    (snippet,) = m["snippets"]
    assert snippet.startswith("…") and snippet.endswith("…")
    assert "«needle»" in snippet
    assert "\n" not in snippet
    assert len(snippet) < 2 * search.SNIPPET_CONTEXT + 20


def test_scattered_words_are_not_exact():
    """The index matches rows holding the words in any order and field; the
    record must say so rather than counting them as the phrase — but still show
    a snippet of what the index saw (the longest word)."""
    row = {
        "messages": [
            {"role": "user", "content": "Is ChatGPT any good?"},
            {"role": "assistant", "content": "I am not able to compare myself to others."},
        ]
    }
    m = find_matches(row, "sft", "I am ChatGPT")

    assert not m["exact"]
    assert m["where"] == ["prompt"]
    assert m["snippets"] == ["Is «ChatGPT» any good?"]


def test_a_row_can_match_in_several_places():
    row = {
        "chosen": [
            {"role": "user", "content": "say ChatGPT"},
            {"role": "assistant", "content": "ChatGPT"},
        ],
        "rejected": [{"role": "assistant", "content": "No — ChatGPT is someone else."}],
    }
    m = find_matches(row, "dpo", "ChatGPT")

    assert m["exact"]
    assert m["where"] == ["prompt", "chosen", "rejected"]


@pytest.mark.live
def test_live_search_returns_indexed_rows():
    """The upstream canary: the /search response shape cmd_search leans on —
    num_rows_total, row_idx, and rows the walker can read. Uses the smallest
    Dolci mix and a word every instruction dataset contains."""
    j = hf.search_rows("allenai/Dolci-Instruct-RL", "question", length=10)

    assert j["num_rows_total"] > 0
    assert j["rows"], "the index matched rows but returned none"
    first = j["rows"][0]
    assert isinstance(first["row_idx"], int)
    assert list(texts(first["row"], "rlvr")), "walker got no text out of a live RL row"


def test_snippets_are_capped_but_where_is_not():
    row = {
        "messages": [
            {"role": "user", "content": "echo " + "needle " * 10},
            {"role": "assistant", "content": "needle " * 10},
        ]
    }
    m = find_matches(row, "sft", "needle")

    assert len(m["snippets"]) == search.MAX_SNIPPETS
    assert m["where"] == ["prompt", "response"]
