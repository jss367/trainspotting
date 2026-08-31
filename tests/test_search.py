"""Searching a training example rather than a prompt.

The point of the layer is that a string in a response is findable and correctly
attributed, so the tests are about which side a hit lands on — including the two
ways that goes wrong: a shared prompt counted on both sides of a DPO pair, and a
rejected completion read as if it were a chosen one.
"""

import re

import pytest
from conftest import row_fixture

from trainspotting import registry, search

STAGES = [
    (model_name, stage["stage"])
    for model_name, model in registry.MODELS.items()
    for stage in registry.post_training_stages(model)
]
STAGE_IDS = [f"{m}.{s}" for m, s in STAGES]

# The side that carries text beyond the prompt, per stage. RL rows store no
# response at all, so the most this can promise there is the answer key.
NON_PROMPT_SIDE = {"sft": "response", "dpo": "chosen", "rlvr": "verifier"}


def compile_(pattern, case_sensitive=False):
    return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_every_stage_yields_more_than_its_prompt(model_name, stage):
    """The canary for the whole layer: if a stage's fields are prompt-only, a
    search of it is the prompt-only search this exists to replace."""
    row = row_fixture(model_name, stage)["row"]
    sides = {f["side"] for f in search.fields(row, stage)}

    assert "prompt" in sides
    assert NON_PROMPT_SIDE[stage] in sides, f"{model_name}/{stage}: only found {sides}"
    assert sides <= set(search.SIDES[stage])


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_a_response_string_is_found_in_a_real_row(model_name, stage):
    """Take a phrase out of a non-prompt field of a saved row and search for it:
    it comes back, on the side it was taken from."""
    row = row_fixture(model_name, stage)["row"]
    side = NON_PROMPT_SIDE[stage]
    field = next(f for f in search.fields(row, stage) if f["side"] == side)
    phrase = field["text"][20:60]

    hits = search.search_row(row, stage, compile_(re.escape(phrase)))

    assert side in {h["side"] for h in hits}


def test_a_dpo_prompt_is_counted_once_not_on_both_sides():
    """Both completions repeat the prompt verbatim. Attributing it to chosen and
    rejected would report every prompt hit as a hit on both sides of the pair."""
    row = {
        "prompt": "who are you?",
        "chosen": [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": "I am Olmo, trained by Ai2."},
        ],
        "rejected": [
            {"role": "user", "content": "who are you?"},
            {"role": "assistant", "content": "I am ChatGPT, trained by OpenAI."},
        ],
    }

    prompt_fields = [f for f in search.fields(row, "dpo") if f["side"] == "prompt"]

    assert len(prompt_fields) == 1
    assert [f["side"] for f in search.fields(row, "dpo")] == ["prompt", "chosen", "rejected"]


def test_the_same_string_on_each_dpo_side_is_a_different_finding():
    """"I am ChatGPT" in rejected teaches the opposite of the same string in
    chosen, so the two cannot share a count."""
    pair = {
        "prompt": "who are you?",
        "chosen": [{"role": "assistant", "content": "I am ChatGPT."}],
        "rejected": [{"role": "assistant", "content": "I am Olmo."}],
    }
    flipped = {
        "prompt": "who are you?",
        "chosen": [{"role": "assistant", "content": "I am Olmo."}],
        "rejected": [{"role": "assistant", "content": "I am ChatGPT."}],
    }
    pattern = compile_("i am chatgpt")

    assert [h["side"] for h in search.search_row(pair, "dpo", pattern)] == ["chosen"]
    assert [h["side"] for h in search.search_row(flipped, "dpo", pattern)] == ["rejected"]

    records = [
        {"hits": search.search_row(pair, "dpo", pattern)},
        {"hits": search.search_row(flipped, "dpo", pattern)},
    ]
    assert search.side_counts(records, "dpo") == {"prompt": 0, "chosen": 1, "rejected": 1}
    assert search.pair_split(records) == {"chosen_only": 1, "rejected_only": 1, "both": 0}


def test_a_hit_on_both_sides_is_neither():
    """A string in both completions says nothing about which way the pair pushes,
    and counting it as a chosen hit would claim it does."""
    row = {
        "chosen": [{"role": "assistant", "content": "As ChatGPT I can help."}],
        "rejected": [{"role": "assistant", "content": "ChatGPT here!"}],
    }
    records = [{"hits": search.search_row(row, "dpo", compile_("chatgpt"))}]

    assert search.pair_split(records) == {"chosen_only": 0, "rejected_only": 0, "both": 1}
    assert search.side_counts(records, "dpo") == {"prompt": 0, "chosen": 1, "rejected": 1}


def test_every_assistant_turn_is_searched_not_just_the_first():
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "who made you?"},
            {"role": "assistant", "content": "OpenAI did."},
        ]
    }

    hits = search.search_row(row, "sft", compile_("openai"))

    assert [(h["side"], h["turn"]) for h in hits] == [("response", 3)]


def test_reasoning_content_is_searched():
    """A separate column in the WildChat-derived mixes; text that lives only
    there is still text the model is fit to."""
    row = {
        "messages": [
            {"role": "assistant", "content": "Sure.", "reasoning_content": "I am ChatGPT."}
        ]
    }

    hits = search.search_row(row, "sft", compile_("i am chatgpt"))

    assert [h["role"] for h in hits] == ["assistant/reasoning"]
    assert hits[0]["side"] == "response"


def test_a_tool_call_is_searched_and_lands_on_the_response_side():
    """The Dolci message schemas keep tool and function calls in their own
    columns. A turn whose output is a call has an empty `content`, so reading
    only that would report zero for a tool name."""
    row = {
        "messages": [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}
                ],
            },
        ]
    }

    for pattern in ("get_weather", "oslo"):
        hits = search.search_row(row, "sft", compile_(pattern))
        assert [(h["side"], h["role"]) for h in hits] == [("response", "assistant/tool_calls")], pattern


def test_a_refusal_string_is_response_text():
    row = {"messages": [{"role": "assistant", "refusal": "I won't help with that."}]}

    hits = search.search_row(row, "sft", compile_("won't help"))

    assert [(h["side"], h["role"]) for h in hits] == [("response", "assistant/refusal")]


def test_a_dpo_tool_call_keeps_its_side():
    row = {
        "chosen": [{"role": "assistant", "function_call": {"name": "refuse_politely"}}],
        "rejected": [{"role": "assistant", "function_call": {"name": "run_exploit"}}],
    }

    assert [h["side"] for h in search.search_row(row, "dpo", compile_("run_exploit"))] == ["rejected"]
    assert [h["side"] for h in search.search_row(row, "dpo", compile_("refuse_politely"))] == ["chosen"]


def test_offered_tools_are_prompt_text_not_response_text():
    """`functions` is the menu of tools the turn was offered, not anything the
    model produced, so a hit there is not a hit in the response."""
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": "Sure.",
                "functions": [{"name": "delete_everything", "description": "dangerous"}],
            }
        ]
    }

    hits = search.search_row(row, "sft", compile_("delete_everything"))

    assert [(h["side"], h["role"]) for h in hits] == [("prompt", "assistant/functions")]


def test_a_thinking_span_inside_a_response_is_searched():
    """`context.build` folds the reasoning away for display; search does not,
    because a model that says it while thinking still said it."""
    row = {"messages": [{"role": "assistant", "content": "<think>I am ChatGPT</think>Hello."}]}

    assert search.search_row(row, "sft", compile_("i am chatgpt"))


def test_rl_rows_separate_the_answer_key_from_the_reference_rollouts():
    """No RL row stores a response. What it does store is what the verifier
    checks and what the reference model generated, which are different claims
    about the example and so cannot share a side."""
    row = {
        "prompt": "user: what is 2 + 2?",
        "ground_truth": ["4"],
        "outputs": ["The answer is 4."],
    }

    sides = {f["side"]: f["role"] for f in search.fields(row, "rlvr")}

    assert sides == {"prompt": "prompt", "verifier": "ground_truth", "rollout": "output"}


def test_an_rl_prompt_carried_twice_is_counted_once():
    row = {"prompt": "solve it", "source_prompt": [{"role": "user", "content": "solve it"}]}

    assert len([f for f in search.fields(row, "rlvr") if f["side"] == "prompt"]) == 1


def test_case_insensitive_by_default_and_exact_when_asked():
    row = {"messages": [{"role": "assistant", "content": "I am chatgpt."}]}

    assert search.search_row(row, "sft", compile_("ChatGPT"))
    assert not search.search_row(row, "sft", compile_("ChatGPT", case_sensitive=True))


def test_a_hit_counts_every_occurrence_in_the_field():
    row = {"messages": [{"role": "assistant", "content": "ChatGPT, ChatGPT, ChatGPT."}]}

    hits = search.search_row(row, "sft", compile_("chatgpt"))

    assert len(hits) == 1
    assert hits[0]["count"] == 3


def test_side_counts_count_rows_not_hits():
    """Forty repeats in one response are one row's worth of evidence."""
    row = {"messages": [{"role": "assistant", "content": "ChatGPT " * 40}]}
    records = [{"hits": search.search_row(row, "sft", compile_("chatgpt"))}]

    assert search.side_counts(records, "sft") == {"prompt": 0, "response": 1}


def test_snippet_marks_what_it_cut():
    text = "x" * 500 + "ChatGPT" + "y" * 500
    pattern = compile_("chatgpt")

    snip = search.snippet(text, pattern.search(text), pad=10)

    assert snip == "…" + "x" * 10 + "ChatGPT" + "y" * 10 + "…"


def test_snippet_of_a_short_field_is_the_whole_field():
    text = "I am ChatGPT."

    assert search.snippet(text, compile_("chatgpt").search(text)) == text


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": None},
        {"messages": [{"role": "assistant", "content": ""}]},
        {"messages": [{"role": "assistant", "content": "   "}]},
        {"messages": ["not a dict"]},
    ],
)
def test_a_row_with_no_text_yields_no_fields(row):
    assert search.fields(row, "sft") == []


def test_list_valued_cells_are_searched_as_text():
    """`ground_truth` and `outputs` arrive as lists; joining them is what makes
    a string spanning the list findable at all."""
    row = {"prompt": "q", "ground_truth": ["alpha", "beta"]}

    assert search.search_row(row, "rlvr", compile_("beta"))


def test_every_stage_declares_its_sides():
    """Pinned to the stages the registry has, so a fourth arrives with its own
    side names rather than falling into the RL branch by default."""
    assert set(search.SIDES) == {s for _, s in STAGES}
