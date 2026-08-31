"""What `raw` on a stored turn promises.

The site makes two claims that are true or false rather than approximate — this
span is a shared opening, this pair carries no gradient — and both rest on
comparing stored turns byte for byte. `raw` is the only thing that licenses
that comparison, so it has to mean exactly "this record holds the turn as it was
written", and nothing observable downstream can stand in for it.
"""

from trainspotting import context


def turn(message):
    return context._turns([message])[0]


def test_plain_turn_is_raw():
    assert turn({"role": "assistant", "content": "hello"})["raw"] is True


def test_truncated_turn_is_not_raw():
    long = "x" * (context.MAX_TEXT + 1)
    t = turn({"role": "assistant", "content": long})
    assert "raw" not in t
    assert t["chars"] == context.MAX_TEXT + 1


def test_thinking_span_is_not_raw():
    t = turn({"role": "assistant", "content": "<think>weighing it up</think>the answer"})
    assert "raw" not in t
    assert t["reasoning"]["text"] == "weighing it up"


def test_empty_thinking_span_is_not_raw():
    """The case no downstream check can see: the markers go, and with an empty
    body no reasoning field is left to show that anything was removed."""
    t = turn({"role": "assistant", "content": "<think></think>the answer"})
    assert "reasoning" not in t
    assert t["text"] == "the answer"
    assert "raw" not in t
    # ... and it must not be mistaken for the message that really is just that
    assert turn({"role": "assistant", "content": "the answer"})["raw"] is True


def test_output_beside_content_is_not_raw():
    """`search` reads these as part of the turn; this record does not keep them,
    so two turns can agree on `content` and still differ in what was scored."""
    for field in ("reasoning_content",) + search_fields():
        t = turn({"role": "assistant", "content": "the answer", field: [{"a": 1}]})
        assert "raw" not in t, f"{field} left the turn marked raw"


def search_fields():
    from trainspotting import search

    return search.STRUCTURED_TURN_FIELDS + search.INPUT_TURN_FIELDS


def test_empty_auxiliary_field_still_raw():
    """An empty list is the schema's way of saying there were no tool calls."""
    assert turn({"role": "assistant", "content": "the answer", "tool_calls": []})["raw"] is True


def test_auxiliary_only_turn_is_kept_and_not_raw():
    """A turn can be nothing but a tool call. Dropping it would let two
    completions that differ only there read as the same conversation."""
    turns = context._turns(
        [
            {"role": "assistant", "content": None, "tool_calls": [{"name": "search"}]},
            {"role": "assistant", "content": "the answer"},
        ]
    )
    assert len(turns) == 2
    assert turns[0]["text"] == "" and "raw" not in turns[0]
    assert turns[0]["omitted"] == ["tool_calls"]
    assert turns[1]["raw"] is True


def test_empty_message_is_kept():
    """An empty message is still a turn in the sequence the model was scored on:
    it contributes its role header and end-of-turn token, and `_shared_turns`
    branches at one that appears on a single side. Dropping it would let two
    completions differing only there read as the same conversation. It is stored
    faithfully — empty is what it said — so it stays raw."""
    for message in ({"role": "assistant", "content": ""},
                    {"role": "assistant", "content": None},
                    {"role": "assistant", "content": None, "tool_calls": []}):
        t = turn(message)
        assert t["text"] == "" and t["chars"] == 0
        assert t["raw"] is True
        assert "omitted" not in t


def test_an_empty_turn_on_one_side_shifts_the_rest():
    """Which is how it protects the branch point: the turn lists stop lining up
    at its position, so nothing after it reads as shared."""
    with_empty = context._turns(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
         {"role": "assistant", "content": "the answer"}]
    )
    without = context._turns(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "the answer"}]
    )
    assert len(with_empty) == 3 and len(without) == 2
    assert with_empty[1] != without[1]


def test_falsy_content_that_is_not_empty_is_kept():
    assert turn({"role": "assistant", "content": 0})["text"] == "0"


def test_non_messages_are_skipped():
    assert context._turns(["not a message", None, 7]) == []


def test_omitted_names_what_was_left_out():
    t = turn({"role": "assistant", "content": "the answer", "refusal": "no"})
    assert t["omitted"] == ["refusal"]
    assert "raw" not in t
