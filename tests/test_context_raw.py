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
