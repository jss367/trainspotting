"""Parsing the classifier's reply.

`_parse` is the only thing standing between a model that wandered off the
requested format and a batch of silently unlabelled prompts, so both halves
matter: what it recovers, and what it refuses to invent.
"""

import pytest

from trainspotting import classify

YES_NO = ["yes", "no"]


def test_clean_array():
    text = '[{"i": 0, "label": "honesty"}, {"i": 1, "label": "capability"}]'

    assert classify._parse(text, 2) == {0: "honesty", 1: "capability"}


def test_prose_wrapped_output():
    """The system prompt asks for only JSON; models add a preamble anyway."""
    text = (
        "Here are the labels for the prompts you gave me:\n"
        '[{"i": 0, "label": "honesty"}, {"i": 1, "label": "tool_use"}]\n'
        "Let me know if you want the reasoning for any of them."
    )

    assert classify._parse(text, 2) == {0: "honesty", 1: "tool_use"}


def test_markdown_fenced_output():
    text = '```json\n[{"i": 0, "label": "tool_use"}]\n```'

    assert classify._parse(text, 1) == {0: "tool_use"}


def test_truncated_output_yields_nothing():
    """Cut off at max_tokens mid-array. Better an unlabelled batch than labels
    invented from a half-parsed array."""
    text = '[{"i": 0, "label": "honesty"}, {"i": 1, "label": "capa'

    assert classify._parse(text, 2) == {}


def test_no_array_at_all_yields_nothing():
    text = "I labeled the first prompt as honesty and the second as capability."

    assert classify._parse(text, 2) == {}


@pytest.mark.xfail(
    strict=True,
    reason="the `\\[.*\\]` search is greedy, so a bracket after the array "
    "(a citation, a footnote marker) swallows it and loses the whole batch",
)
def test_a_bracket_after_the_array_should_not_lose_the_batch():
    text = 'Labels:\n[{"i": 0, "label": "honesty"}]\nSee note [1] for the borderline case.'

    assert classify._parse(text, 1) == {0: "honesty"}


def test_labels_outside_the_taxonomy_are_dropped():
    """Including case variants — the site groups by exact label string."""
    text = '[{"i": 0, "label": "Honesty"}, {"i": 1, "label": "safety"}, {"i": 2, "label": "helpfulness"}]'

    assert classify._parse(text, 3) == {2: "helpfulness"}


def test_indices_outside_the_batch_are_dropped():
    """`labels[start + i]` would write into another batch's slot."""
    text = '[{"i": 0, "label": "honesty"}, {"i": 9, "label": "honesty"}, {"i": -1, "label": "honesty"}]'

    assert classify._parse(text, 2) == {0: "honesty"}


def test_malformed_items_are_skipped_without_losing_the_good_ones():
    text = (
        '[null, "honesty", {"label": "honesty"}, {"i": "x", "label": "honesty"}, '
        '{"i": 1, "label": "honesty"}]'
    )

    assert classify._parse(text, 2) == {1: "honesty"}


def test_a_stringified_index_is_accepted():
    assert classify._parse('[{"i": "0", "label": "honesty"}]', 1) == {0: "honesty"}


def test_a_repeated_index_takes_the_last_label():
    text = '[{"i": 0, "label": "honesty"}, {"i": 0, "label": "capability"}]'

    assert classify._parse(text, 1) == {0: "capability"}


def test_a_partial_answer_is_kept():
    """A batch of 20 that came back with 18 labels leaves two prompts as None
    rather than discarding the 18."""
    text = '[{"i": 0, "label": "honesty"}, {"i": 2, "label": "capability"}]'

    assert classify._parse(text, 3) == {0: "honesty", 2: "capability"}


def test_question_mode_accepts_only_yes_and_no():
    text = '[{"i": 0, "label": "yes"}, {"i": 1, "label": "no"}, {"i": 2, "label": "honesty"}]'

    assert classify._parse(text, 3, YES_NO) == {0: "yes", 1: "no"}


def test_taxonomy_labels_are_rejected_in_question_mode_and_vice_versa():
    """The two modes share a parser; a label from the wrong mode is a sign the
    wrong system prompt went out, so it must not count."""
    assert classify._parse('[{"i": 0, "label": "yes"}]', 1) == {}
    assert classify._parse('[{"i": 0, "label": "helpfulness"}]', 1, YES_NO) == {}


def test_empty_array():
    assert classify._parse("[]", 3) == {}


def test_the_system_prompt_lists_exactly_the_taxonomy():
    """A label in LABELS that the model is never told about can never be
    assigned; one in the prompt but not LABELS gets parsed away."""
    for label in classify.LABELS:
        assert f"- {label}:" in classify.SYSTEM


def test_question_prompts_interpolate_the_question():
    question = "does this teach the model to care about human lives?"

    for template in (classify.ASK_SYSTEM, classify.ASK_DOC_SYSTEM):
        rendered = template.format(question=question)
        assert question in rendered
        # The JSON example in these templates is brace-escaped for .format; if an
        # escape is wrong the rendering raises or eats the braces.
        assert '{"i": <index>, "label": "yes" or "no"}' in rendered
