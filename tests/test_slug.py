"""Naming a result file after the question or pattern that produced it.

A slug that collapses two different runs to one name overwrites a result
silently, which is the one failure a result file cannot report.
"""

from trainspotting.cli import _slug


def test_a_plain_pattern_keeps_its_readable_name():
    assert _slug("I cannot") == "i-cannot"
    assert _slug("Is this about caring?") == "is-this-about-caring"


def test_a_pattern_with_no_ascii_letters_still_gets_a_name_of_its_own():
    """Reduction leaves nothing here, so every such run would write to the same
    file and overwrite an unrelated one."""
    chinese, punctuation = _slug("我是ChatGPT"), _slug(r"^\s*[:;]-\)\s*$")

    assert chinese and punctuation
    assert chinese != punctuation
    assert _slug("我是ChatGPT") == chinese  # and stable across runs


def test_two_long_patterns_agreeing_on_their_first_60_characters_differ():
    head = "as an ai language model i cannot and will not ever help you with"
    assert _slug(head + " maths") != _slug(head + " code")


def test_a_long_slug_stays_a_filename():
    slug = _slug("x " * 200)

    assert len(slug) <= 60 + 9  # 60 characters plus the disambiguating hash
    assert slug.strip("-") == slug


def test_a_regex_keeps_the_punctuation_that_changes_what_it_matches():
    """`a.b` and `a+b` are different searches that reduce to the same `a-b`;
    prose spelled two ways is one question, a regex punctuated two ways is not."""
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("a.b") != _pattern_slug("a+b")
    assert _pattern_slug("I cannot") != _pattern_slug("I  cannot")


def test_a_pattern_that_is_already_its_own_slug_keeps_the_readable_name():
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("i-cannot") == "i-cannot"


def test_a_pattern_slug_is_disambiguated_once_not_twice():
    """One hash, whichever reduction lost the information."""
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("").count("-") == 1
    assert len(_pattern_slug("我是ChatGPT").split("-")) == 2
    assert len(_pattern_slug("x " * 200).split("-")) == 31  # 30 x's plus the hash


def test_case_sensitivity_names_a_different_run():
    """The same pattern with and without --case-sensitive is two regexes, so it
    cannot be two writes to one file."""
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("ChatGPT") != _pattern_slug("ChatGPT", case_sensitive=True)
    assert _pattern_slug("i-cannot") == "i-cannot"
    assert _pattern_slug("i-cannot", case_sensitive=True) != "i-cannot"


def test_literal_and_regex_modes_name_different_runs():
    """The same punctuation is data in one mode and syntax in the other, so
    neither run may silently overwrite the other after an expensive scan."""
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("a.b") != _pattern_slug("a.b", regex=True)
    assert _pattern_slug("i-cannot") != _pattern_slug("i-cannot", regex=True)


def test_a_long_plain_pattern_is_still_a_filename():
    """A 300-character literal is its own slug, so the readable shortcut would
    hand back a basename no filesystem accepts — after the whole sampling run
    had already been paid for."""
    from trainspotting.cli import MAX_SLUG_CHARS, _pattern_slug

    long_a, longer_a = _pattern_slug("a" * 300), _pattern_slug("a" * 301)

    assert len(long_a) <= MAX_SLUG_CHARS + 9
    assert long_a != longer_a
