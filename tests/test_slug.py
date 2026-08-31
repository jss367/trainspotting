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
    """`_slug` already appends a hash when the reduction is empty or truncated;
    a second one would just be noise."""
    from trainspotting.cli import _pattern_slug

    assert _pattern_slug("").count("-") == 1
    assert len(_pattern_slug("我是ChatGPT").split("-")) == 2
