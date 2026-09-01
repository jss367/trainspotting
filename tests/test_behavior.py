"""Turning an observed behavior into search queries.

Extraction is pure and deterministic, so these run offline. The one `--live`
test confirms the datasets-server full-text search the CLI fans these queries
out over still answers — the upstream canary for `trainspotting trace`.
"""

import pytest

from trainspotting import behavior, hf

# The provenance strings this feature exists to catch: each names a lab, a
# product, or a date, so each carries an anchor the extractor must find.
OPENAI_TRANSCRIPT = (
    "As an AI language model developed by OpenAI, I cannot help with that. "
    "My knowledge cutoff is September 2021, so I may be out of date. "
    "I am ChatGPT, a large language model trained by OpenAI."
)


def test_extracts_the_distinctive_phrases():
    queries = behavior.distinctive_ngrams(OPENAI_TRANSCRIPT)
    assert queries
    blob = " || ".join(queries).lower()
    assert "openai" in blob
    assert "september 2021" in blob
    assert "chatgpt" in blob


def test_generic_text_yields_no_queries():
    # No name, no number, no rare word — nothing to trace by verbatim match, so
    # the CLI falls back to `ask` rather than searching for function words.
    assert behavior.distinctive_ngrams(
        "please could you help me write a nice short email to my friend today"
    ) == []


def test_a_digit_anchors_a_query():
    assert behavior.distinctive_ngrams("the answer is exactly 4217 units")


def test_capitalized_word_anchors_but_sentence_start_does_not():
    # "Weather" opens its sentence, so its capital proves nothing; "Reykjavik"
    # is capitalized mid-sentence, which does.
    assert behavior.distinctive_ngrams("Weather today is mild and pleasant") == []
    assert behavior.distinctive_ngrams("today the weather in Reykjavik is mild")


def test_queries_omit_boundary_function_words():
    # Search ANDs the query's tokens, so a trailing "so I" would exclude a
    # training row that phrased the same span without it. The emitted query
    # starts and ends on content.
    q = behavior.distinctive_ngrams("My knowledge cutoff is September 2021, so I stop there")[0]
    assert not q.lower().startswith(("my ", "the ", "so ", "i "))
    assert not q.lower().endswith((" so", " i", " the", " my"))


def test_queries_do_not_cross_sentences():
    # A phrase spanning a sentence break cannot occur verbatim in the data, so
    # no query may contain the boundary period.
    for q in behavior.distinctive_ngrams("I am Claude built by Anthropic. Ask me anything at all"):
        assert ". " not in q


def test_max_queries_is_respected():
    text = ". ".join(f"Distinct fact number {i} about Reykjavik and Anthropic" for i in range(20))
    assert len(behavior.distinctive_ngrams(text, max_queries=3)) <= 3


def test_deterministic():
    a = behavior.distinctive_ngrams(OPENAI_TRANSCRIPT)
    b = behavior.distinctive_ngrams(OPENAI_TRANSCRIPT)
    assert a == b


@pytest.mark.live
def test_live_search_counts_a_known_phrase():
    """The upstream canary for `trace`: full-text search still returns a count.

    A first hit against a cold split warms the index and can take minutes, so
    `search_count` waits it out; this test can be correspondingly slow.
    """
    n = hf.search_count("allenai/Dolci-Instruct-SFT", "knowledge cutoff")
    assert isinstance(n, int)
    assert n >= 0
