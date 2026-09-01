"""Turning an observed behavior into search queries.

Extraction is pure and deterministic, so these run offline, and so is the CLI's
reporting once the counts are stubbed. The `--live` tests are the upstream
canaries for `trainspotting trace`: that the datasets-server full-text search
still answers, and that its index still reaches the nested columns the Dolci
transcripts live in.
"""

import sys

import pytest

from trainspotting import behavior, cli, hf

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


def test_a_closing_quote_still_ends_the_sentence():
    # `OpenAI." Next` is two sentences. Splitting only on punctuation-then-space
    # left them as one segment, which let a query straddle the join and promoted
    # "Next" — a sentence-opening capital, so no evidence of anything — to an
    # anchor by putting it mid-segment.
    text = 'I was made by OpenAI." Next sentence mentions Paris and 1999.'
    assert list(behavior._sentences(text)) == [
        "I was made by OpenAI.",
        "Next sentence mentions Paris and 1999.",
    ]
    for q in behavior.distinctive_ngrams(text):
        assert '." ' not in q


def test_two_anchors_in_one_sentence_get_separate_queries():
    # Fourteen words, so every eight-word window overlaps every other one.
    # Requiring disjoint windows capped this input at a single query and made
    # the date — half the behavior being traced — unreachable at any
    # --max-queries.
    queries = behavior.distinctive_ngrams(
        "As an AI language model developed by OpenAI,"
        " my knowledge cutoff is September 2021."
    )
    blob = " || ".join(queries).lower()
    assert "openai" in blob
    assert "september 2021" in blob


def test_a_one_word_shift_is_not_a_second_query():
    # The other half of the overlap rule: a window that only slides along by a
    # word or two is the same span, and eight offsets of one sentence would
    # spend the whole --max-queries budget on it.
    queries = behavior.distinctive_ngrams(
        "Alice Bob Carol Dave Eve Frank Grace Heidi Ivan Judy Karl Liam"
    )
    assert len(queries) < 5


def test_max_queries_is_respected():
    text = ". ".join(f"Distinct fact number {i} about Reykjavik and Anthropic" for i in range(20))
    assert len(behavior.distinctive_ngrams(text, max_queries=3)) <= 3


def test_deterministic():
    a = behavior.distinctive_ngrams(OPENAI_TRANSCRIPT)
    b = behavior.distinctive_ngrams(OPENAI_TRANSCRIPT)
    assert a == b


def _run_trace(monkeypatch, capsys, argv, counts):
    """Run `trainspotting trace` with the network stubbed. `counts` maps a
    dataset id to the (matches, partial) every query against it returns."""
    monkeypatch.setattr(hf, "num_rows", lambda dataset, *a, **k: 1_000_000)
    monkeypatch.setattr(
        hf, "search_count", lambda dataset, query, *a, **k: counts[dataset]
    )
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    return capsys.readouterr().out


def test_trace_marks_a_partially_indexed_stage_as_a_lower_bound(monkeypatch, capsys):
    """The server's full-text index stops at the first 5 GB of a split, and the
    Think SFT mixes are 36 GB. Dividing a prefix's matches by the whole split's
    row count reads as a low density, so the largest stages would sort last for
    being large — a ranking that inverts the answer rather than degrading it."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-think", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Think-SFT-7B": (10, True),
            "allenai/Dolci-Think-DPO-7B": (10, False),
            "allenai/Dolci-Think-RL-7B": (10, False),
        },
    )
    sft = next(line for line in out.splitlines() if line.startswith("## sft"))
    dpo = next(line for line in out.splitlines() if line.startswith("## dpo"))
    assert "≥" in sft and "≥" not in dpo
    assert "first 5 GB" in out


def test_trace_escapes_the_queries_it_echoes(monkeypatch, capsys):
    """A query is a slice of text the user pasted. An escape sequence survives
    tokenization, so printing it raw would let a transcript recolour the report
    quoting it back."""
    out = _run_trace(
        monkeypatch,
        capsys,
        [
            "trainspotting",
            "trace",
            "olmo-3-7b-instruct",
            "the \x1b[31mOpenAI model 2021 said so",
        ],
        {
            "allenai/Dolci-Instruct-SFT": (1, False),
            "allenai/Dolci-Instruct-DPO": (1, False),
            "allenai/Dolci-Instruct-RL": (1, False),
        },
    )
    assert "\x1b" not in out
    assert "x1b" in out


@pytest.mark.live
def test_live_search_counts_a_known_phrase():
    """The upstream canary for `trace`: full-text search still returns a count.

    A first hit against a cold split warms the index and can take minutes, so
    `search_count` waits it out; this test can be correspondingly slow.
    """
    n, partial = hf.search_count("allenai/Dolci-Instruct-SFT", "knowledge cutoff")
    assert isinstance(n, int)
    assert n >= 0
    # 3.06 GB of parquet, under the server's 5 GB indexing cap, so this split is
    # the case where a count really is over all of it. The flag flipping here
    # would mean the cap moved and every density printed for this stage became a
    # lower bound.
    assert partial is False


@pytest.mark.live
def test_live_search_reaches_nested_transcript_columns():
    """The load-bearing assumption of `trace`: the index sees the transcript.

    Dolci SFT keeps the whole conversation in one nested `messages` column and
    DPO keeps its completions in nested `chosen`/`rejected`. There is no scalar
    string column holding the assistant text `trace` is looking for, so a search
    that only indexed scalar columns would report a stage's counts off its
    metadata and never say so.

    Instruct DPO is the clean test of it: every text column is a list of
    {role, content} structs, and the scalar strings are a prompt hash, two model
    names and a preference-type label. A word like this one cannot be in any of
    those, so a non-zero count can only have come from inside the completions.
    """
    n, _ = hf.search_count("allenai/Dolci-Instruct-DPO", "photosynthesis")
    assert n > 0
