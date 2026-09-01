"""Turning an observed behavior into search queries.

Extraction is pure and deterministic, so these run offline, and so is the CLI's
reporting once the counts are stubbed. The `--live` tests are the upstream
canaries for `trainspotting trace`: that the datasets-server full-text search
still answers, and that its index still reaches the nested columns the Dolci
transcripts live in.
"""

import sys

import pytest
import requests

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


def test_a_digit_behind_punctuation_in_a_token_still_anchors():
    # `_WORD.search` read only the first word segment of a whitespace-delimited
    # token, so `version-4217` scored as the ordinary word "version" and a
    # lowercase sentence around it produced nothing at all.
    assert behavior.distinctive_ngrams("the build is version-4217 in the log")
    assert behavior.distinctive_ngrams("iso-9001 compliance was claimed")


def test_an_identifier_opening_on_a_stopword_letter_still_anchors():
    """`t`, `s` and `i` are on the stopword list for the contractions they end,
    so an identifier whose first segment is one of them was rejected before its
    digits were ever looked at. A digit is a claim about the whole token, so it
    is read first."""
    assert behavior.distinctive_ngrams("the robot is T-800 and protects people")
    assert behavior.distinctive_ngrams("the index tracks S&P500 closely")
    assert behavior.distinctive_ngrams("the report cites I/O-2024 standards")
    # The contractions the list exists for carry neither signal, so they still
    # reach it and still weigh nothing.
    assert behavior.distinctive_ngrams("don't do that thing please") == []
    assert behavior.distinctive_ngrams("it is what it is") == []


def test_a_segment_too_short_for_a_window_is_still_a_query():
    """The anchor is what makes a query selective, not its length. A three-word
    minimum threw away exactly the segments that are nothing but anchor: the
    sentence splitter breaks at a colon, so a pasted header line reduced to two
    short segments and `trace` answered "no distinctive phrase" to a name."""
    assert behavior.distinctive_ngrams("Knowledge cutoff: September 2021") == ["September 2021"]
    assert behavior.distinctive_ngrams("Assistant: ChatGPT") == ["ChatGPT"]
    assert behavior.distinctive_ngrams("Model: GPT-4") == ["GPT-4"]
    # Still nothing without an anchor, however short the segment.
    assert behavior.distinctive_ngrams("Reply: yes") == []


def test_capitalized_word_anchors_but_sentence_start_does_not():
    # "Weather" opens its sentence, so its capital proves nothing; "Reykjavik"
    # is capitalized mid-sentence, which does.
    assert behavior.distinctive_ngrams("Weather today is mild and pleasant") == []
    assert behavior.distinctive_ngrams("today the weather in Reykjavik is mild")


def test_an_interior_capital_anchors_wherever_the_token_sits():
    """A transcript often opens on the name, and `ChatGPT is a large language
    model...` held no other anchor — so the sentence-initial rule discarded the
    behavior `trace` exists to find and sent the user to `ask`. English
    capitalizes a sentence's first letter and nothing else in it, so a capital
    further into the token is a name however the token is positioned."""
    for text in (
        "ChatGPT is a large language model trained to assist people",
        "OpenAI trained me to be helpful and harmless",
    ):
        assert behavior.distinctive_ngrams(text), text
    # Still not a licence for any capitalized opening word: "Weather" is shaped
    # by the sentence rule and stays evidence of nothing (above).
    assert behavior.distinctive_ngrams("Anthropic is a company that builds models") == []


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


def test_a_second_anchor_in_one_sentence_gets_its_own_query():
    """The README's documented input carries four anchors in one sentence —
    `AI`, `OpenAI`, `September`, `2021` — and every window holding the cutoff
    date overlaps the window holding the provenance claim. Rejecting overlap
    outright meant `September 2021` could not be reached at any
    `--max-queries`, which is half of the example the feature is sold on."""
    text = "As an AI language model developed by OpenAI, my knowledge cutoff is September 2021."

    queries = behavior.distinctive_ngrams(text, max_queries=6)

    assert any("OpenAI" in q for q in queries)
    assert any("September 2021" in q for q in queries), queries


def test_a_window_bringing_no_new_anchor_is_still_dropped():
    """Overlap is allowed for a new anchor, not for a shifted copy of the same
    one — otherwise one anchor would spend the whole query budget."""
    text = "The report from OpenAI says the same thing over and over again."

    queries = behavior.distinctive_ngrams(text, max_queries=6)

    assert len(queries) == 1, queries


def test_a_quoted_sentence_ends_where_the_quote_closes():
    """`OpenAI." Then it` is not a phrase any training row contains, so a query
    must not be stitched across that boundary."""
    text = 'He said "I was made by OpenAI." Then it mentioned September 2021.'

    for q in behavior.distinctive_ngrams(text, max_queries=6):
        assert not ("OpenAI" in q and "September" in q), q


def _run_trace(monkeypatch, capsys, argv, counts, revisions=("abc1234", "abc1234")):
    """Run `trainspotting trace` with the network stubbed. `counts` maps a
    dataset id to the (matches, partial) every query against it returns, and
    `revisions` is what the before/after revision lookups answer."""
    monkeypatch.setattr(hf, "num_rows", lambda dataset, *a, **k: 1_000_000)
    monkeypatch.setattr(
        hf, "search_count", lambda dataset, query, *a, **k: counts[dataset]
    )
    seen = iter(revisions * len(counts))
    monkeypatch.setattr(hf, "dataset_revision", lambda dataset, *a, **k: next(seen))
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


def test_a_partially_indexed_stage_is_left_out_of_the_ranking(monkeypatch, capsys):
    """A lower bound over a denominator counting unsearched rows cannot be
    ordered against an exact density: here SFT's bound is below DPO's real
    figure while its true density may be well above it. Ranking them together
    named the wrong stage as the place to look."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-think", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Think-SFT-7B": (10, True),
            "allenai/Dolci-Think-DPO-7B": (99, False),
            "allenai/Dolci-Think-RL-7B": (0, False),
        },
    )
    heading = out.index("## Not ranked")
    assert out.index("## dpo") < heading < out.index("## sft")
    # The reader is pointed at the stage whose density means what it says, and
    # the bound is called out rather than read as a stage with less of the
    # behavior.
    assert "Dolci-Think-DPO-7B/viewer" in out
    assert "Dolci-Think-SFT-7B/viewer" not in out
    assert "sft reported a lower bound" in out


def test_a_bound_above_every_exact_density_is_the_stage_to_read(monkeypatch, capsys):
    """The one comparison a lower bound does settle. A bound of 200/M against an
    exact 2/M means the partial stage really is the densest — its true figure is
    at least the bound — so refusing to name it would point the reader at a
    stage this run has evidence is not the leader."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-think", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Think-SFT-7B": (100, True),
            "allenai/Dolci-Think-DPO-7B": (1, False),
            "allenai/Dolci-Think-RL-7B": (0, False),
        },
    )
    assert "Dolci-Think-SFT-7B/viewer" in out


def test_the_handoff_goes_to_the_matched_rows_not_a_random_draw(monkeypatch, capsys):
    """`trainspotting search` attributes a hit to a side, but over 300 random
    rows: at the densities a signature string produces it finds none of the
    matches, so a trace that recommended it was sending the reader to a
    confident zero. The viewer's `?q=` runs the same index the count came
    from."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-instruct", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Instruct-SFT": (3, False),
            "allenai/Dolci-Instruct-DPO": (1, False),
            "allenai/Dolci-Instruct-RL": (0, False),
        },
    )
    assert "?q=" in out
    assert "300-row random draw" in out
    # Not presented as the way to see these rows.
    assert "trainspotting search olmo-3-7b-instruct" not in out


def test_nothing_found_does_not_speak_for_the_unindexed_rows(monkeypatch, capsys):
    """Zero over an indexed prefix is not zero over the split, so it cannot join
    the fully searched stages in a flat "nothing here"."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-think", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Think-SFT-7B": (0, True),
            "allenai/Dolci-Think-DPO-7B": (0, False),
            "allenai/Dolci-Think-RL-7B": (0, False),
        },
    )
    assert "No stage contained these phrases" in out
    assert "sft was only partly indexed" in out


def test_a_republish_mid_trace_takes_the_stage_out_of_the_ranking(monkeypatch, capsys):
    """A trace holds a stage open longer than any sampling path — a cold split
    spends minutes building its index — so the window for `main` to move under
    it is the widest in the tool. The row count is read before the searches, so
    a republish leaves the two halves of the ratio describing different trees:
    not a loose estimate but an estimate of nothing, which is why it is reported
    outside the ranking rather than noted inside it."""
    out = _run_trace(
        monkeypatch,
        capsys,
        ["trainspotting", "trace", "olmo-3-7b-instruct", "I am ChatGPT, by OpenAI"],
        {
            "allenai/Dolci-Instruct-SFT": (3, False),
            "allenai/Dolci-Instruct-DPO": (1, False),
            "allenai/Dolci-Instruct-RL": (0, False),
        },
        revisions=("aaaaaaa", "bbbbbbb"),
    )
    assert out.count("republished mid-search: aaaaaaa -> bbbbbbb") == 3
    assert "Stages ranked by matches per million rows" not in out
    # A zero from a crossed stage cannot say the phrases are absent either.
    assert "says nothing about those stages either way" in out
    # And nothing is recommended off one of those figures.
    assert "?q=" not in out


def test_a_crossed_stage_cannot_win_the_recommendation(monkeypatch, capsys):
    """The stage with the largest number is republished, so its density is a
    ratio between two datasets. Leaving it in the ranking would let it take the
    headline and the viewer link on the strength of a figure that estimates
    nothing."""
    # SFT moves; the other two are stable, so only SFT is crossed.
    revisions = iter(["aaaaaaa", "bbbbbbb", "ccccccc", "ccccccc", "ccccccc", "ccccccc"])
    monkeypatch.setattr(hf, "dataset_revision", lambda dataset, *a, **k: next(revisions))
    monkeypatch.setattr(hf, "num_rows", lambda dataset, *a, **k: 1_000_000)
    counts = {
        "allenai/Dolci-Instruct-SFT": (9999, False),
        "allenai/Dolci-Instruct-DPO": (5, False),
        "allenai/Dolci-Instruct-RL": (0, False),
    }
    monkeypatch.setattr(hf, "search_count", lambda dataset, query, *a, **k: counts[dataset])
    monkeypatch.setattr(
        sys,
        "argv",
        ["trainspotting", "trace", "olmo-3-7b-instruct", "I am ChatGPT, by OpenAI"],
    )
    cli.main()
    out = capsys.readouterr().out

    heading = out.index("## Not ranked: these splits were republished")
    assert out.index("## dpo") < heading < out.index("## sft")
    # DPO's 5 real matches beat SFT's 9999 crossed ones for the recommendation.
    assert "Dolci-Instruct-DPO/viewer" in out
    assert "Dolci-Instruct-SFT/viewer" not in out
    assert "Not compared at all: sft" in out


def _live_search_count(dataset, phrase, split="train"):
    """`hf.search_count`, skipping while the server is still building the index.

    A cold split answers 500 "the dataset index is loading" until the server has
    built a full-text index over it, which on a multi-gigabyte mix outlasts the
    several minutes `search_count` will properly spend waiting. A run that hits
    that has learned nothing about the contract under test, so it skips: the
    alternative is a canary reporting upstream breakage every time the index has
    been evicted.

    Any 5xx skips, not just that one body, because what these tests assert is
    the *answer* the endpoint gives — which columns it reached, whether it
    admitted to a partial index. A server that is not answering has not
    contradicted either claim, and matching on the message text made the test
    fail the first time the wording drifted.

    The private `_get` is deliberate, for its retry budget: two attempts, so a
    cold split costs seconds instead of `search_count`'s full patience being
    spent discovering the same thing. A real run wants that patience; a canary
    does not. Once the pre-flight answers, the index is warm and the
    `search_count` call below is what is actually under test.
    """
    try:
        hf._get(
            "search",
            server_error_retries=2,
            dataset=dataset,
            config="default",
            split=split,
            query=phrase,
            offset=0,
            length=1,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code >= 500:
            # Quoted, so a skip that is really an upstream change is readable
            # in the run output rather than indistinguishable from a cold index.
            pytest.skip(f"{dataset}: /search is not answering — {e.response.text[:200]}")
        raise
    return hf.search_count(dataset, phrase, split=split)


@pytest.mark.live
def test_live_search_counts_a_known_phrase():
    """The upstream canary for `trace`: full-text search still returns a count.

    A first hit against a cold split warms the index and can take minutes, so
    `search_count` waits it out; this test can be correspondingly slow.
    """
    n, partial = _live_search_count("allenai/Dolci-Instruct-SFT", "knowledge cutoff")
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
    n, _ = _live_search_count("allenai/Dolci-Instruct-DPO", "photosynthesis")
    assert n > 0
