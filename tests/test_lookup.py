"""How many documents a lookup asks the index for, and what it claims about them.

The API is a trivial POST; the part that goes wrong locally is the call
arithmetic around it. Two mistakes are symmetrical and both silent: asking for
more documents than there are occurrences (the server pads with repeats, so a
two-occurrence phrase comes back ten times) and asking for more than the caller
wanted (`--docs 11` spent two full ten-document calls and returned twenty). The
second is worse than a broken promise, because `drawn` is the denominator every
share on the case-study page is computed over.
"""

import pytest

from trainspotting import lookup


@pytest.fixture
def calls(monkeypatch):
    """Record every maxnum the sampler asks for, and answer with that many
    distinct documents — the index pads to maxnum, so a real reply is never
    short unless the index is."""
    seen = []

    def fake_post(payload):
        seen.append(payload["maxnum"])
        return {
            "documents": [
                {"doc_ix": len(seen) * 100 + i, "spans": [["text", None]], "doc_len": 10}
                for i in range(payload["maxnum"])
            ]
        }

    monkeypatch.setattr(lookup, "_post", fake_post)
    return seen


def test_a_request_over_the_cap_asks_for_the_remainder_not_another_full_call(calls):
    out = lookup.sample_documents("idx", "q", occurrences=1000, want=11)

    assert calls == [10, 1]
    assert out["drawn"] == 11
    assert not out["exhaustive"]


def test_a_request_under_the_cap_is_one_call_for_exactly_what_was_asked(calls):
    out = lookup.sample_documents("idx", "q", occurrences=1000, want=4)

    assert calls == [4]
    assert out["drawn"] == 4


def test_never_asks_for_more_than_the_occurrences_that_exist(calls):
    """The server samples with replacement and pads, so asking for ten of a
    two-occurrence phrase returns the same two documents five times over."""
    out = lookup.sample_documents("idx", "q", occurrences=2, want=10)

    assert calls == [2]
    assert out["exhaustive"]


def test_exhaustive_needs_the_caller_to_have_asked_for_all_of_them(calls):
    """Three of eight occurrences is a sample, not a census, even though one
    call could have seen all eight."""
    out = lookup.sample_documents("idx", "q", occurrences=8, want=3)

    assert calls == [3]
    assert not out["exhaustive"]
    assert lookup.sample_documents("idx", "q", occurrences=8, want=8)["exhaustive"]


def test_a_phrase_with_no_occurrences_costs_no_call(calls):
    out = lookup.sample_documents("idx", "q", occurrences=0, want=5)

    assert calls == []
    assert out["drawn"] == 0 and out["documents"] == []


def test_exhaustive_answers_to_the_reply_not_the_request(monkeypatch):
    """A count at or under the cap asks for every occurrence, but the index can
    still answer short. Computing the flag before looking at the reply let a
    phrase counted at eight occurrences come back with three documents — or
    none — while the CLI and the site called the list complete."""
    monkeypatch.setattr(
        lookup,
        "_post",
        lambda payload: {"documents": [{"doc_ix": 1, "spans": [["t", None]], "doc_len": 9}]},
    )

    out = lookup.sample_documents("idx", "q", occurrences=8, want=10)

    assert out["drawn"] == 1
    assert not out["exhaustive"]


def test_a_short_reply_ends_the_draw_instead_of_spinning(monkeypatch):
    """An index that answers with fewer documents than asked has no more to
    give. Subtracting what came back rather than what was asked for would keep
    re-requesting the shortfall forever."""
    seen = []

    def fake_post(payload):
        seen.append(payload["maxnum"])
        return {"documents": []}

    monkeypatch.setattr(lookup, "_post", fake_post)

    out = lookup.sample_documents("idx", "q", occurrences=1000, want=50)

    assert seen == [10]
    assert out["drawn"] == 0


def test_repeats_across_calls_are_counted_once_as_documents_and_kept_in_drawn(monkeypatch):
    """Every call redraws from the whole occurrence list, so the same document
    recurs. The copy is what a reader opens; the count is what any share over
    the sample is weighted by."""
    monkeypatch.setattr(
        lookup,
        "_post",
        lambda payload: {
            "documents": [
                {"doc_ix": 7, "spans": [["text", None]], "doc_len": 10}
                for _ in range(payload["maxnum"])
            ]
        },
    )

    out = lookup.sample_documents("idx", "q", occurrences=1000, want=12)

    assert out["drawn"] == 12
    assert len(out["documents"]) == 1
    assert out["documents"][0]["occurrences_drawn"] == 12


def test_a_negative_document_count_is_a_usage_error_but_zero_is_not():
    """`--docs 0` asks for counts without documents, which is a real choice.
    `--docs -1` reached the sampler as a negative budget, skipped retrieval, and
    reported "0 documents from 0 draws" for a phrase with thousands of copies."""
    import argparse

    from trainspotting.cli import _count_int

    assert _count_int("0") == 0
    assert _count_int("12") == 12
    with pytest.raises(argparse.ArgumentTypeError):
        _count_int("-1")
