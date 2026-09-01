"""How many documents a lookup asks the index for, and what it claims about them.

The API is a trivial POST; the part that goes wrong locally is the call
arithmetic around it. Two mistakes are symmetrical and both silent: asking for
more documents than there are occurrences (the server pads with repeats, so a
two-occurrence phrase comes back ten times) and asking for more than the caller
wanted (`--docs 11` spent two full ten-document calls and returned twenty). The
second is worse than a broken promise, because `drawn` is the denominator every
share on the case-study page is computed over.
"""

import json

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
                {"doc_ix": 7, "spans": [["text", None]], "doc_len": 10,
                 "metadata": json.dumps({"path": "cc_en_head/cc_en_head-0001.json.gz"})}
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


def test_two_documents_sharing_a_doc_ix_are_not_merged(monkeypatch):
    """`doc_ix` numbers a document inside one suffix-array shard, so it is not
    an identity: merging on it alone would fold two documents into one and file
    both draws under the first one's domain, which `domain_shares` then reports
    as a site with twice the presence it has."""
    hits = [
        {"doc_ix": 7, "spans": [["a", None]], "doc_len": 10,
         "metadata": json.dumps({"path": "cc_en_head/cc_en_head-0001.json.gz"})},
        {"doc_ix": 7, "spans": [["b", None]], "doc_len": 10,
         "metadata": json.dumps({"path": "cc_en_tail/cc_en_tail-0900.json.gz"})},
    ]
    monkeypatch.setattr(lookup, "_post", lambda payload: {"documents": hits})

    out = lookup.sample_documents("idx", "q", occurrences=2, want=2)

    assert len(out["documents"]) == 2
    assert out["drawn"] == 2


def test_the_same_document_drawn_twice_is_counted_once(monkeypatch):
    hit = {"doc_ix": 7, "spans": [["a", None]], "doc_len": 10,
           "metadata": json.dumps({"path": "cc_en_head/cc_en_head-0001.json.gz"})}
    monkeypatch.setattr(lookup, "_post", lambda payload: {"documents": [hit, dict(hit)]})

    out = lookup.sample_documents("idx", "q", occurrences=2, want=2)

    assert len(out["documents"]) == 1
    assert out["documents"][0]["occurrences_drawn"] == 2


@pytest.mark.parametrize(
    ("meta", "where"),
    [
        ({"metadata": {"url": "http://example.com/a"}}, "inner url"),
        ({"metadata": {"WARC-Target-URI": "http://example.com/a"}}, "inner WARC-Target-URI"),
        ({"metadata": {"metadata": {"url": "http://example.com/a"}}}, "deep url"),
        ({"metadata": {"metadata": {"WARC-Target-URI": "http://example.com/a"}}}, "deep WARC-Target-URI"),
        ({"metadata": {"id": "http://example.com/a"}}, "an id that is a URL"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_url_is_found_wherever_the_subset_put_it(meta, where):
    """The subsets disagree about where provenance lives, which is why
    `infinigram.doc_provenance` reads five places. A URL missed here costs the
    document its domain, and `domain_shares` — which the study's headline share
    is computed over — files it under nothing."""
    rec = lookup.normalize({"doc_ix": 1, "spans": [["t", None]], "metadata": json.dumps(meta)})

    assert rec["url"] == "http://example.com/a", f"missed the URL in {where}"
    assert rec["domain"] == "example.com"


def test_a_document_with_no_url_anywhere_reports_none():
    rec = lookup.normalize({"doc_ix": 1, "spans": [["t", None]], "metadata": json.dumps({"metadata": {}})})

    assert rec["url"] is None and rec["domain"] is None


@pytest.mark.parametrize(
    "metadata",
    [
        "{not json at all",
        None,
        "null",
        '"a bare string"',
        "[]",
        {"metadata": {"url": "http://example.com/a"}},   # already decoded
    ],
    ids=["malformed", "missing", "json-null", "bare-string", "list", "already-decoded"],
)
def test_one_unreadable_document_does_not_abort_the_sample(metadata, monkeypatch):
    """Five heterogeneous corpora feed this. A document whose provenance cannot
    be read has no provenance; it is not a reason to discard the other 59."""
    hits = [
        {"doc_ix": 1, "spans": [["a", None]], "doc_len": 10, "metadata": metadata},
        {"doc_ix": 2, "spans": [["b", None]], "doc_len": 10,
         "metadata": json.dumps({"path": "x.json.gz", "metadata": {"url": "http://example.com/b"}})},
    ]
    monkeypatch.setattr(lookup, "_post", lambda payload: {"documents": hits})

    out = lookup.sample_documents("idx", "q", occurrences=2, want=2)

    assert len(out["documents"]) == 2
    assert any(d["url"] == "http://example.com/b" for d in out["documents"])


def test_hits_with_no_identity_at_all_are_not_merged(monkeypatch):
    """A hit whose metadata would not decode has no file and no URL — which is
    exactly what the malformed-metadata fallback produces. Two such hits share
    only a shard-local index, which is not evidence they are the same document,
    and merging them would attribute both draws to one of their domains."""
    monkeypatch.setattr(
        lookup,
        "_post",
        lambda payload: {
            "documents": [
                {"doc_ix": 7, "spans": [["a", None]], "doc_len": 10, "metadata": "{broken"},
                {"doc_ix": 7, "spans": [["b", None]], "doc_len": 10, "metadata": "{also broken"},
            ]
        },
    )

    out = lookup.sample_documents("idx", "q", occurrences=2, want=2)

    assert len(out["documents"]) == 2
    assert all(d["occurrences_drawn"] == 1 for d in out["documents"])


def test_a_url_identifies_a_document_when_the_file_does_not(monkeypatch):
    """Provenance that names no shard file can still name the page."""
    hit = {"doc_ix": 7, "spans": [["a", None]], "doc_len": 10,
           "metadata": json.dumps({"metadata": {"url": "http://example.com/a"}})}
    monkeypatch.setattr(lookup, "_post", lambda payload: {"documents": [hit, dict(hit)]})

    out = lookup.sample_documents("idx", "q", occurrences=2, want=2)

    assert len(out["documents"]) == 1
    assert out["documents"][0]["occurrences_drawn"] == 2
