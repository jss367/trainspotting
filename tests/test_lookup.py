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


@pytest.mark.parametrize(
    "url",
    [
        "https://marginalrevolution.com/post",
        "https://marginalrevolution.com:443/post",       # explicit port
        "http://marginalrevolution.com:8080/post",
        "https://user:pw@marginalrevolution.com/post",   # userinfo
        "https://WWW.MarginalRevolution.COM/post",       # case and www.
    ],
    ids=["plain", "https-port", "http-port", "userinfo", "shouty-www"],
)
def test_a_site_is_the_same_site_however_the_url_is_written(url):
    """`casestudy.run` compares this domain against the subject's site by
    string equality, and that comparison decides `on_subject_site` — the
    study's headline. A port or a login in the URL must not make a page
    somebody else's."""
    rec = lookup.normalize(
        {"doc_ix": 1, "spans": [["t", None]], "metadata": json.dumps({"metadata": {"metadata": {"url": url}}})}
    )

    assert rec["domain"] == "marginalrevolution.com"


@pytest.mark.parametrize(
    "url",
    [
        "http://[not-an-address/x",     # urlsplit raises ValueError on the bracket
        "https://[::1",
        12345,                          # a corpus that recorded an id, not a URL
        {"href": "http://example.com"},
        [],
    ],
    ids=["bad-bracket", "unclosed-v6", "integer", "dict", "list"],
)
def test_a_url_that_will_not_parse_costs_one_document_its_domain(url):
    """Not the whole sample. Five corpora with disagreeing subsets feed this and
    the metadata is whatever a crawl recorded, so a URL that raises on parse is
    a document with no readable host — the same answer as a document that
    recorded none. Letting it raise discarded every document drawn with it."""
    rec = lookup.normalize(
        {"doc_ix": 1, "spans": [["t", None]],
         "metadata": json.dumps({"metadata": {"metadata": {"url": url}}})}
    )

    assert rec["domain"] is None
    # A string that will not parse is still what the corpus recorded and is
    # worth showing; only its host is unreadable. Anything that is not a string
    # is not a URL at all, and is dropped — `_identity` puts this value in a
    # dict key, and a dict cannot be one.
    assert rec["url"] is None or isinstance(rec["url"], str)


@pytest.mark.parametrize(
    "score",
    [None, "n/a", {}, [], float("nan"), float("inf")],
    ids=["null", "text", "dict", "list", "nan", "inf"],
)
def test_a_quality_score_that_will_not_convert_is_unavailable(score):
    """Same rule as the URL, one field over: unusable is absent, not fatal.

    NaN and the infinities convert but do not serialize — `json` writes them as
    bare tokens no browser will parse — so a score the study cannot publish is
    also unavailable rather than stored.
    """
    attrs = {"dolma17_hq": [[0, 100, score]]}
    rec = lookup.normalize(
        {"doc_ix": 1, "spans": [["t", None]],
         "metadata": json.dumps({"metadata": {"attributes": attrs}})}
    )

    assert rec["quality"] is None


def test_the_excerpt_is_the_window_around_the_match():
    """The spans are the window infini-gram centred on the hit; `text` is the
    document from its beginning. Reading `text` first and cutting it to fit
    meant a long document's excerpt was its opening paragraphs, which need not
    contain the query at all — and a study whose whole point is "read the
    documents and see what the number meant" cannot show a reader an excerpt
    that omits what matched."""
    rec = lookup.normalize({
        "doc_ix": 1,
        "text": "the opening paragraphs, which do not say it",
        "spans": [["…before ", None], ["the needle", "q"], [" after…", None]],
    })

    assert rec["excerpt"] == "…before the needle after…"


def test_text_is_still_read_when_there_are_no_spans():
    rec = lookup.normalize({"doc_ix": 1, "text": "all there is", "spans": []})

    assert rec["excerpt"] == "all there is"


@pytest.fixture
def ranked(monkeypatch):
    """A two-shard index holding five occurrences of a phrase across three
    documents, answered by rank the way `find` + `get_doc_by_rank` do. Records
    every rank asked for so a test can check the walk covered each one once."""
    asked = []
    # rank -> (path, doc_ix): one document seen at three ranks, two seen once.
    # A document lives in one suffix-array shard, so all of its ranks are in
    # that shard's range: the repeated one is shard 0's, the two singles shard 1's.
    where = {
        10: ("a.json.gz", 1), 11: ("a.json.gz", 1), 12: ("a.json.gz", 1),
        50: ("b.json.gz", 2), 51: ("c.json.gz", 3),
    }

    def fake_post(payload):
        if payload["query_type"] == "find":
            return {"cnt": 5, "segment_by_shard": [[10, 13], [50, 52]]}
        assert payload["query_type"] == "get_doc_by_rank"
        asked.append((payload["s"], payload["rank"]))
        path, ix = where[payload["rank"]]
        return {
            "doc_ix": ix, "doc_len": 10, "spans": [["text", None]],
            "metadata": json.dumps({"path": path}),
        }

    monkeypatch.setattr(lookup, "_post", fake_post)
    return asked


def test_every_document_walks_each_rank_once_and_is_exhaustive(ranked):
    out = lookup.every_document("idx", "q")
    assert ranked == [(0, 10), (0, 11), (0, 12), (1, 50), (1, 51)]
    assert out["drawn"] == 5
    assert out["exhaustive"]
    # Three documents, and the repeated one carries its real occurrence count.
    assert [(d["shard"], d["occurrences_drawn"]) for d in out["documents"]] == [
        ("a.json.gz", 3), ("b.json.gz", 1), ("c.json.gz", 1),
    ]


def test_every_document_is_not_exhaustive_when_a_rank_returns_nothing(ranked, monkeypatch):
    real = lookup._post

    def flaky(payload):
        if payload.get("rank") == 51:
            return {"blocked": True}
        return real(payload)

    monkeypatch.setattr(lookup, "_post", flaky)
    out = lookup.every_document("idx", "q")
    assert out["drawn"] == 4
    assert not out["exhaustive"]


def test_probe_all_uses_the_rank_walk(ranked, monkeypatch):
    monkeypatch.setattr(lookup, "count", lambda i, q: {"occurrences": 5, "approx": False, "tokens": []})
    out = lookup.probe("idx", "q", docs="all")
    assert out["exhaustive"] and out["drawn"] == 5 and len(ranked) == 5


def test_probe_all_with_no_occurrences_costs_no_call(ranked, monkeypatch):
    monkeypatch.setattr(lookup, "count", lambda i, q: {"occurrences": 0, "approx": False, "tokens": []})
    lookup.probe("idx", "q", docs="all")
    assert ranked == []


def _ranked_without_metadata(segments, monkeypatch):
    """Every rank resolves to `doc_ix` 7 with metadata that will not decode, so
    the record has neither a path nor a URL — the shape `_identity` refuses to
    merge on. The rank walk knows the shard and does not need to refuse."""

    def fake_post(payload):
        if payload["query_type"] == "find":
            return {"cnt": sum(hi - lo for lo, hi in segments), "segment_by_shard": segments}
        return {"doc_ix": 7, "doc_len": 10, "spans": [["text", None]], "metadata": "{broken"}

    monkeypatch.setattr(lookup, "_post", fake_post)


def test_every_document_merges_repeats_in_one_shard_even_with_no_metadata(monkeypatch):
    """Two ranks in the same shard landing on the same `doc_ix` are one document
    seen twice, whatever its metadata says — unlike `sample_documents`, which has
    to leave two such hits apart because it does not know their shard."""
    _ranked_without_metadata([[10, 12]], monkeypatch)

    out = lookup.every_document("idx", "q")

    assert out["drawn"] == 2 and out["exhaustive"]
    assert len(out["documents"]) == 1
    assert out["documents"][0]["occurrences_drawn"] == 2


def test_every_document_keeps_the_same_doc_ix_in_different_shards_apart(monkeypatch):
    """`doc_ix` is shard-local: the same number in two shards is two files."""
    _ranked_without_metadata([[10, 11], [50, 51]], monkeypatch)

    out = lookup.every_document("idx", "q")

    assert out["drawn"] == 2 and out["exhaustive"]
    assert len(out["documents"]) == 2
    assert all(d["occurrences_drawn"] == 1 for d in out["documents"])
