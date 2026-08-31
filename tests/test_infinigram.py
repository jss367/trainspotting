"""The pure half of the infini-gram search path.

The API calls themselves are trivial POSTs; what can quietly go wrong locally
is the mapping from the API's shapes to what `find` shows — picking ranks
across suffix-array shards, marking matched spans, and digging provenance out
of the per-source metadata shapes. The live test is the canary for the API
contract itself.
"""

import pytest

from trainspotting import infinigram
from trainspotting.infinigram import doc_provenance, snippet, spread_picks


def test_spread_picks_spans_shards():
    """Ten matches over three shards, five picks: evenly spaced global
    positions 0,2,4,6,8 land in every shard, offset by each segment's start."""
    segments = [[100, 104], [200, 203], [300, 303]]  # sizes 4, 3, 3

    picks = spread_picks(segments, 5)

    assert picks == [(0, 100), (0, 102), (1, 200), (1, 202), (2, 301)]


def test_spread_picks_clamps_to_match_count():
    assert spread_picks([[7, 9]], 5) == [(0, 7), (0, 8)]


def test_spread_picks_empty_segments():
    """A shard with no matches contributes nothing; none at all yields nothing."""
    assert spread_picks([[5, 5], [80, 81]], 3) == [(1, 80)]
    assert spread_picks([[5, 5]], 3) == []
    assert spread_picks([], 3) == []


def test_snippet_marks_matched_spans():
    doc = {
        "spans": [
            ["the ", None],
            ["climate change", "0"],
            [" debate", None],
        ]
    }
    assert snippet(doc) == "the «climate change» debate"


def test_provenance_web_crawl_shape():
    """The crawl sources nest WARC metadata two levels down and put the URL in
    several places; the shard path and source tag sit at fixed levels."""
    doc = {
        "metadata": (
            '{"path": "dclm-0977.json.zst", "linenum": 650820, "metadata":'
            ' {"id": "https://example.org/page", "source": "dclm-hero-run",'
            ' "metadata": {"url": "https://example.org/page"}}}'
        )
    }
    assert doc_provenance(doc) == {
        "path": "dclm-0977.json.zst",
        "source": "dclm-hero-run",
        "url": "https://example.org/page",
    }


def test_provenance_survives_junk():
    assert doc_provenance({"metadata": "not json"}) == {}
    assert doc_provenance({"metadata": '"a string"'}) == {}
    assert doc_provenance({}) == {}


@pytest.mark.live
def test_live_find_and_retrieve():
    """The API contract end to end: a phrase that cannot fail to occur, its
    count, and one retrieved document whose snippet contains the marked match."""
    found = infinigram.find(infinigram.DEFAULT_INDEX, "climate change")

    assert found["cnt"] > 1_000_000  # 72.9M when this was written
    assert len(found["segment_by_shard"]) >= 1

    (s, rank), *_ = spread_picks(found["segment_by_shard"], 1)
    doc = infinigram.get_doc(infinigram.DEFAULT_INDEX, "climate change", s, rank, 50)

    assert "«climate change»" in snippet(doc).lower()
