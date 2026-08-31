"""The clustered confidence interval behind every pretraining rate on the site.

`_cluster_wilson` is where "12% of sampled documents match" turns into a range,
and the interval is only honest if the design effect is right: documents from
one shard share a topic, so a binomial interval over the document count is too
narrow. These pin the arithmetic and both degenerate branches.
"""

import pytest

from trainspotting.cli import _cluster_wilson, _wilson


def docs(spec):
    """spec: {shard: (matches, total)}."""
    return [
        {"shard": shard, "match": i < k}
        for shard, (k, m) in spec.items()
        for i in range(m)
    ]


def test_design_effect_by_hand():
    """Two shards of five, one unanimously matching and one unanimously not.

        M = 10, C = 2, p = 0.5
        Σ (y_c - p·m_c)² = (5 - 2.5)² + (0 - 2.5)² = 12.5
        Var(p) = 2 / (1 · 100) · 12.5 = 0.25
        binomial = 0.5 · 0.5 / 10 = 0.025
        deff = 0.25 / 0.025 = 10  ->  n_eff = 10 / 10 = 1

    Ten documents that disagree perfectly along shard lines carry the
    information of one, and Wilson at n=1 spans almost the whole unit interval.
    """
    lo, hi, n_eff = _cluster_wilson(docs({"a": (5, 5), "b": (0, 5)}))

    assert n_eff == pytest.approx(1.0)
    # _wilson(0.5, 1): denom 4.8416, centre 0.5, half 1.96·√1.2104 / 4.8416
    assert lo == pytest.approx(0.05462, abs=5e-5)
    assert hi == pytest.approx(0.94538, abs=5e-5)


def test_one_document_per_shard_is_almost_the_ordinary_interval():
    """The docstring's claim: with every cluster of size one the design effect is
    C/(C-1), so n_eff is exactly n-1 and the interval is Wilson's."""
    records = [{"shard": f"s{i}", "match": i < 3} for i in range(10)]

    lo, hi, n_eff = _cluster_wilson(records)

    assert n_eff == pytest.approx(9.0)
    assert (lo, hi) == _wilson(0.3 * 9, 9)


def test_agreeing_clusters_do_not_narrow_the_interval():
    """Clusters that each split the same way have zero between-cluster variance,
    which is a deff below 1. The floor keeps that from *shrinking* the interval
    below the binomial one."""
    lo, hi, n_eff = _cluster_wilson(docs({"a": (1, 2), "b": (1, 2)}))

    assert n_eff == pytest.approx(4.0)
    assert (lo, hi) == _wilson(2, 4)


def test_single_cluster_falls_back_to_the_cluster_count():
    """Ten documents from one shard: the design effect is unestimable, so they
    count as one observation rather than ten."""
    lo, hi, n_eff = _cluster_wilson(docs({"only": (5, 10)}))

    assert n_eff == pytest.approx(1.0)
    assert (lo, hi) == _wilson(0.5, 1)


@pytest.mark.parametrize("matches", [0, 2])
def test_unanimous_outcome_falls_back_to_the_cluster_count(matches):
    """"No matches at all" is a likely answer to a pointed question, and it is
    exactly where a binomial interval over 16 documents would overclaim. Eight
    shards of two documents is n_eff = 8, not 16."""
    spec = {f"s{i}": (matches, 2) for i in range(8)}

    lo, hi, n_eff = _cluster_wilson(docs(spec))

    assert n_eff == pytest.approx(8.0)
    assert (lo, hi) == _wilson(matches / 2 * 8, 8)
    assert n_eff < 16


def test_no_records():
    assert _cluster_wilson([]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("shard", [None, ""])
def test_missing_shard_labels_collapse_into_one_cluster(shard):
    """An unlabelled sample is treated as a single cluster — the cautious end,
    not silently independent."""
    records = [{"shard": shard, "match": i < 3} for i in range(10)]
    assert _cluster_wilson(records)[2] == pytest.approx(1.0)


def test_absent_shard_key_collapses_into_one_cluster():
    records = [{"match": i < 3} for i in range(10)]
    assert _cluster_wilson(records)[2] == pytest.approx(1.0)


def test_clusters_by_the_requested_key():
    records = [{"topic": f"t{i}", "shard": "same", "match": i < 3} for i in range(10)]

    assert _cluster_wilson(records, key="topic")[2] == pytest.approx(9.0)
    assert _cluster_wilson(records, key="shard")[2] == pytest.approx(1.0)


def test_wider_than_the_uncorrected_interval_when_shards_disagree():
    """The whole point: clustering can only widen. A binomial interval over the
    same counts would be narrower and wrong."""
    records = docs({"a": (4, 5), "b": (1, 5), "c": (5, 5), "d": (0, 5)})

    lo, hi, n_eff = _cluster_wilson(records)
    plain_lo, plain_hi = _wilson(10, 20)

    assert n_eff < 20
    assert hi - lo > plain_hi - plain_lo


def test_wilson_stays_inside_the_unit_interval():
    """Wilson is kept over a normal interval because several of these rates sit
    at 0 or 1, where a normal interval leaves the unit interval."""
    assert _wilson(0, 5)[0] == 0.0
    assert _wilson(5, 5)[1] == 1.0
    assert _wilson(0, 0) == (0.0, 0.0)
