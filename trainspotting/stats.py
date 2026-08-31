"""Confidence intervals for a rate measured on a sample.

Two of them, because the two halves of this tool sample differently. A
post-training run draws rows independently, so the ordinary Wilson interval
applies. A pretraining run draws shards and reads documents out of each, so its
observations are clustered and the binomial interval would be too narrow.

Kept in one module because the CLI computes an interval and the site displays
one, and a second implementation of either is a number that can silently
disagree with the file it came from.
"""

import math


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. k and n may be non-integer effective counts."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cluster_wilson(records: list[dict], key: str = "shard") -> tuple[float, float, float]:
    """Wilson interval widened by the design effect of clustering by `key`.

    Documents drawn from one shard share a topic cluster, so they are not
    independent observations and a binomial interval over the document count is
    too narrow. Rescaling the match count to the number of distinct shards, which
    is what this used to do, is not an interval over shards either: a shard
    contributing five matches and one contributing a single non-match would round
    to "two successes out of two clusters", hiding the disagreement between them.

    So take the design effect properly, from the Taylor-linearised variance of the
    ratio estimator over clusters:

        Var(p) = C / ((C - 1) · M²) · Σ (y_c - p·m_c)²

    where cluster c holds m_c documents of which y_c match, and M = Σ m_c. Divide
    by the binomial variance to get the design effect, and evaluate Wilson at the
    effective sample size n/deff. Wilson is kept rather than a normal interval
    because it still behaves at rates near 0 and 1, which several of these are.

    Returns (lo, hi, n_effective). With one document per shard every cluster has
    size one, the design effect is C/(C-1) ≈ 1, and this is the ordinary interval.
    """
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0.0
    clusters: dict[str, list[int]] = {}
    for r in records:
        clusters.setdefault(r.get(key) or "", []).append(1 if r["match"] else 0)
    C = len(clusters)
    p = sum(r["match"] for r in records) / n
    if C < 2 or p in (0.0, 1.0):
        # The design effect is unestimable here, not 1. With a single cluster
        # there is nothing to compare it against; with a unanimous outcome the
        # observed between-cluster variance is zero, which is 0/0 rather than
        # evidence of independence. Either way, falling back to the document
        # count would hand a clustered run the narrow interval for n independent
        # observations — and "no matches at all" is a likely answer to a pointed
        # question, so that branch fires exactly when the number matters. Use the
        # cluster count instead: assume documents sharing a shard told us one
        # thing, not m_c things. At one document per shard C is n and nothing
        # changes.
        return (*wilson(p * C, C), float(C))
    ss = sum((sum(ys) - p * len(ys)) ** 2 for ys in clusters.values())
    var_cluster = C / ((C - 1) * n**2) * ss
    var_binomial = p * (1 - p) / n
    deff = max(1.0, var_cluster / var_binomial) if var_binomial else 1.0
    n_eff = max(1.0, n / deff)
    return (*wilson(p * n_eff, n_eff), n_eff)
