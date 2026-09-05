"""Conservative finite-sample screens, not a certificate of posterior convergence.

Ordinary split R-hat checks agreement across chains and their halves. A batch
means effective sample size is a rough autocorrelation screen. Both are applied
to the query, the localized mean, every candidate, and the centered products
whose averages form covariances. These are deliberately described as screens:
loss diagnostics cannot establish mixing in all weight-space directions.
"""

import math


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def screen(chains):
    half = len(chains[0]) // 2
    split = [part for c in chains for part in (c[:half], c[-half:])]
    within = mean([variance(c) for c in split])
    if within <= 0:
        return {"split_rhat": None, "batch_ess": 0.0, "max_time_correlation": None}
    between = half * variance([mean(c) for c in split])
    rhat = math.sqrt(((half - 1) * within + between) / (half * within))
    width = max(2, int(math.sqrt(len(chains[0]))))
    blocks = [mean(c[i:i + width]) for c in chains
              for i in range(0, len(c) - width + 1, width)]
    flat = [x for c in chains for x in c]
    mc_variance = variance(blocks) / len(blocks)
    ess = min(len(flat), variance(flat) / mc_variance) if mc_variance > 0 else 0.0
    correlations = []
    t = list(range(len(chains[0])))
    mt = mean(t)
    for c in chains:
        mc = mean(c)
        denom = math.sqrt(sum((x - mc) ** 2 for x in c) * sum((x - mt) ** 2 for x in t))
        correlations.append(abs(sum((x - mc) * (i - mt) for i, x in enumerate(c)) / denom)
                            if denom else 1.0)
    return {"split_rhat": rhat, "batch_ess": ess,
            "max_time_correlation": max(correlations)}


def diagnostics(draws, localized=None):
    """Screen raw loss draws; missing, short or degenerate runs are inconclusive.

    Thresholds are explicit screening conventions, not significance tests.
    """
    limits = {"min_chains": 4, "min_draws": 100, "max_split_rhat": 1.05,
              "min_batch_ess": 400, "max_time_correlation": 0.5}
    result = {"version": 1, "status": "inconclusive", "thresholds": limits,
              "reasons": [], "observables": []}
    reasons = result["reasons"]
    if not draws or not draws[0] or len(draws[0][0]) < 2:
        reasons.append("raw query and candidate draws are missing")
        return result
    n, cols = len(draws[0]), len(draws[0][0])
    if any(len(c) != n or any(len(d) != cols for d in c) for c in draws):
        reasons.append("raw draw dimensions are inconsistent")
        return result
    if not all(math.isfinite(v) for c in draws for d in c for v in d):
        reasons.append("raw draws contain non-finite losses")
        return result
    if len(draws) < limits["min_chains"] or n < limits["min_draws"]:
        reasons.append("at least 4 chains and 100 retained draws per chain are needed for the screens")
    if n < 8:
        return result
    indices = list(range(1, cols)) if localized is None else [i + 1 for i in localized]
    if not indices or any(i < 1 or i >= cols for i in indices):
        reasons.append("localized candidate indices are invalid")
        return result
    query = [[d[0] for d in c] for c in draws]
    observables = [("query", query), ("localized_mean", [[mean([d[i] for i in indices])
                                                         for d in c] for c in draws])]
    for i in range(1, cols):
        candidate = [[d[i] for d in c] for c in draws]
        observables.append((f"candidate_{i - 1}", candidate))
        products = []
        for qs, xs in zip(query, candidate):
            mq, mx = mean(qs), mean(xs)
            products.append([(q - mq) * (x - mx) for q, x in zip(qs, xs)])
        observables.append((f"covariance_{i - 1}", products))
    for name, chains in observables:
        result["observables"].append({"name": name, **screen(chains)})
    metrics = result["observables"]
    for label, predicate in (
        ("degenerate or disagreeing chain halves (split R-hat > 1.05)",
         lambda x: x["split_rhat"] is None or x["split_rhat"] > limits["max_split_rhat"]),
        ("insufficient effective draws (batch means estimate < 400)",
         lambda x: x["batch_ess"] < limits["min_batch_ess"]),
        ("strong retained-draw time trend (absolute correlation > 0.5)",
         lambda x: x["max_time_correlation"] is None
         or x["max_time_correlation"] > limits["max_time_correlation"]),
    ):
        failed = [m["name"] for m in metrics if predicate(m)]
        if failed:
            reasons.append(f"{label}: {', '.join(failed[:4])}"
                           + (f" and {len(failed) - 4} more" if len(failed) > 4 else ""))
    if not reasons:
        result["status"] = "checks_passed"
    return result
