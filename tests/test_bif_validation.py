"""Check the committed validation claims against their actual loss draws."""

import json
from pathlib import Path

import pytest

from trainspotting import bif


def test_exact_posterior_validation_matches_its_saved_draws():
    path = Path(__file__).resolve().parents[1] / "results/validation/standalone-text-sensitivity.json"
    res = json.loads(path.read_text())
    assert res["passed"]
    for run in res["runs"].values():
        stats = bif.influence(run["draws"])
        assert [s["cov"] for s in stats] == pytest.approx([s["cov"] for s in run["covariances"]])
        # For w ~ N(0, 1/4), Cov((w-1)^2/2, (w-a)^2/2) = 1/32 + a/4.
        assert run["exact_covariances"] == [1 / 32, -7 / 32, 9 / 32]
        assert all(abs(s["cov"] - exact) < res["tolerances"]["covariance_absolute_error"]
                   for s, exact in zip(stats, run["exact_covariances"]))
        weights = [(d[2] - d[3]) / 2 for c in run["draws"] for d in c]
        assert bif._mean(weights) == pytest.approx(run["sample_mean"])
        assert bif._cov(weights, weights) == pytest.approx(run["sample_variance"])
        assert abs(bif._mean(weights)) < res["tolerances"]["mean_absolute_error"]
        assert abs(bif._cov(weights, weights) - 0.25) < res["tolerances"]["variance_absolute_error"]
        assert bif.diagnostics(run["draws"], [0])["status"] == "checks_passed"
    intervention = res["intervention"]
    expectations = [bif._mean(d[0] for c in intervention[side]["draws"] for d in c)
                    for side in ("minus", "plus")]
    derivative = (expectations[1] - expectations[0]) / (2 * intervention["delta"])
    predicted = -2 * bif.influence(res["runs"]["baseline"]["draws"])[2]["cov"]
    assert derivative == pytest.approx(intervention["sampled_finite_difference"])
    assert predicted == pytest.approx(intervention["predicted_derivative"])
    assert abs(derivative + 9 / 16) < res["tolerances"]["derivative_absolute_error"]
    assert abs(derivative - predicted) < res["tolerances"]["derivative_absolute_error"]
