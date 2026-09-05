"""Run the production SGLD loop against an exactly solvable Gaussian posterior.

    python -m trainspotting.bif_validation --output results/validation/standalone-text-sensitivity.json

CPU only. The fixed settings and tolerances below are part of the experiment.
The result records failures as well as successes. It does not validate an LLM.
"""

import argparse
import json
import platform
from pathlib import Path

from . import bif


SETTINGS = dict(device="cpu", chains=4, draws=2000, burn_in=1000, every=5,
                lr=0.02, nbeta=2.0, gamma=2.0, batch=1, eval_batch=4, seed=11)
TOLERANCES = dict(covariance_absolute_error=0.04, mean_absolute_error=0.05,
                  variance_absolute_error=0.035, derivative_absolute_error=0.08)
CENTERS = [0.0, -1.0, 1.0]


def run_gaussian(settings=None, perturbation=0.0):
    import torch

    settings = {**SETTINGS, **(settings or {})}
    model = torch.nn.Module()
    model.w = torch.nn.Parameter(torch.tensor(0.0))
    candidates = [{"center": a, "localizer": i == 0} for i, a in enumerate(CENTERS)]
    query = {"center": 1.0, "localizer": False}

    def losses(model, chunk, device):
        # Only candidate zero defines L. The other two are scored observables.
        # Perturb L by delta * loss_plus without renormalizing its weights.
        return torch.stack([0.5 * (model.w - e["center"]) ** 2
                            + (perturbation * 0.5 * (model.w - 1) ** 2 if e["localizer"] else 0)
                            for e in chunk])

    run = bif.sample(model, candidates, query, localize=[0], loss_fn=losses, **settings)
    assert float(model.w.detach()) == 0.0, "the sampler must restore its origin"
    stats = bif.influence(run["losses"])
    weights = [(d[2] - d[3]) / 2 for c in run["losses"] for d in c]
    # Precision = gamma + nbeta * (1 + delta); the mean shifts toward +1.
    precision = settings["gamma"] + settings["nbeta"] * (1 + perturbation)
    v = 1 / precision
    mu = settings["nbeta"] * perturbation / precision
    exact = [0.5 * v * v + (mu - 1) * (mu - a) * v for a in CENTERS]
    exact[0] += perturbation * exact[2]
    return {"settings": settings, "perturbation": perturbation,
            "exact_mean": mu, "exact_variance": v, "exact_covariances": exact,
            "sample_mean": bif._mean(weights), "sample_variance": bif._cov(weights, weights),
            "covariances": stats,
            "query_mean": bif._mean(d[0] for c in run["losses"] for d in c),
            "diagnostics": bif.diagnostics(run["losses"], [0]), "draws": run["losses"]}


def validate():
    import torch

    # Extend burn-in and duration independently, change seed, and halve the
    # discretization step while preserving the time between retained draws.
    variants = {"baseline": {}, "longer_burn_in": {"burn_in": 2000},
                "longer_sampling": {"draws": 4000}, "independent_seed": {"seed": 23},
                "smaller_step": {"lr": 0.01, "burn_in": 2000, "every": 10}}
    runs = {}
    for name, settings in variants.items():
        print(f"Validating {name}...", flush=True)
        run = run_gaussian(settings)
        run["checks"] = {
            "mean": abs(run["sample_mean"] - run["exact_mean"]) < TOLERANCES["mean_absolute_error"],
            "variance": abs(run["sample_variance"] - run["exact_variance"]) < TOLERANCES["variance_absolute_error"],
            "covariance": all(abs(s["cov"] - e) < TOLERANCES["covariance_absolute_error"]
                              for s, e in zip(run["covariances"], run["exact_covariances"])),
            "signs": run["covariances"][1]["cov"] < 0 < run["covariances"][2]["cov"],
            "sampling_screens": run["diagnostics"]["status"] == "checks_passed",
        }
        runs[name] = run
    delta = 0.02
    print("Validating controlled reweighting...", flush=True)
    # Common random numbers reduce finite-difference variance; independent-seed
    # stability is tested separately above. The chains themselves are distinct.
    minus, plus = run_gaussian(perturbation=-delta), run_gaussian(perturbation=delta)
    observed = (plus["query_mean"] - minus["query_mean"]) / (2 * delta)
    predicted = -SETTINGS["nbeta"] * runs["baseline"]["covariances"][2]["cov"]
    exact = -SETTINGS["nbeta"] * runs["baseline"]["exact_covariances"][2]
    intervention = {"delta": delta, "predicted_derivative": predicted,
                    "sampled_finite_difference": observed, "exact_derivative": exact,
                    "passed": abs(observed - exact) < TOLERANCES["derivative_absolute_error"]
                    and abs(observed - predicted) < TOLERANCES["derivative_absolute_error"],
                    "minus": minus, "plus": plus}
    return {"experiment": "Exact posterior and controlled reweighting validation",
            "scope": "Gaussian sampler validation only; language-model attribution remains unvalidated",
            "python": platform.python_version(), "torch": str(torch.__version__),
            "tolerances": TOLERANCES, "runs": runs, "intervention": intervention,
            "passed": all(all(r["checks"].values()) for r in runs.values()) and intervention["passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Validation {'passed' if result['passed'] else 'failed'}: {args.output}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
