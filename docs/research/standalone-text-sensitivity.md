# Standalone text sensitivity: validation and remaining limits

The production sampler recovers an exactly known Gaussian posterior and predicts
the change caused by reweighting an example. The existing Pythia demonstration
fails the sampling screens. This supports keeping the command as a restricted
research experiment; it does not validate language-model training attribution.

## Controlled experiment

The scalar parameter starts at zero. With mean loss `L(w) = w²/2`, inverse
temperature `nβ = 2`, and localization `γ = 2`, the target posterior is exactly
`Normal(0, 1/4)`. Query loss is `(w−1)²/2`. Three candidate losses are `(w−a)²/2`
for `a = 0, −1, +1`; only the first defines the unperturbed sampling objective.
Their exact covariances with the query are `0.03125, −0.21875, 0.28125`.

This uses the same stochastic gradient Langevin dynamics (SGLD) loop, noise
generation, prior update, and loss-draw collection as the public command, with a quadratic loss adapter. It exercises
parameter restoration too. It does not exercise a language model's loss landscape.

All five configurations passed the prespecified tolerances and sampling screens:

| Configuration | Parameter mean (exact 0) | Variance (exact 0.25) | Negative covariance (exact −0.21875) | Positive covariance (exact 0.28125) |
|---|---:|---:|---:|---:|
| Baseline | 0.00228 | 0.25744 | −0.22112 | 0.28940 |
| Longer burn-in | 0.00519 | 0.25640 | −0.22027 | 0.28734 |
| Longer sampling | −0.01071 | 0.25020 | −0.21823 | 0.28968 |
| Independent seed | 0.01294 | 0.24768 | −0.21628 | 0.27669 |
| Smaller step | −0.01724 | 0.24334 | −0.21318 | 0.28061 |

The baseline uses four chains, 1,000 burn-in steps, 2,000 retained draws per
chain, one retained draw every five steps, step size 0.02, and seed 11. Variants
independently double burn-in, double retained draws, change seed to 23, or halve
the step size while doubling burn-in and the interval between retained draws.
Absolute error tolerances are 0.05 for the mean, 0.035 for variance, 0.04 for
covariance, and 0.08 for the intervention derivative. These settings were fixed
before running the experiment; the runner records failures and exits nonzero.

For the intervention, replace `L` with `L + δ·(w−1)²/2` at `δ = ±0.02`, without
renormalizing the objective. The exact posterior has mean `2δ/(4+2δ)` and
variance `1/(4+2δ)`. The query-loss derivative at zero is exactly **−0.5625**.
The covariance estimate predicts **−0.57880**; the finite difference between
separately sampled perturbed posteriors is **−0.56119**. The perturbations use
common random numbers to reduce finite-difference noise; independent-seed
stability is evaluated separately above.

## Language-model result

The historical Pythia-70m run uses 200 standalone documents, four chains, 30
burn-in steps and 50 retained draws per chain. It is **inconclusive**:

- Every chain's mean candidate loss continues rising across retained draws;
  correlation with time is 0.996–0.999.
- Removing a linear time trend removes 88–99% of each chain's average
  query–candidate covariance. This is a diagnostic, not a corrected estimator.
- All 200 descriptive covariances are positive. The smallest positive values
  are not evidence of negative influence.

The original draws are preserved. Stored ranks are cleared, and both the command
and the report withhold rankings when raw-draw diagnostics fail. Passing the
Gaussian experiment does not rehabilitate this language-model run.

## Reproduce and assess

Install the optional dependencies and run:

```bash
pip install -e '.[bif,dev]'
python -m trainspotting.bif_validation \
  --output results/validation/standalone-text-sensitivity.json
pytest -q tests/test_bif.py tests/test_bif_validation.py
```

[The committed validation artifact](../../results/validation/standalone-text-sensitivity.json)
contains settings, tolerances, diagnostics, all loss draws, and the controlled
intervention. Tests recompute its claims from those draws. A dedicated CPU CI
job reruns the numerical experiment and uploads its result even on failure.

The screens use ordinary split R-hat, a batch-means effective sample size, and
time correlation on losses and covariance products. They are conservative
heuristics, not proof of convergence in weight space or calibrated error bars.
The current scope is standalone corpus text with Pythia-70m in float32. Chat,
preference optimization, other checkpoints, packed training-sequence attribution,
and claims about which data caused a behavior remain deferred. A language-model
result needs stable sampling across settings and a controlled data intervention
before that scope should expand.

The mathematical starting point is [Kreer et al., Bayesian Influence Functions
for Hessian-Free Data Attribution](https://arxiv.org/abs/2509.26544), particularly
the covariance identity and the discussion of systematic sampling bias.
