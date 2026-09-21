# How training changed the model

The site’s model development panel follows every registered training phase.
It keeps training exposure, weight movement, prediction differences, and
published evaluations separate. A missing measurement is never displayed as zero.
Standalone datasets have no model development panel.

## Measurements available in this checkout

- **Pythia 70M, deduplicated:** actual weights and inference at initialization,
  step 1,000, step 10,000, and the final step 143,000. The committed result
  contains all 70,426,624 unique parameters, group-level distances, 18 fixed
  prompts, and before-and-after continuations. These checkpoints cover the
  pretraining boundaries but only three intervals of its training trajectory.
- **OLMo 3 7B Instruct, 7B Think, 32B Think, and OLMo 3.1 32B Instruct:**
  publisher-reported evaluations across instruction tuning, preference
  training, and reinforcement learning. Each artifact identifies its source
  table with an immutable revision and a content hash. These are not results
  from our probe set. No pretraining or midtraining scores are inferred.
- Other phases and models explicitly show the missing measurements. OLMo weight
  comparisons have not been run in this checkout.

## Weight movement

For aligned starting and ending parameter vectors `a` and `b`, containing `N`
unique parameters:

```
net RMS change       = sqrt(sum((b - a)^2) / N)
relative change      = sqrt(sum((b - a)^2) / sum(a^2))
observed movement    = sum(RMS(checkpoint[i + 1] - checkpoint[i]))
```

The relative value has no definition when the starting vector has zero norm;
we report null, not infinity. Tied parameters are counted once through the
model’s `named_parameters()`. Buffers are excluded. Distances accumulate in
float64 in bounded chunks, from weights loaded as float32. Reported movement
includes all weight changes, including any weight decay, not just loss-driven
updates. It is not multiplied by the learning rate again.

The observed movement is a lower bound on the full path along a training run:
intermediate updates can move out and back between saved checkpoints. More
frequent checkpoints can increase that bound. Even when a phase has both
boundaries, this does not mean every update was measured. A two-checkpoint
measurement has observed movement equal to net change.

Only compare aligned parameters within one model lineage. Different parameter
counts, layer scales, permutations, or parameterizations do not create a
universal scale of learning. A percentage of starting weight norm is not a
percentage of knowledge. Distances between phases are not causal attribution
of the final model’s behavior. Checkpoint averaging or merging must be disclosed
in the ancestry source and cannot be interpreted as a pure optimizer update.

The chart spaces checkpoints equally for readability, not by elapsed time.
The phase mini bars share a root-mean-square distance scale within one model.
No unobserved trajectory is interpolated into the measurements.

## Prediction differences

Each checkpoint receives the identical raw text of each committed probe, with
no chat template or added special tokens. The runner checks vocabulary, special
token IDs, and encoded inputs across checkpoints and refuses a mismatch.
It evaluates the **next-token probability distribution at the end of each
prompt**, not the entire generated continuation.

For distributions `p` and `q`, with `m = (p + q) / 2`:

```
Jensen–Shannon divergence = (KL(p || m) + KL(q || m)) / 2
```

Logarithms use base two. The result is between zero and one bit. The panel
averages prompts equally, with separate topic averages. Each checkpoint is
compared against the phase’s first measured checkpoint. All prompts and their
SHA-256 fingerprint are stored with the result, alongside model revisions,
package versions, precision, device, and timestamp.

The 18 prompts are an exploratory set written for this feature. Topic labels
such as uncertainty and refusal organize prompts; they are **not measurements
of honesty, safety, refusal rates, or capability**. There are only three prompts
per topic, and there is no population-level confidence claim. Higher divergence
means more change, not improvement. Raw prompts help hold inputs constant;
they do not reproduce the intended chat interface of every instruction model.

Greedy continuations of up to 16 tokens illustrate the changes. These examples
are not graded and may be incomplete. Checkpoint-level distributions are used
in memory; the result retains their aggregate distances and the illustrations.

## Run a measurement

The optional dependencies download model weights and execute inference locally:

```bash
pip install -e '.[changes]'
trainspotting changes pythia-70m-deduped
python scripts/export_site_data.py
```

By default Pythia uses steps 0, 1,000, 10,000 and 143,000. Other checkpoints
can be chosen with a JSON plan. For OLMo 7B Instruct, the supplied plan names
all six phase boundaries:

```bash
trainspotting changes olmo-3-7b-instruct \
  --plan measurement-plans/olmo-7b-instruct-training-change.json
python scripts/export_site_data.py
```

This is a substantial run: the current runner keeps the first and previous
parameter vectors on the CPU while evaluating the next checkpoint. Plan for
approximately three float32 weight copies of memory, inference overhead, and
storage for all downloaded checkpoints. For 7B that is roughly 84 GB before
inference overhead; the small Pythia run needs far less. This
checkout was measured on CPU. Use `--device cuda` or `--device mps` to change
the inference device; weight comparisons remain on CPU.

A plan has a nonempty `stages` list. Each stage provides its registered stage
identifier, `coverage` (`full_stage` or `partial`), a `lineage_source`, and at
least two ordered `checkpoints`. Each checkpoint provides a Hugging Face `repo`
and `revision`; a display `label` and increasing `step` values are optional.
Mutable revisions are resolved to commit hashes before loading, and the output
links to those commits. The plan’s ancestry and phase coverage are assertions
to verify from the training release, not facts inferred from matching shapes.
Never substitute the unrelated OLMo midtraining experiments labeled
`from-2T-ckpt` for the final model’s training trajectory.

Use `--probes path.json` to supply a nonempty list of `{topic, prompt}` objects.
Keep that file and its hash fixed when comparing runs. Successful measurements
atomically replace `results/<target>.changes.json`; failures preserve the
previous artifact. A custom plan replaces that target’s full measurement file,
so include every phase you want retained.

## Published evaluations

```bash
python scripts/capture_training_evaluations.py
python scripts/export_site_data.py
```

The capture script imports only explicitly checked model columns from each
publisher’s stage-comparison table. It stores the source revision, source hash,
and retrieval time. A changed header requires review before it can be imported.
The site displays the publisher’s scores and differences between adjacent
reported phases, in percentage points. It does not average unrelated benchmarks
or assume missing earlier scores are zero. Benchmark names are the source’s
identifiers; each links back to its model card through the source-table link.

## Validation

Offline tests cover reversals versus net distance, parameter-weighted
aggregation, zero norms, incompatible checkpoints, non-finite weights,
divergence endpoints and symmetry, invalid plans, atomic failure behavior,
missing-versus-zero display, signed benchmark deltas, provenance, and chart
edge cases. The small Pythia artifact is also a real end-to-end measurement.
