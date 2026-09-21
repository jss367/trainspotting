# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3) and
post-training (Dolci) data are public, and Pythia (EleutherAI), whose
pretraining corpus and training order are public — and, with the same layers,
any dataset on its own.

Everything runs against public APIs (the HuggingFace datasets-server, Ai2's
infini-gram) without downloading a dataset, and every result carries the
dataset revision, the timestamp and the prompt hash it was produced under. The
results are committed and served as a static site from [`docs/`](docs/README.md).

## What it answers

Each is a command; each is a layer of the same audit.

| Layer | Command | What it says |
|---|---|---|
| Facts | `facts` | Stage sizes for the whole pipeline (pretrain → midtrain → long-context → SFT → DPO → reinforcement learning). |
| Sources | `sources` | Exact composition of each post-training mix, from precomputed column statistics. |
| Values | `classify`, `ask` | How much of the post-training data is about being helpful, honest and harmless versus skill content, by sampling prompts and labeling them with Claude. |
| Language | `languages` | Which natural language each sampled prompt is in, detected locally. |
| Context | `context` | The whole training example behind a prompt: the response fit, the pair pushed between, or the verifier that scores. |
| Preference | `pairs` | What tells the two sides of a DPO pair apart besides the answer: how much longer the chosen side is, and which model wrote each side. |
| Strings, in the samples | `search` | Where a string appears in the sampled examples, and on which side. |
| Strings, over every row | `grep` | How many rows of a mix contain a string, exactly, over the whole mix. |
| Direction | `stance` | Which way an example pushes on a question: toward, away, or neither. |
| Budget | `budget` | Every stage's rate times its size, in tokens the model was fit to, on one scale. |
| Behaviour | `trace` | From a transcript to the stages that most densely hold its distinctive phrases. |
| Benchmarks | `contaminate` | Whether a benchmark's test items are in the training data, at which stage, on which side. |
| Training order | `steps` | For Pythia, where along the run a string was seen. |
| Model development | `changes` | Checkpoint weight distances and prediction changes on fixed prompts; the site also compares published stage evaluations. |
| Corpora | `pretrain`, `lookup`, `find` | Random documents from the pretraining mixes, and exact counts of one text in an indexed corpus. |
| Loss sensitivity | `bif` | Experimental: local loss covariances on standalone text, Pythia-70m only. |

Full descriptions, worked examples and the result-file formats are in
[docs/commands.md](docs/commands.md).

The Olmo 3 Instruct and Think pipelines' final stage mixes **RLVR** (reinforcement
learning with verifiable, programmatic rewards) and **RLAIF** (reinforcement
learning from AI feedback, using an LLM judge). They train the same policy
within one stage after preference tuning; neither is a later stage than the
other. The site separates these reward families in the mix composition and
individual examples. Prompt counts describe the released dataset, not shares
of training updates. See the [Olmo 3 reward design](https://arxiv.org/html/2512.13961v2#S4.SS4.SSS1).
The historical `rlvr` identifier remains in CLI arguments, data paths, and
permalinks for this shared stage. The RL-Zero models train directly from the
base model; their reward families depend on the selected domain.

## Install

```bash
pip install -e .
```

`classify`, `ask` and `stance` need an Anthropic API key (`ANTHROPIC_API_KEY`).
`grep` and `contaminate` need DuckDB (`pip install -e '.[grep]'`). `bif` needs
torch and transformers (`pip install -e '.[bif]'`) and downloads weights.
`steps` needs `tokenizers` (`pip install -e '.[steps]'`).

`changes` also downloads weights and runs local inference; install
`pip install -e '.[changes]'`. See [the training-change guide](docs/training-change.md)
for measurements, coverage, and reproduction.

## Start here

```bash
trainspotting facts olmo-3-7b-instruct           # the pipeline, stage by stage
trainspotting sources olmo-3-7b-instruct --json  # exact mix composition, no key needed
trainspotting context olmo-3-7b-instruct         # store the sampled examples, no key needed
trainspotting classify olmo-3-7b-instruct        # label the sampled prompts (API key)
trainspotting grep olmo-3-7b-instruct "ChatGPT"  # exact count over every row of each mix
trainspotting report olmo-3-7b-instruct          # everything committed about the model, as markdown
```

Every command takes a target: a model (`olmo-3-7b-instruct`, `olmo-3-7b-think`,
`olmo-3-32b-think`, `olmo-3.1-32b-instruct`, the five `olmo-3-7b-rl-zero-*`
models, `pythia-12b-deduped`, `pythia-70m-deduped`) or a dataset on its own
(`wildchat-1m`). See [docs/targets.md](docs/targets.md) for what each is and how
to add one.

Sampled layers draw 1,000 rows per stage by default (`--sample`, `--seed`).
The runs committed under `results/` before September 2026 were drawn at 300;
each result file records its own `sample`. `scripts/refresh_samples.sh <target>`
re-draws every layer of a target at the current default, in the order the
row-index joins between them require. Its `labels` phase also refreshes existing
replicate runs and their agreement summaries, adding one classifier run for each
stage that already has a replicate or agreement artifact. It preserves saved
classifiers for both runs and does not add repeatability checks to new stages.

## Checking the classifier

The values shares rest on a language model's labels, and the model samples at
temperature, so the check that travels with them is a second run:

```bash
trainspotting classify olmo-3-7b-think --replicate   # label the same draw a second time
trainspotting agreement olmo-3-7b-think              # agreement, its interval, Cohen's kappa, share drift
```

`agreement` writes `<target>.<stage>.agreement.json` and the site prints the
result beside the shares it qualifies.

## Values questions

`scripts/human_life_value.sh` is the committed battery behind one question, how
much training teaches that human lives matter. `scripts/values_battery.sh` runs
five more the same way (self-identity, knowledge cutoff, refusal, sycophancy,
admitting uncertainty), each as `ask` → `stance` → `budget`.

## Tests

```bash
pip install -e ".[dev]"
pytest                  # offline, no API key; needs node on PATH for the site suites
pytest --live           # also hit the datasets-server and the infini-gram API
```

What the suite pins, and why, is in [docs/methods.md](docs/methods.md#tests).

## Read more

- [docs/commands.md](docs/commands.md) — every layer, with worked examples and result formats
- [docs/targets.md](docs/targets.md) — the models, datasets and corpora, and adding one
- [docs/methods.md](docs/methods.md) — the taxonomy, how the data is read, the tests, and the [caveats](docs/methods.md#caveats)
- [docs/research/](docs/research/) — the loss-sensitivity write-up
