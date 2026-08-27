# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — currently the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3)
and post-training (Dolci) data are public. The tool answers three kinds of
question, in increasing order of depth:

1. **Facts** — stage sizes for a model's whole training pipeline (pretrain →
   midtrain → long-context → SFT → DPO → RLVR), hardcoded in a registry.
2. **Sources** — exact composition of each post-training mix (which source
   datasets, which domains, which reward types), computed from HuggingFace's
   precomputed column statistics. No downloads, exact counts.
3. **Values** — how much of the post-training data is about being **helpful,
   honest, and harmless** versus pure skill content (math, code, formatting,
   tool use). No such labels exist in the data, so this layer samples prompts
   and classifies them with Claude.

Pretraining-scale corpora (Dolma 3 Mix is 5.9T tokens, ~24 TB of text) only get
the facts treatment. For provenance questions against pretraining data, use
Ai2's [OLMoTrace](https://allenai.org/blog/olmotrace) / infini-gram instead.

## Install

```bash
pip install -e .
```

The `values` layer needs an Anthropic API key (`ANTHROPIC_API_KEY`).

## Usage

```bash
# Stage sizes for a model's full pipeline
trainspotting facts olmo-3-7b-instruct

# Exact source/domain/reward-type composition of each post-training stage
trainspotting sources olmo-3-7b-instruct --json

# Sample 300 prompts per stage and label each one with Claude
trainspotting classify olmo-3-7b-instruct --sample 300

# Combined markdown report with Wilson 95% CIs on the sampled estimates
trainspotting report olmo-3-7b-instruct

# Free-form question: sample each stage and judge every prompt yes/no
trainspotting ask olmo-3-7b-instruct \
  "Is this training example about caring about human lives?" \
  --slug caring-about-human-lives
```

`ask` writes `results/<model>.<stage>.ask-<slug>.json` with every sampled
prompt and its yes/no judgment, so the estimate is auditable: read the matched
prompts and check they mean what you think. The
[site](https://jss367.github.io/trainspotting/) renders committed ask runs as
"Custom question" cards, and every bar (taxonomy or ask) clicks open to the
literal prompts behind the count.

Registered models: `olmo-3-7b-instruct`, `olmo-3-7b-think`, `olmo-3-32b-think`.

## Taxonomy

Each sampled prompt gets exactly one primary label:

| Label | Meaning |
|---|---|
| `harmlessness` | Handling unsafe/harmful requests: refusals, jailbreak resistance, safety-sensitive advice |
| `honesty` | Truthfulness and calibration: admitting uncertainty, refusing to fabricate, correcting false premises, resisting pressure to agree |
| `helpfulness` | General assistance: chat, writing, advice, explanation, everyday Q&A |
| `capability` | Skill content: math, code, science, logic |
| `instruction_following` | Precise formal constraints (formats, word counts) |
| `tool_use` | Function calling / agentic tool use |
| `other` | None of the above |

## How it works

Everything reads the [HuggingFace datasets-server
API](https://huggingface.co/docs/dataset-viewer) — `/info` for schemas and row
counts, `/statistics` for exact value frequencies of label columns, `/rows`
for sampling — so no dataset is ever downloaded. The classifier sends batches
of prompts to Claude (`claude-opus-5` by default) and records one label per
prompt in `results/`.

## Caveats

- The values layer classifies **prompts**. For RLVR stages the values are also
  carried by the reward (verifier or judge rubric), which the prompt text does
  not show; the `sources` layer's reward-type breakdown is the complement.
- `/statistics` truncates frequencies for very high-cardinality columns (e.g.
  `dataset_source` in Dolci-Think-SFT, thousands of values): the returned
  counts are exact but not exhaustive. Percentages in `sources` output are
  against the full row count, so a short list that sums well below 100% means
  the column has a long tail the API did not enumerate.
- Sampled estimates come with Wilson 95% intervals in `report`; 300 samples
  gives roughly ±5% worst case.
- Registry facts (token counts) are from the Olmo 3 paper
  ([arXiv:2512.13961](https://arxiv.org/abs/2512.13961)) and the
  [release blog](https://allenai.org/blog/olmo3).

## Adding a model

Add an entry to `trainspotting/registry.py`: stages with either `tokens`
(facts only) or an `hf_dataset` plus `prompt_path` / `source_columns` schema
hints. Any fully open pipeline with datasets on the Hub works the same way.
