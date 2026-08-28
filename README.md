# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — currently the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3)
and post-training (Dolci) data are public. The tool answers five kinds of
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
4. **Language** — which natural language each prompt is written in. The Dolci
   datasets carry no language column (the closest is Instruct-SFT's single
   `Multilingual` domain bucket, 4.6% of rows), so this layer detects it
   locally with py3langid over the sampled prompts. No API key, no cost.
5. **Context** — the rest of the training example behind any prompt: the
   response the model is fit to (SFT), the pair it is pushed between (DPO), or
   the verifier that scores it (RL). A prompt on its own can read as the
   opposite of what it teaches, so every count clicks through to this.

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

# Store the full training example behind each sampled prompt (no API key needed)
trainspotting context olmo-3-7b-instruct

# Detect the natural language of each sampled prompt (local, no API key needed)
trainspotting languages olmo-3-7b-instruct

# Same thing without re-fetching: reuse the prompts a committed classify run already holds
trainspotting languages olmo-3-7b-instruct --from-labels
```

`ask` writes `results/<model>.<stage>.ask-<slug>.json` with every sampled
prompt and its yes/no judgment, so the estimate is auditable: read the matched
prompts and check they mean what you think. The
[site](https://jss367.github.io/trainspotting/) renders committed ask runs as
"Custom question" cards, and every bar (taxonomy or ask) clicks open to the
literal prompts behind the count.

`languages` writes `results/<model>.<stage>.languages.json` in the same shape,
with an ISO 639-1 code per prompt. Detection runs line by line and is weighted
by length, because a lot of these prompts are mixed — an English translation or
judge template wrapped around a question in another language. Code fences,
LaTeX, and URLs are stripped first; anything left too short or split too evenly
between languages comes back `undetermined` rather than being guessed at. The
site renders it as its own card, with English on one scale and every other
language on its own, and every language clicks open to its prompts.

Because sampling is deterministic, `--from-labels` reads the prompts straight
out of `results/<model>.<stage>.labels.json` instead of paging HuggingFace
again. It produces byte-identical output and takes about a second.

py3langid covers 97 languages, so anything outside that set lands on its
nearest neighbour — Somali prompts in Dolci come back as Afrikaans. It also
does not separate close pairs: most of what it calls Indonesian in Dolci is
Malay. Treat the long tail as approximate and read the prompts, which is what
the drill-down is for.

`context` re-fetches the same rows (sampling is deterministic in `--sample`
and `--seed`, so it lands on exactly the rows a `classify` or `ask` run
labeled) and writes `results/<model>.<stage>.context.json`, joined to the
labeled prompts by their first 400 characters. Those files are a cache of
upstream rows, so they are gitignored; `scripts/export_site_data.py` copies
them into `docs/data/`, which is the committed copy. Responses live only here —
`classify` and `ask` records keep the prompt and the label, so the files the
site loads to draw a chart stay small.

Each prompt on the site then carries a **see it in training context** button
that opens the whole example, drawn the way its stage trains:

| Stage | What the view shows |
|---|---|
| SFT | every turn, the reasoning span folded away, and how much of the example is target text versus context |
| DPO | chosen and rejected side by side, which model wrote each, how the pair was labeled, and the length gap between them |
| RL | no stored response — the verifier, what it checks (ground truth, constraint), and how often reference rollouts passed |

This matters for reading the numbers. A prompt like *"Write a program to decide
if a child should be saved based on race"* counts as harmlessness content, and
only the pair behind it shows the model is trained toward refusing it.

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
  not show; the `sources` layer's reward-type breakdown and the `context`
  layer's verifier view are the complement.
- `context` names each RL mix's verifier by matching its `dataset_source`
  against known mixes (math answer match, code unit tests, constraint checker,
  LLM judge). The raw source tag travels with every record, so the inference is
  checkable, and RL rows carry no judge rubric at all.
- Context fields are cut at 4,000 characters. Every view links to the exact row
  on HuggingFace, which is where the untruncated example lives.
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
