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

The pretraining corpora get their own path, because the datasets-server cannot
sample them: exact composition from the shard listing, readable random documents,
and the same free-form questions. There is no context layer there — a corpus
document has no surrounding training example, it *is* the example. See
[Pretraining data](#pretraining-data).

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

# Sample documents from the pretraining, midtraining and long-context corpora
trainspotting pretrain olmo-3-7b-think --sample 300

# Score those documents against the same question as the post-training stages
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain
```

`ask` judges post-training **prompts**; with `--pretrain` it also judges the
pretraining **documents** already sampled by `trainspotting pretrain` — reading
the committed copy under `docs/data/` when `results/` has none, so the samples
shipped with the repo work on a fresh clone — using a
different rubric (a corpus document is not a request to a model, so it is scored
on what fitting the text implies rather than on how a model should respond). It
reads the committed sample rather than re-drawing, so a second question costs one
API call and scores exactly the same documents.

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
LaTeX, and URLs are stripped first. Confidence divides the winning language's
weight by the prompt's full letter mass, so both ways of being unsure pull it
down: text split between languages, and text the detector was never sure about.
Anything at or below half comes back `undetermined` rather than being guessed
at — which covers bare greetings, mostly-tabular tasks, and code with comments
in a second language. The
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

## Pretraining data

Dolma 3 is on the Hub, but the dataset viewer cannot sample it. It indexes only
the first ~5 GB of each repo (`"partial": true`), and the shards are ordered by
topic cluster, so walking `/rows` walks one cluster at a time — offsets 0–100k of
the 150B mix are adult content, 300k onward are art. A sample drawn that way is a
tour of whichever clusters sort first.

`trainspotting pretrain` reads the repo files instead. Dolma 3 ships as
`.jsonl.zst` shards under paths that name their own provenance:

```
data/common_crawl-crime_and_law-0007/shard_00000112.jsonl.zst
data/olmocr_science_pdfs-health/shard_00000044.jsonl.zst
data/stack_edu-Python-0001/shard_00000009.jsonl.zst
```

The sampler resolves the dataset ref to a commit SHA, lists every shard once at
that revision (cached in `.shard-cache/`, keyed on the SHA so an upstream
republish invalidates it rather than serving stale paths), draws shards with
probability proportional to compressed size, and pulls each pick's head with an
HTTP range request — a zstd stream decodes from the front, so ~96 KB over the
wire yields a document out of a shard that may be 400 MB. Nothing is downloaded.

A shard whose head decodes to nothing gets one retry at 8x the read. That case is
not rare: the long-context mixes hold documents past 200k characters, so a third
of `lc_synth-rex_s2pdf`'s shards yield zero documents at 96 KB and would
otherwise drop out silently — biasing the sample against exactly the long
documents that stage exists to train on.

That makes the composition **exact**: it comes from the full file listing, not a
sample, and every result file records the revision it was listed at so the count
is checkable against a specific tree rather than a moving `main`. For `allenai/dolma3_mix-6T-1025-7B` — the actual mix Olmo 3 7B was
pretrained on — that is 65,718 shards, 3.9 TB compressed, 49 source/topic groups.
Only the documents are sampled.

Each document carries its source, topic cluster, shard path, and the filtering
metadata Dolma 3 recorded for it: which Common Crawl snapshot it came from, what
the quality classifier scored it, the language ID confidence, and how many exact
duplicates it had.

| Stage | Corpus sampled |
|---|---|
| pretrain | `allenai/dolma3_mix-6T-1025-7B` (7B models) / `allenai/dolma3_mix-6T` (32B) |
| midtrain | `allenai/dolma3_dolmino_mix-100B-1125` |
| long-context | `allenai/dolma3_longmino_mix-100B-1125` |

Sampling is one document per shard by default, which costs a round trip per
document and keeps the draws independent. `--docs-per-shard N` is roughly N times
faster and returns correlated documents — neighbours in a shard share a topic
cluster — so an interval computed over the document count understates the true
one by about that factor.

A shard draw that contributes fewer documents than asked for — unreachable, an
all-empty head, every reachable record already seen, or simply a shard whose head
does not hold `--docs-per-shard` of them — has its shortfall made up by later
draws of other shards, which weights the sample by reachable-document density on
top of size. A head that underfills is retried once at 8x before that happens.
Every result file records how often it still did (`short_draws`) and the site
says so when it is non-zero: 0 for the pretrain and midtrain samples, 12 of ~390
draws for long-context, whose shards hold few reachable documents apiece.

Shards are drawn with replacement, so the same shard can come up twice; the
per-document RNG is seeded on the pick index as well as the shard path so a
repeat draw yields a different document, and an explicit dedupe catches the
residual chance collision. Byte-weighting makes repeats common wherever a big
corpus sits in few shards (`lc_synth-rex_s2pdf` is 21 GB over 55 shards).

The 7B pretraining mix has one wrinkle Ai2 documents: some olmOCR science PDFs
were redacted to `[REMOVED]` after the model was trained, so a few sampled
documents show a placeholder where the model saw real text. The 32B mix
(`dolma3_mix-6T`) is complete.

### What this does and doesn't answer

Sampling answers "what is in here" — the unconditional question, with a rate and
a confidence interval. [OLMoTrace](https://allenai.org/blog/olmotrace) /
infini-gram answers "is this specific string in here", which needs a query you
already have. They are complements; for "did this exact document train the
model", use OLMoTrace.

The bias sampling leaves is positional. A range request only reaches the front of
a shard, so each sampled document comes from its shard's first few hundred (one
picked uniformly from them), never the tail. Shard draws are proper; position
within a shard is not corrected for. That caveat
travels in every result file and is printed on the site.

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

The post-training layers read the [HuggingFace datasets-server
API](https://huggingface.co/docs/dataset-viewer) — `/info` for schemas and row
counts, `/statistics` for exact value frequencies of label columns, `/rows`
for sampling. The pretraining layer reads the Hub tree API for the shard listing
and then range-requests shard heads directly, because the datasets-server's index
of those repos is both partial and topic-ordered. Either way no dataset is ever
downloaded. The classifier sends batches to Claude (`claude-opus-5` by default)
and records one label per prompt or document in `results/`.

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
  gives roughly ±5% worst case. Post-training intervals assume independent draws.
  Corpus intervals do not: they are widened by the measured design effect of
  clustering by shard, so `--docs-per-shard` runs whose matches bunch inside
  shards get an honestly wider interval, and runs whose matches are spread
  evenly are not penalised for the grouping alone. The corrected interval is
  computed once, in the CLI, and stored in the result file rather than
  recomputed by the site.
- The pretraining sampler only sees documents a range request can reach — the
  first few hundred in each shard, one drawn uniformly from those. Shards are
  drawn properly; position within a shard is not corrected for.
- Registry facts (token counts) are from the Olmo 3 paper
  ([arXiv:2512.13961](https://arxiv.org/abs/2512.13961)) and the
  [release blog](https://allenai.org/blog/olmo3).

## Adding a model

Add an entry to `trainspotting/registry.py`. A stage carries either an
`hf_dataset` plus `prompt_path` / `source_columns` schema hints (post-training,
served by the datasets-server), or a `sample_dataset` pointing at a repo of
`.jsonl.zst` shards (pretraining, read by range request), or just `tokens` for a
facts-only row. Any fully open pipeline on the Hub works the same way; the shard
path parser in `trainspotting/pretrain.py` is the piece most likely to need a new
naming convention added.
