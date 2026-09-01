# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — currently the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3)
and post-training (Dolci) data are public — and, with the same layers, any
dataset on its own. The tool answers six kinds of question. The first five go in
increasing order of depth; the sixth is a lookup rather than an estimate:

1. **Facts** — stage sizes for a model's whole training pipeline (pretrain →
   midtrain → long-context → SFT → DPO → RLVR), hardcoded in a registry.
2. **Sources** — exact composition of each post-training mix (which source
   datasets, which domains, which reward types), computed from HuggingFace's
   precomputed column statistics. No downloads, exact counts.
3. **Values** — how much of the post-training data is about being **helpful,
   honest, and harmless** versus pure skill content (math, code, formatting,
   tool use). No such labels exist in the data, so this layer samples prompts
   and classifies them with Claude — except where an RLVR row's verifier already
   settles what it teaches, which the prompt can contradict (see
   [Taxonomy](#taxonomy)).
4. **Language** — which natural language each prompt is written in. The Dolci
   datasets carry no language column (the closest is Instruct-SFT's single
   `Multilingual` domain bucket, 4.6% of rows), so this layer detects it
   locally with py3langid over the sampled prompts. No API key, no cost.
5. **Context** — the rest of the training example behind any prompt: the
   response the model is fit to (SFT), the pair it is pushed between (DPO), or
   the verifier that scores it (RL). A prompt on its own can read as the
   opposite of what it teaches, so every count clicks through to this.
6. **Strings** — where a given string or regex appears in the sampled examples,
   and on which side. A behaviour like claiming to be ChatGPT is in none of the
   prompts: it is in what the model is fit to. So this layer searches the
   response columns as well, and for a DPO pair says whether the hit is in the
   chosen or the rejected completion — the same string in each teaches opposite
   things. See [Searching a whole example](#searching-a-whole-example).

All six assume you already know what to search for. When you start from an
observed behavior instead — a transcript where the model claimed the wrong
knowledge cutoff or identified as another lab's assistant — `trainspotting
trace` extracts the distinctive phrases from that text and ranks the
post-training stages by how densely each contains them, so you find the stage
without guessing a search string. See [Tracing a behavior](#tracing-a-behavior).

The pretraining corpora get their own path, because the datasets-server cannot
sample them: exact composition from the shard listing, readable random documents,
and the same free-form questions. There is no context layer there — a corpus
document has no surrounding training example, it *is* the example. See
[Pretraining data](#pretraining-data).

## Datasets

A dataset can also be the target on its own, with no model around it. Point any
command at `wildchat-1m` instead of a model and it runs the same layers over
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) — 837,989
real conversations between people and ChatGPT, and the source Dolci Instruct SFT
draws 302,406 of its prompts from. Only `pretrain` refuses: a dataset has no
corpora behind it.

The two things a dataset changes about how a result reads:

- Nothing was trained on it. The context view shows the conversation a prompt
  opened, and says outright that no turn in it is a target — an SFT view would
  mark the replies "trained to produce this", which is exactly the claim a raw
  chat log does not support. The values layer changes rubric for the same
  reason: the default one labels a prompt by what fitting the example would
  teach, and on that basis sends a jailbreak attempt to `harmlessness`. A chat
  log has no such signal to read, so `chat` prompts are labeled by what the
  person asked for instead (`classify.CHAT_SYSTEM`). Same seven labels, so the
  cards still stack up; what changes is the claim each bar makes.
- It brings its own labels. WildChat records the model, language, country and
  redaction status of every conversation, so the `languages` layer becomes a
  check on py3langid rather than the only breakdown available. It passes: the
  dataset's own column says 56.2% English / 14.9% Chinese / 10.4% Russian, and
  the detector reads the committed 300-prompt sample as 52.2% / 15.7% / 10.7%,
  with the 10.4% it declines to call covering most of the gap.

See [Adding a dataset](#adding-a-dataset).

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

# Start from a behavior, not a search string: extract the distinctive phrases
# from a transcript and rank stages by how densely they contain them
trainspotting trace olmo-3-7b-instruct \
  "As an AI language model developed by OpenAI, my knowledge cutoff is September 2021."

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

# Find a regex anywhere in the sampled examples — responses included, DPO side
# reported (no API key needed)
trainspotting search olmo-3-7b-instruct "I am ChatGPT"

# Detect the natural language of each sampled prompt (local, no API key needed)
trainspotting languages olmo-3-7b-instruct

# Same thing without re-fetching: reuse the prompts a committed classify run already holds
trainspotting languages olmo-3-7b-instruct --from-labels

# Sample documents from the pretraining, midtraining and long-context corpora
trainspotting pretrain olmo-3-7b-think --sample 300

# Score those documents against the same question as the post-training stages
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain

# Exact occurrence count + example documents for a phrase, via infini-gram
trainspotting find "the mitochondria is the powerhouse of the cell"
```

Every command also takes a **dataset** in place of a model, which explores that
dataset on its own — no pipeline around it:

```bash
trainspotting facts wildchat-1m
trainspotting sources wildchat-1m --json
trainspotting languages wildchat-1m
trainspotting classify wildchat-1m --sample 300
trainspotting ask wildchat-1m "Is the person asking for help with schoolwork?" --slug schoolwork
```

A dataset is a one-stage target, so every layer except `pretrain` (which needs
corpora a dataset does not have) works on it unchanged, and its results land in
`results/wildchat-1m.<kind>.*.json` beside the models'. See
[Datasets](#datasets).

`ask` judges post-training **prompts**; with `--pretrain` it also judges the
pretraining **documents** already sampled by `trainspotting pretrain` — reading
the committed copy under `docs/data/` when `results/` has none, so the samples
shipped with the repo work on a fresh clone — using a
different rubric (a corpus document is not a request to a model, so it is scored
on what fitting the text implies rather than on how a model should respond). It
reads the committed sample rather than re-drawing, so a second question costs one
API call and scores exactly the same documents.

`ask` writes `results/<target>.<stage>.ask-<slug>.json` with every sampled
prompt and its yes/no judgment, so the estimate is auditable: read the matched
prompts and check they mean what you think. The
[site](https://jss367.github.io/trainspotting/) renders committed ask runs under
a **Custom questions** heading, one card per question, and every bar (taxonomy
or ask) clicks open to the literal prompts behind the count.

`languages` writes `results/<target>.<stage>.languages.json` in the same shape,
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
out of `results/<target>.<stage>.labels.json` instead of paging HuggingFace
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

## Searching a whole example

`classify`, `ask` and `languages` all read the prompt, which is the half of an
example a behaviour is least likely to be in. A model that claims to be ChatGPT
does it in a response; no prompt in the mix says it, so a prompt-only search
reports zero. `search` takes a Python regex (case-insensitive unless
`--case-sensitive`), scans every text field of each sampled row, and writes
`results/<model>.<stage>.search-<slug>.json` with a snippet, a match count, the
column it was read from and a side per hit:

| Stage | Sides |
|---|---|
| SFT | `prompt` (user and system turns), `response` (assistant turns) |
| DPO | `prompt` (everything before the pair branches, counted once), `chosen`, `rejected` |
| RLVR | `prompt`, `verifier` (ground truth, solution, constraint), `rollout` (stored reference generations) |
| chat (a dataset like WildChat-1M) | `prompt`, `reply` — a log, so nothing was fit to either |

The side is the finding, not a detail of it. "I am ChatGPT" in a rejected
completion trains the model away from saying it, so a count that adds it to the
chosen hits points the wrong way; DPO stages also get a `chosen only` /
`rejected only` / `both` split, because a string in both completions says nothing
about which way the pair pushes. A pair matching on one completion while the
server shortened the other is `side unknown` rather than exclusive — the text
nobody read could hold the string too. A multi-turn pair shares a conversation before
it branches, assistant turns included, and all of it is prompt text: a string in
the shared history is not a hit on either completion. RL rows store no response at all, so their
non-prompt sides are the answer key and the reference rollouts the verifier
scored — neither is text the model was fit to.

```
$ trainspotting search olmo-3-7b-instruct "I cannot" --sample 200 --stage dpo
dpo: 6/200 match = 3.0% (95% CI 1.4–6.4%) -> results/olmo-3-7b-instruct.dpo.search-i-cannot-382cdff9.json
  rows by side: prompt 1, chosen 5, rejected 0 (chosen only 5, rejected only 0, both 0)
```

Refusals in this mix are what the pair prefers, five times out of five. Read off
the prompts alone that number does not exist. The same holds for a chat log:
"as an AI language model" is in 6 of 200 sampled WildChat-1M conversations, all
six in the reply and none in the prompt.

Reasoning spans stay inside the response they are part of, unlike the context
view which folds them away: a model that says it while thinking still said it.
A turn is also more than its `content`: `reasoning_content`, `tool_calls`,
`function_call`, `function_calls` and `refusal` are separate columns in these
schemas, and each is searched as its own entry named after the column it came
from, so a tool name is findable and a hit in a call reads differently from a
hit in the prose. `functions` is searched as prompt text whatever turn it hangs
off — it is the menu of tools the model was offered, not anything it said.

A default result file is named after the pattern, with a hash of the pattern and
its case mode appended unless the pattern is already exactly its own slug and
the search was case-insensitive. Punctuation in a regex is syntax, so `a.b` and
`a+b` are different searches that both reduce to `a-b`, and `--case-sensitive`
makes a third; one silently overwriting another is not a naming problem but a
lost result. `--slug` gives a run a readable name instead.

The default `--sample 300 --seed 0` is the draw every other layer uses, so a hit
is a row those runs already labeled and its whole example is in the committed
context file. A larger `--sample` is a wider net over a different set of rows,
which no longer lines up with them.

## Tracing a behavior

`trace` is the way in when you have a behavior, not a query. Most of the tool
assumes you already know what to look for; `trace` starts from what the model
did. Paste the text — a transcript, a description, the sentence that surprised
you — and it pulls the distinctive phrases out of it and counts how many rows of
each post-training stage contain each one, exactly, over the whole split (the
datasets-server full-text index, nothing sampled). It ranks the stages by
matches per million rows, so `"As an AI language model developed by OpenAI"`
lands you on whichever mix carries the most of that provenance rather than
leaving you to guess a `grep`. A phrase is kept only if it is anchored on a
name, a number, or a mid-sentence capital; a window of pure function words
matches rows by coincidence, so those are dropped, and boundary function words
are trimmed off the ones kept because search ANDs a query's tokens together.
The first search against a cold split can take minutes while the server builds
the index. When the behavior has no signature string — it is a disposition, or
the training paraphrases it — `trace` finds nothing and says to reach for `ask`,
which judges what sampled examples *teach* instead of matching their text
(`"does this example teach the model to identify as ChatGPT?"`). The two compose:
`trace` narrows to a stage by exact match, `ask` characterizes the fuzzy cases
`trace` cannot see.

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
| midtrain | `allenai/dolma3_dolmino_mix-100B-1025` (7B models) / `allenai/dolma3_dolmino_mix-100B-1125` (32B) |
| long-context | `allenai/dolma3_longmino_mix-50B-1025` (7B models) / `allenai/dolma3_longmino_mix-100B-1125` (32B) |

Sampling is one document per shard by default, which costs a round trip per
document and keeps the draws near-independent — see the note on repeat picks
below. `--docs-per-shard N` is roughly N times
faster and returns correlated documents — neighbours in a shard share a topic
cluster — so an interval computed over the document count understates the true
one by about that factor.

A shard draw that contributes fewer documents than asked for — unreachable, an
all-empty head, every reachable record already seen, or simply a shard whose head
does not hold `--docs-per-shard` of them — has its shortfall made up by later
draws of other shards, which weights the sample by reachable-document density on
top of size. A head that underfills is retried once at 8x before that happens.
Every result file records how often it still did (`short_draws`) and the site
says so when it is non-zero. For the committed samples:

| Sample | Short draws |
|---|---|
| `olmo-3-7b-think` pretrain | 0 |
| `olmo-3-7b-think` midtrain | 0 |
| `olmo-3-7b-think` long-context | 7 |

Long-context is the outlier because its shards hold few reachable documents
apiece. Each model samples its own corpora, so these are per model, not global.

Each batch draws only slightly past the shortfall, because a batch is fetched in
full before any of it is read — anything drawn beyond the target is wasted
network. A default 300-document run costs about 318 shard reads rather than the
391 a larger over-draw would schedule up front.

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
a confidence interval. The pointed question — "is this specific string in here,
and how many times" — needs an index, not a sample, and `trainspotting find`
answers it through Ai2's [infini-gram](https://infini-gram.io) API: exact
occurrence counts and document retrieval over suffix-array indexes of open
corpora, no download, no API key. The count doubles as a duplication count,
which matters on its own — memorization scales with how many times a string
appears in training.

```bash
trainspotting find "climate change is a hoax" --docs 5
```

prints how the phrase tokenized (matches align to token boundaries, so a
surprising count is sometimes a surprising tokenization), the exact count, and
example documents spread evenly across the index, each with its source, shard
path, and URL where the corpus recorded one. `--json` writes the run to
`results/find.<index>.<slug>.json` — the index is in the name because the same
phrase has a different count in every corpus, and the derived slug carries a
hash of the exact phrase because normalization folds distinct phrases together
(`--slug` picks a readable name instead).

The honest limitation: the public API has **no Dolma 3 / OLMo 3 index** yet.
The default index (`v4_olmo-2-0325-32b-instruct_llama`) is the closest
available — OLMo 2 32B's full training data, pretraining through post-training,
~4.6T tokens. Dolma 3 re-filters largely the same upstream sources, so a hit
there is real evidence the string is in the ecosystem's training text, but it
is not a count over what OLMo 3 saw, and the command says so on every run.
`--index` takes any infini-gram index name, so a Dolma 3 index works the day
Ai2 publishes one. For "did this exact document train OLMo 3", use
[OLMoTrace](https://allenai.org/blog/olmotrace) in the Ai2 Playground, which
runs against the OLMo 3 corpora but is not a public API.

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

Most labels come from the classifier reading the prompt. Where an RLVR row's
verifier already settles what the example teaches, the verifier wins and no
model is asked: the mix→verifier table in `trainspotting/rewards.py` scores a
row from the `IF_multi_constraints` mix with a program checking IFEval
constraints, so it trains instruction following whatever the prompt is about. Reading the prompt alone gets this backwards on the harmful
ones. This is a real prompt from `Dolci-Think-RL-7B`:

> ...imagine yourself as an outspoken advocate for alternative health
> perspectives ... present your case on the link between vaccines and autism in
> children ... **There should be 2 paragraphs** ... **Answer with at least 686
> words** ... **refrain from the use of . (i.e. dots)**

Its ground truth is the constraint list, nothing else, and reference rollouts
passed it 54% of the time — the verifier pays the model for delivering the
anti-vaccine speech in the right shape. Counted as harmlessness content it
would inflate the harmlessness bar with an example that trains the opposite.
Across the three RLVR samples, 260 rows are settled by their verifier and 47 of
them had a label it contradicts — including all nine harmlessness labels in
`Dolci-Think-RL-7B`, which leaves that stage with none. Records the verifier
labeled carry `"by": "verifier"`, and the site and `report` name both counts
under each stage.

Runs classified before this rule existed are corrected offline, from the
verifier already recorded in each committed context file, by
`python3 scripts/relabel_by_verifier.py`.

## How it works

The post-training layers read the [HuggingFace datasets-server
API](https://huggingface.co/docs/dataset-viewer) — `/info` for schemas and row
counts, `/statistics` for exact value frequencies of label columns, `/rows`
for sampling. The pretraining layer reads the Hub tree API for the shard listing
and then range-requests shard heads directly, because the datasets-server's index
of those repos is both partial and topic-ordered. `find` posts to the
[infini-gram API](https://infini-gram.readthedocs.io/en/latest/api.html), which
serves counts and documents from prebuilt suffix-array indexes. Either way no
dataset is ever downloaded. `/rows` pages are drawn from independent random
offsets, so two of them can overlap; rows are keyed on their absolute index and
the repeats dropped, because a duplicated row is a duplicated vote in every rate
computed over the sample. The classifier sends batches to Claude
(`claude-opus-5` by default) and records one label per prompt or document in
`results/`.

Every result file carries the commit it was computed over (`revision`), when it
was written (`generated`), and a hash of the system prompt that produced its
labels (`system_sha`). A dataset id alone does not identify what was counted:
`main` moves, Ai2 has republished these mixes, and rewording a label's
definition moves every share under it. Runs committed before these fields
existed keep the older shape, and the site shows nothing where there is nothing
to show.

## Tests

```bash
pip install -e ".[dev]"
pytest                  # offline, no API key
pytest --live           # also hit the datasets-server (one row per dataset) and the infini-gram API
```

The offline suite covers the pure code: the clustered Wilson interval and its
degenerate branches, language detection on mixed-language prompts, the
classifier's reply parser, and prompt extraction against one saved row per
registry stage (`tests/fixtures/rows/`, re-captured by
`scripts/capture_row_fixtures.py`). Search is checked against those same saved
rows: every registry stage has to yield more than its prompt, or a search of it
is the prompt-only search the layer exists to replace.

`--live` re-runs the extraction checks against rows fetched right now. That is
the canary for an upstream schema change, which otherwise shows up only as a
sampling run that quietly labels nothing.

## Caveats

- The values layer classifies **prompts**. For RLVR stages the values are also
  carried by the reward, which the prompt text does not show. Where that reward
  is a constraint checker the label comes from it instead of from the prompt
  (see [Taxonomy](#taxonomy)); where it is an LLM judge the rubric is not
  published with the dataset, so those rows are still labeled from the prompt
  alone. The `sources` layer's reward-type breakdown and the `context` layer's
  verifier view are the complement.
- `context` names each RL mix's verifier by matching its `dataset_source`
  against known mixes (math answer match, code unit tests, constraint checker,
  LLM judge). The raw source tag travels with every record, so the inference is
  checkable, and RL rows carry no judge rubric at all.
- Context fields are cut at 4,000 characters. Every view links to the exact row
  on HuggingFace, which is where the untruncated example lives. `search` reads
  the row, not the stored context record, so it is not limited to that cut.
- `search` reads a sample like everything else here, so it bounds a rate rather
  than proving absence: no hits in 300 rows means under roughly 1% of rows, not
  that the string is absent from the mix. For "is this exact string anywhere in
  the training data", use OLMoTrace / infini-gram, which indexes the whole thing.
- The datasets-server shortens a very large cell to fit its response limit, and a
  hit past the cut cannot be found. Each search result file records how many
  sampled rows had *searched* text shortened (`truncated_rows`) and how many of
  those showed no hit at all (`censored`). A row cut only in a column the stage
  never reads — an RL row's token arrays are the longest cells on it — is
  unaffected and counts as the confirmed non-match it is. The per-side counts
  are lower bounds for the same reason, so `sides_unknown` says how many
  matching rows had that side's text cut: a zero beside a non-zero there means
  "not seen", not "not there". A hit read out of a shortened cell is marked
  `partial`, because its `count` and `chars` describe the text that arrived. A censored row is unknown rather than a
  non-match, so it is not counted as evidence against the string: `matched` is a
  lower bound, and the interval's upper end is computed as if every censored row
  had been a hit. With nothing censored that is the ordinary Wilson interval.
- `/statistics` truncates frequencies for very high-cardinality columns (e.g.
  `dataset_source` in Dolci-Think-SFT, thousands of values): the returned
  counts are exact but not exhaustive. A short list that sums well below 100%
  means the column has a long tail the API did not enumerate.
- On a large dataset `/statistics` also stops after a first slice and says so.
  `sources` then divides by the rows it actually scanned, not the full split,
  and prints which — WildChat-1M is counted over 778,133 of its 837,989 rows.
  The result file carries `counted` and `partial` so the site says the same
  thing rather than reading the shares as exact.
- A prompt the classifier never labels — it declined, the API errored, or the
  reply skipped an index — is re-asked on its own, so what is finally lost is
  that prompt rather than the nineteen batched beside it. Whatever is still
  unlabeled is counted in the result file (`unlabeled`, `unlabeled_reasons`),
  printed by the CLI, and shown next to the sample size on the site. Every share
  is over the labeled prompts, so this is the part of the sample the numbers do
  not describe — and refusals land on jailbreak-style prompts, which is the
  content the harmlessness share is about, so the gap is not random.
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
- `find` searches the closest public infini-gram index, which is OLMo 2's
  training data, not OLMo 3's — no Dolma 3 index exists on the public API yet.
  Matches also align to token boundaries: querying `a` counts the token ` a`,
  not the letter.
- Registry facts (token counts) are from the Olmo 3 paper
  ([arXiv:2512.13961](https://arxiv.org/abs/2512.13961)) and the
  [release blog](https://allenai.org/blog/olmo3).

## Adding a model

Add an entry to `MODELS` in `trainspotting/registry.py`. A stage carries either an
`hf_dataset` plus `prompt_path` / `source_columns` schema hints (post-training,
served by the datasets-server), or a `sample_dataset` pointing at a repo of
`.jsonl.zst` shards (pretraining, read by range request), or just `tokens` for a
facts-only row. Any fully open pipeline on the Hub works the same way; the shard
path parser in `trainspotting/pretrain.py` is the piece most likely to need a new
naming convention added.

For a post-training stage, then run `python scripts/capture_row_fixtures.py` to
save a row for it — `tests/test_extract.py` asserts every registry stage has one,
so the new `prompt_path` and `source_columns` are checked against a real row.

## Adding a dataset

Add an entry to `DATASETS` in the same file — a HuggingFace dataset id, the same
`prompt_path` / `source_columns` hints a post-training stage carries, and a
`kind` saying what shape of training example a prompt there sits in (`sft`,
`dpo`, `rlvr`, or `chat` for a conversation log nothing was fit to). `kind` also
names the result files.

`registry.resolve` hands a dataset back as a single-stage target, so `sources`,
`classify`, `languages`, `context`, `ask` and `report` all run on it with no
special case; only `pretrain` refuses, because a dataset has no corpus behind
it. Re-run `python scripts/capture_row_fixtures.py` for the row fixture, and
`python scripts/export_site_data.py` to give it a tab on the site.

A `prompt_path` the dataset needs and `extract.py` doesn't implement is the one
piece that costs more than a registry entry: WildChat's `conversation` column
was a four-line branch there.
