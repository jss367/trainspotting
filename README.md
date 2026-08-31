# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — currently the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3)
and post-training (Dolci) data are public. The tool answers six kinds of
question, in increasing order of depth:

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
6. **Strings** — how many rows of a mix contain a given string, exactly, over
   every row rather than a sample. The five layers above all estimate an
   unconditional rate from 300 prompts; this one answers the question you have
   once you know what you are looking for, which a sample that size cannot: a
   pattern in 0.1% of a mix is expected to miss it entirely. Counts are then read
   across the pipeline as a ranking — which stage most plausibly taught the
   string, by rate rather than by hits, and which side of the example it is on.
   See [Searching for a string](#searching-for-a-string).

The pretraining corpora get their own path, because the datasets-server cannot
sample them: exact composition from the shard listing, readable random documents,
and the same free-form questions. There is no context layer there — a corpus
document has no surrounding training example, it *is* the example. See
[Pretraining data](#pretraining-data).

## Install

```bash
pip install -e .
```

The `values` layer needs an Anthropic API key (`ANTHROPIC_API_KEY`). The `grep`
layer needs DuckDB (`pip install -e '.[grep]'`); nothing else does.

## Usage

```bash
# Stage sizes for a model's full pipeline
trainspotting facts olmo-3-7b-instruct

# Exact source/domain/reward-type composition of each post-training stage
trainspotting sources olmo-3-7b-instruct --json

# Sample 300 prompts per stage and label each one with Claude
trainspotting classify olmo-3-7b-instruct --sample 300

# Combined markdown report: sampled rates with Wilson 95% CIs, and every
# committed string trace ranked by which stage most plausibly taught it
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

# Exact count of the rows of each mix containing a string (needs DuckDB)
trainspotting grep olmo-3-7b-think "ChatGPT" --stage dpo
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
[site](https://jss367.github.io/trainspotting/) renders committed ask runs under
a **Custom questions** heading, one card per question, and every bar (taxonomy
or ask) clicks open to the literal prompts behind the count.

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

## Searching for a string

`trainspotting grep <model> <pattern>` counts the rows of every post-training
mix whose text contains a pattern. Exactly, over all of them — not a sample:

```
$ trainspotting grep olmo-3-7b-think "ChatGPT" --stage dpo
# grep 'ChatGPT' — 1 stage(s), 1.39 GB to read

- dpo      150,000 rows    1.39 GB  prompt/response  (allenai/Dolci-Think-DPO-7B)

scanning dpo (1.39 GB) ...
dpo: 773/150,000 rows = 0.515%
  prompt     647
  response   521
      434 /    17,596 =  2.47% of it  filtered_wc_sample_500k
      192 /     5,220 =  3.68% of it  Wildchat-1m-gpt-4.1-regeneration-not-english
      105 /    13,955 =  0.75% of it  Wildchat-1M-gpt-4.1-regenerated-english
       24 /    23,202 =  0.10% of it  ultrafeedback_cleaned_olmo2_7b
        6 /     3,884 =  0.15% of it  tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k
      … seven more sources, one to four rows each
```

That question cannot be asked of a sample. `classify` and `ask` draw 300 prompts
per stage; at 0.5% they would expect one or two matches, and at the rates the
rarer patterns come in they would expect none — with no interval around zero
saying whether the thing is absent or just rare.

The route is the datasets-server's own Parquet conversion of each repo, scanned
in place by DuckDB over HTTP range requests. Parquet is columnar, so the scan
pays for the columns it searches and skips the rest: the Think RL mix is 1.9 GB
on disk but 1.1 GB of that is the reference rollouts, which `--field prompt`
never reads. Nothing is downloaded to disk. The two cheaper routes do not work
here — `/search` and `/filter` both need a server-side index that is not built
for any Dolci repo (they answer "the dataset index is loading", or 502,
indefinitely), and `/statistics` counts whole values of a column rather than
looking inside one.

**Rows, not occurrences.** A row saying "ChatGPT" four times counts once, which
is the unit a training run sees. A row matching in two groups counts once in the
total and once in each group, so the group numbers sum to more than the total.

**Which part of the example matched.** Every column is mapped to the part of the
training example it holds, and the counts are broken down by it — the same cut
the `context` layer draws:

| Field | What it is | Columns |
|---|---|---|
| `prompt` | what the model is asked | `prompt`, the user and system turns of `messages` / `chosen` / `rejected` / `source_prompt`, a tool schema in `functions` |
| `response` | what it is fit to, or pushed between | the assistant turns of those message lists, `function_calls`, an RL mix's reference `outputs` |
| `reference` | what scores it | `ground_truth`, `reward_model.ground_truth`, `solution`, `constraint` |

This is the distinction that matters for identity text, and the one the values
layer cannot make: `classify` and `ask` read prompts, so a phrase that only ever
appears in the response or in an RL reference answer is invisible to them.

`--field` narrows the search to a subset. It saves bytes only where a column
belongs to one group: an SFT or DPO mix keeps both sides of the conversation in
one `messages` column chunk, so `--field prompt` there costs exactly what
searching all of it costs. The printed plan shows the real figure either way.

**Cost is printed before anything is read**, from the shard footers, and a plan
over 5 GB stops rather than starting a long transfer by surprise (`--max-gb`,
`--yes`). The SFT mixes are the expensive ones: 36 GB of message text for
`Dolci-Think-SFT-7B` against 1.4 for its DPO mix.

**Exactness.** The count is over every row of the revision named in the result
file, which is the Parquet branch's commit rather than a moving `main`. Two
things qualify it: a repo the server converted only part of is flagged and its
counts are a lower bound (none of the Dolci mixes are, today), and any top-level
text column the layer does not recognise as prompt, response or reference is
printed as unsearched rather than quietly skipped. Inside a message list only
`content` and the two tool subfields are read — Dolci-Instruct-DPO's `chosen`
struct also carries the WildChat request's `country`, `state`, `language` and
hashed IP, which are request metadata rather than training text and are not
searched. `tests/test_grep.py` fails if a saved schema stops mapping the way it
did, so an upstream rename cannot silently shrink a count.

Results land in `results/<model>.<stage>.grep-<slug>.json` with the per-source
breakdown and `--examples` snippets centred on the match, because a count is
only worth what reading its matches says. Of the 134 matches in
`Dolci-Think-RL-7B`, 12 are unit-test string literals asserting on
`'OpenAI ChatGPT'` — which the snippets say and the number does not. The site
does not render `grep` runs yet.

### Which stage it most plausibly came from

A count on its own is presence. Stacked across a pipeline it is easy to read
backwards, because a bigger number is not a bigger effect: a stage with ten
times the rows shows more of any string for that reason alone. So a multi-stage
`grep` ends with a comparison, and `trainspotting report` prints the same one for
every committed run:

```
### `as an AI language model` — where it most plausibly comes from

- dpo  — 61 of 150,000 rows, 0.041% (1 in 2,459).  prompt 46 · response 39 · reference not counted
  - produce side: 39 rows, 0.026% of the stage.
  - concentrated in `tulu-3-sft-coconot-regenerated`: 33 of its 790 rows, 4.2% — 103× the stage's own rate.
- rlvr — 400 of 102,014 rows, 0.39% (1 in 255).  prompt 209 · response 3 · reference 232
  - produce side: 232–235 rows, 0.23% of the stage.
  - concentrated in `hamishivi/rlvr_general_mix`: 387 of its 20,636 rows, 1.9% — 5× the stage's own rate.
- sft — not searched. That is not a zero: no row of this stage has been read for
  this pattern.
- pretrain, midtrain, long-context — out of reach for this layer, which is also
  not a zero.

Most plausibly rlvr. 387 of the 20,636 `hamishivi/rlvr_general_mix` rows (1.9%,
5× the stage) hold it, and 232–235 of the stage's matches are on the produce
side. Highest produce-side rate of the 2 stages with any: 8.7× dpo's.
```

Three things go into that, and all three are already exact:

**The rate, not the count.** Every stage against its own row count, and every
source against its own. Inside a stage the same correction applies twice over:
`llm_judged` holds 267 of Instruct-DPO's 521 `ChatGPT` matches, a clear majority
— and at 124,980 rows it holds them at 0.21% against 0.20% for the mix, so it is
the biggest source rather than the origin. A source is called a concentration
only when it holds at least a tenth of the matches at twice the stage's rate;
otherwise the line says the matches are spread, which is the more common answer
and the one a top-N list hides.

**Which side matched.** A string in a prompt is text the model was trained to
read; a string in the response a stage fits, or in the reference answer a
verifier scores rollouts against, is text the objective pushes it to emit. When
the question is why a model *says* something, only the second is evidence, so
the ranking runs on the produce-side rate and falls back to the overall rate
only when no run read a produce-side column. The source attribution follows the
same cut: under the produce-side basis the concentration is computed over
produce-side rows per source, because a source that supplied only prompts
supplied none of the evidence the ranking ran on. `by_group` counts rows per
group and one row can match two, so the union is reported as the interval it is
(232–235) rather than as the sum — bounded from below by the largest group and
by the rows that matched no prompt, which settles it outright where nothing
matched a prompt —  — and the interval is carried into the ranking
rather than collapsed to its low end. Two stages whose intervals overlap, or
touch at a single value, are not ordered by these counts, and the verdict says
so instead of picking one — which on the row basis, where every rate is a point,
is the plain tie: equal rates are reported as equal rather than resolved by the
sort. A source's own produce-side count is a union in the same way, so it prints
as `40–60 of its 1,000 rows` where its groups can overlap.

Every comparison drawn from a bounded count runs against the end that makes it
hold, or is not drawn. The advantage over the runner-up is the leader's floor
over the runner-up's ceiling and reads "at least 8.7×", and where that quotient
falls below one there is no advantage to report. A source is called the largest
contributor only when its floor clears every other source's ceiling; otherwise
the line says no largest is established and names the biggest counted floor. The
runner-up in the advantage clause is the stage with the highest ceiling rather
than the second-highest floor, because that is the one that binds how big a lead
the counts guarantee — and where that runner-up did not read all its own columns
there is no bound to state, so the multiple is dropped rather than printed
alongside a caveat that contradicts it.

A source concentration is measured over the columns its run opened, so a run
that read `response` where the mix also has `reference` gets that said alongside
the number — and the same on the row basis, where a `--field prompt` run's
concentration is over prompts alone. Where every run in a trace is limited the same way — as
every committed one is, having been written before result files recorded the
sides their mix holds — it is said once for the trace instead of under each
stage and again in the verdict.

Where a source's share rests on that interval, both tests run against the end
that makes them hold: the lift is the source's floor over the stage's ceiling,
so it reads "at least 5×" rather than "5×" and collapses to the plain figure
when the interval does, and the share floor is a share of the ceiling, since a
source only supplies a tenth of the evidence if it does so against the most the
evidence could be. Where every stage with produce-side evidence is excluded from
the ranking, the verdict elects nobody rather than the comparable stage that
measured zero there.

A run that did not read everything its own mix has on the side being ranked can
only have undercounted, so its rate is a floor rather than a figure. That is
harmless where it wins — a floor above the leader beats the leader — and unsound
where it loses, because the columns nobody opened could put it on top. Losing
stages in that position are named, and the fallback to the overall rate says
"nothing matched on the produce side of the runs that read one" rather than
"no stage matched on the produce side" whenever some run never read one.

**What was not looked at.** A stage scanned and found empty, a stage nobody
scanned, a stage this layer cannot reach, and a stage scanned but not all the
way through are four different answers, and flattening them into "no hits" is
what turns *we did not look* into *it is not there*. They are named apart.

A zero counts for the whole stage only when the whole stage was read. Three
things break that, and each is printed rather than assumed: the datasets-server
converted only part of the repo, `--field` narrowed the search to some of the
sides the mix has, or a text column the layer does not recognise went
unsearched. Any of them makes the result inconclusive — nothing matched *in what
was read* — and keeps it out of the stage-wide claim. A run written before result files
recorded which sides the mix has usually cannot demonstrate it read all of them,
so it lands there too — unless its `fields` already holds every side this layer
maps, which no narrowing could have produced and which the older RLVR sweeps
do. A pattern absent from every stage read end to end does get
said outright: 0 of N rows, exact over all of them, so a model that produces the
string anyway did not take it from those stages. What that points at depends on
what is left. While a reachable stage sits unscanned the ordinary explanation is
that the string is in it, and the verdict says so first; only once every
reachable stage has been read does it reach for a stage out of reach, text
distilled from another model rather than carried across literally, or
generalisation.

The same rule governs the ranking, because a rate only ranks against another
rate when both measure the same thing over the same population. A stage whose
run never opened a produce-side column has no produce-side rate; a stage scanned
over a partial conversion has a rate over the converted subset, which is a prefix
rather than a sample. Both keep their counts on the page and are named as out of
the ranking, rather than sorted to the bottom of it where a gap in the
measurement reads as a low score. Where that empties the ranking, the verdict
reports the matches and says no comparison is available — matches never turn
into a zero because nothing could be ranked.

A "no stage matched on the produce side" reading needs every reachable stage
read, not just every stage with a result file; short of that it is scoped to the
stages read and names the ones that were not.

Runs are grouped by their `--slug`, which is a filename rather than a promise:
rerun one stage under the same slug with a refined regex and the directory holds
two searches with one name. The group key is the slug *and* the pattern and
matching flags, so those render separately and `compare()` raises rather than
ranking one search's counts against another's under whichever pattern sorted
first. The suggested rerun commands then drop the `--slug`, because
`results/<model>.<stage>.grep-<slug>.json` is a write path and pasting the
colliding one back would overwrite the other search's saved stage.

What the ranking deliberately does not do is weight the stages against each
other. Identity behaviour is mostly set after pretraining, so the same rate in
RLVR and in Dolma 3 are not the same evidence — but by how much is not something
these counts measure, and folding a guess into a score would bury it. It is
printed as a caveat and the rates stay comparable on their own terms.

### The same question on the site

The search box in the site's header answers the reading half: it finds a string
in the committed samples and shows every match in place, across every model and
stage at once, prompts and responses alike. Type `ChatGPT` and five sampled
examples come back: the WildChat prompt that opens `Interact as ChatGPT`, a
DPO prompt asking for `a persona distinct from ChatGPT`, an RL reference answer
about `working with AI like ChatGPT`, and two long-context documents, one of
them signed `Generated by ChatGPT`. Each clicks through to the whole training
example, and each is addressable as a link
(`#search/ChatGPT/olmo-3-7b-think.dpo/row-65675`).

This is the sample, so it finds instances and never a rate: 300 rows per stage,
where `grep` reads all 150,000. The two answer different halves of the same
question, and the empty result says so — a string in none of the samples can
still be in thousands of rows.

Matching is literal, case-insensitive, and substring, the way ⌘F is. The page
does not download the 30 MB of samples to do it: `scripts/export_site_data.py`
writes `docs/data/search-index.json`, a map from every three-character run in
the samples to the files holding it, and the page intersects the query's
trigrams to learn which files are worth fetching. `ChatGPT` reads four of the
twelve; a string in none of them reads nothing at all. Trigrams rather than
words because the page matches substrings: a word index has no entry for `GPT`
inside `ChatGPT`, for `quation` inside `équation`, or for anything in a language
that writes without spaces. `tests/test_searchindex.py` pins those cases.

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
of those repos is both partial and topic-ordered. Either way no dataset is ever
downloaded. The classifier sends batches to Claude (`claude-opus-5` by default)
and records one label per prompt or document in `results/`.

## Tests

```bash
pip install -e ".[dev]"
pytest                  # offline, no API key
pytest --live           # also fetch one row per dataset from the datasets-server
```

The offline suite covers the pure code: the clustered Wilson interval and its
degenerate branches, language detection on mixed-language prompts, the
classifier's reply parser, and prompt extraction against one saved row per
registry stage (`tests/fixtures/rows/`, re-captured by
`scripts/capture_row_fixtures.py`). `tests/test_influence.py` pins the ways a
set of counts turns into a wrong story: ranking by hits rather than by rate,
adding overlapping group counts as if they were a union, ordering two stages
whose intervals overlap, naming the largest source as the origin when it holds
the mix rate, crediting a prompt-only source for produce-side evidence, ranking
an unread produce side or a partial conversion's subset rate against a stage
rate, reporting a zero for a stage that matched but could not be ranked, reading
"nothing matched" as a zero when the scan was narrowed, incomplete, or cannot
show what it covered, and reaching past an unscanned stage for a more
interesting explanation of a zero. The suggested rerun commands are built with
shell quoting, which is also pinned: these patterns are regexes, and one holding
a `$` or a backtick would otherwise run something else when pasted.
`tests/test_report_traces.py` covers the grouping the report does before any of
that: one search's stages together, two searches under one slug apart.

`grep` is covered twice over. Its column-to-field mapping runs against one saved
Parquet schema per stage (`tests/fixtures/schemas/`, re-captured by
`scripts/capture_parquet_schemas.py`), and the query it builds runs for real
against small Parquet files written in the test — locally, so the counts,
the role split, the null handling and the byte accounting are checked without a
network. Those tests need DuckDB, which `[dev]` installs.

The site's search index is tested where it can silently lose a match:
`tests/test_searchindex.py` builds a small index and asserts that a query
inside a word, inside an accented word, inside a space-free script, or made of
astral characters still keeps the file that holds it. The same cases run
against the shipped `docs/data/search-index.json` when a checkout has one.

`--live` re-runs the extraction checks against rows fetched right now, and
checks each saved Parquet schema against the current one. That is the canary for
an upstream schema change, which otherwise shows up only as a sampling run that
quietly labels nothing, or a string search that quietly counts less.

## Caveats

- The values layer classifies **prompts**. For RLVR stages the values are also
  carried by the reward, which the prompt text does not show. Where that reward
  is a constraint checker the label comes from it instead of from the prompt
  (see [Taxonomy](#taxonomy)); where it is an LLM judge the rubric is not
  published with the dataset, so those rows are still labeled from the prompt
  alone. The `sources` layer's reward-type breakdown and the `context` layer's
  verifier view are the complement.
- The stage ranking is evidence about where a string is, and only that. It does
  not weight the stages against each other, so a rate in RLVR and the same rate
  in pretraining rank equal even though the late one generally moves behaviour
  more; and a pattern present in a stage is not a demonstration that any
  particular behaviour came from it. For "did this exact document train the
  model", use OLMoTrace.
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
- `grep` reads the datasets-server's Parquet conversion of a mix, not the repo
  files themselves, and covers only the post-training mixes. The pretraining
  corpora are not converted (the same partial index that stops the sampler), so
  for an exact string count over Dolma 3 use
  [OLMoTrace](https://allenai.org/blog/olmotrace) / infini-gram.
- A `grep` count is over the text, which is not the same as over the tokens the
  model saw. A phrase split across two message turns, spelled with styled
  Unicode characters (`𝗖𝗵𝗮𝘁𝗚𝗣𝗧` is in the DPO mix), or transliterated will not
  match. Every count is a lower bound on the concept and an exact figure only
  for the pattern.
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

For a post-training stage, then run `python scripts/capture_row_fixtures.py` to
save a row for it — `tests/test_extract.py` asserts every registry stage has one,
so the new `prompt_path` and `source_columns` are checked against a real row.
