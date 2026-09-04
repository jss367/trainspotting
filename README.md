# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3) and
post-training (Dolci) data are public, and Pythia (EleutherAI), whose
pretraining corpus is public and which has no post-training at all — and, with
the same layers, any dataset on its own. The tool answers ten kinds of
question. The first five go in increasing order of depth; the sixth and seventh
are lookups rather than estimates, and differ in how much of the mix they can
see; the eighth reads a whole example rather than its prompt, the ninth puts
every stage's answer on one scale, and the tenth is the only one that opens the
model:

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
6. **Strings, in the samples** — where a given string or regex appears in the
   sampled examples, and on which side. A behaviour like claiming to be ChatGPT
   is in none of the prompts: it is in what the model is fit to. So this layer
   searches the response columns as well, and for a DPO pair says whether the hit
   is in the chosen or the rejected completion — the same string in each teaches
   opposite things. Reads the committed samples, so it finds instances to read
   rather than a rate. See [Searching a whole example](#searching-a-whole-example).
7. **Strings, over every row** — how many rows of a mix contain a given string,
   exactly, over the whole mix rather than a sample. The same question as the
   layer above with the sampling removed, which is what turns instances into a
   rate: a pattern in 0.1% of a mix is expected to miss a 300-row sample
   entirely, and no interval around zero says whether it is absent or just rare.
   Counts are then read across the pipeline as a ranking — which stage most
   plausibly taught the string, by rate rather than by hits.
   See [Searching for a string](#searching-for-a-string).
8. **Direction** — which way an example pushes on a question: `toward`, `away`,
   or `neither`. A yes/no over prompts cannot say that a stage contains training
   pointing the other way, and this data has some. See
   [Which way an example pushes](#which-way-an-example-pushes).
9. **Budget** — every stage's rate times its size, in tokens the model was fit
   to, so the stages are on one scale and add up. A share of DPO rows and a
   share of Dolma 3 documents are different denominators; this is where they
   become one number. See [How much training is that?](#how-much-training-is-that).
10. **Influence** — which of the sampled examples the model's loss on a given
   text actually moves with, by Bayesian influence function: the posterior
   covariance between the two losses, sampled by SGLD around the released
   weights. Every layer above says where a string *is*; this one says how much
   each example found there pulls the model toward saying the text, which is
   the weighting the stage ranking declines to guess at. Needs the weights and a
   GPU. See [Which examples moved the model](#which-examples-moved-the-model).

Every one of those starts from something you can already name — a string to
search for, or a question you can already phrase. When you start from an
observed behavior instead — a transcript where the model claimed the wrong
knowledge cutoff or identified as another lab's assistant — `trainspotting
trace` extracts the distinctive phrases from that text and ranks the
post-training stages by how densely each contains them, so you find the stage
without guessing a search string. See [Tracing a behavior](#tracing-a-behavior).

The same layers answer a question people ask of every open model: is the
benchmark it is scored on in the data it was trained on? `trainspotting
contaminate` cuts a probe out of each test item, reads every row of each
post-training mix for all of them in one pass, counts each in the pretraining
index, and says for every item which stage holds it and on which side. See
[Is a benchmark in there?](#is-a-benchmark-in-there).

One model also publishes the *order* its pretraining was read in. For Pythia,
`trainspotting steps` reads sampled training batches straight out of that
published stream and reports where along the run a string was seen, which is the
axis the developmental work on those checkpoints is drawn on. See
[Where in training it was seen](#where-in-training-it-was-seen).

The pretraining corpora get their own path, because the datasets-server cannot
sample them: exact composition from the shard listing, readable random documents,
and the same free-form questions. There is no context layer there — a corpus
document has no surrounding training example, it *is* the example. See
[Pretraining data](#pretraining-data).

Two more views are derived from the samples those layers commit rather than
measured by a command of their own: how many tokens each stage of the pipeline
actually is, which is the only thing that says what fraction of the model the
other five layers describe (see [Size](#size)), and which source dataset each
labeled prompt came from (see [Where each label comes from](#where-each-label-comes-from)).

Everything above reads a sample of a mix. Asking whether one *particular* text
is in a pretraining corpus needs an index instead — a whole blog is a rounding
error in trillions of tokens, so no sample will ever land on it. See
[Looking one text up](#looking-one-text-up).

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

## Base models

`pythia-12b-deduped` is a model with a pretraining stage and nothing after it.
EleutherAI built Pythia to study how a model changes *during* pretraining — 16
sizes, 154 checkpoints each, one corpus, one order — and never post-trained it.
So the layers that read prompts have nothing to read. `sources`, `classify`,
`languages` and `context` all exit saying so, `report` prints the pipeline and
stops, and the helpful/honest/harmless question this tool leads with has no
answer here. That is a fact about the model, not a gap in the audit.

What is left is the pretraining half, and it is the best-sampled one in the
registry: the deduplicated Pile is served whole by the dataset viewer, so
`pretrain` draws from all 134 million documents rather than from the head of
each shard. `ask --pretrain` scores those documents against a free-form
question exactly as it does for OLMo 3, which is the one place the two models
are directly comparable.

`find` also lines up here for the first time. Every other registered model is
searched against `v4_olmo-2-0325-32b-instruct_llama`, a stand-in for a Dolma 3
index nobody has published; Pythia has `v4_piletrain_llama`, which is the Pile
itself. The remaining gap is deduplication — that index covers the Pile as
assembled, which is what the plain Pythia models saw, while the registered
target is a `-deduped` one. `find` says so on every run.

All 8 `-deduped` Pythia sizes read exactly this corpus in exactly this order, so
adding `pythia-6.9b-deduped` or any other is a two-line registry entry pointing
at the same stages.

### Where in training it was seen

Every layer above reads a corpus as a set. Pythia is the one model in the
registry whose *order* is also public: EleutherAI released the deduplicated Pile
as GPT-NeoX tokenized and shuffled it, in the sequence the optimizer took it —
`EleutherAI/pile-deduped-pythia-preshuffled`, 143,000 steps of 1,024 sequences
of 2,049 token ids, 600 GB in 21 shards. Step *s* is a fixed 4.2 MB byte range
of that stream, so "when did the model see this string" is a range request
rather than a download:

```bash
trainspotting steps pythia-12b-deduped "OpenAI" --case-sensitive
```

draws one step from each of 64 equal slices of the run, fetches each step's
batch, decodes it with the run's tokenizer, and counts the sequences holding the
pattern. It prints the rate along the run, the same rate over eight stretches of
it, and what that rate says the model had seen by each saved checkpoint:

```
# steps 'OpenAI' — 64 sampled of 143,000 steps, 269 MB to read from EleutherAI/pile-deduped-pythia-preshuffled at 4647773

pretrain: 4/65,536 sequences hold it = 0.006% (95% CI 0.002–0.016%), 8 occurrences, 64 sampled steps
  by stretch of the run (8 slices of 143,000 steps):
          0–17,874       0/  8,192 =  0.000%  (0.000–0.047%)  8 step(s)
     17,875–35,749       1/  8,192 =  0.012%  (0.002–0.069%)  8 step(s)
     35,750–53,624       1/  8,192 =  0.012%  (0.002–0.069%)  8 step(s)
     53,625–71,499       0/  8,192 =  0.000%  (0.000–0.047%)  8 step(s)
     71,500–89,374       1/  8,192 =  0.012%  (0.002–0.069%)  8 step(s)
     89,375–107,249      0/  8,192 =  0.000%  (0.000–0.047%)  8 step(s)
    107,250–125,124      1/  8,192 =  0.012%  (0.002–0.069%)  8 step(s)
    125,125–142,999      0/  8,192 =  0.000%  (0.000–0.047%)  8 step(s)
  from about step 98,706 the run is re-reading the corpus (~207B tokens against a 300B budget)
  expected sequences holding it, seen by checkpoint (if the rate holds along the run):
    step   1,000   ~   62   (24–161)
    step  10,000   ~  625   (243–1.6K)
    step  50,000   ~ 3.1K   (1.2K–8K)
    step 100,000   ~ 6.2K   (2.4K–16.1K)
    step 143,000   ~ 8.9K   (3.5K–23K)
  step 23,237 seq 583: …page in other languages: Russian It’s been nearly two years since researchers from Google, Stanford, UC Berkeley, and OpenAI released th…
```

The result file (`results/pythia-12b-deduped.pretrain.steps-<slug>.json`)
carries the per-step counts, the rate over each stretch, the expected exposure at
all 154 checkpoints, the immutable dataset and tokenizer revisions, and up to 20
snippets with the step and sequence they were read from. `--at 1000 --at 2000`
also reads those exact steps, so the batches around a particular checkpoint can
be inspected directly; because those steps were selected rather than sampled,
they stay out of the rate, interval, slices, and exposure estimates. `--sample`
changes how many steps are drawn; `--regex` and `--case-sensitive` work as they
do in `grep` and produce distinct result filenames.

The reason to want this axis is the work that has been done on it. Timaeus's
developmental interpretability results on Pythia — stagewise structure in the
loss landscape, and influence functions that show a training example's pull on a
behaviour peaking at transitions and sometimes changing sign — are all drawn
against the step number. A corpus rate cannot be put on that axis; this can. What
it puts there is an *exposure*: expected sequences holding the string that the
model had seen by step *k*, with its interval. Whether that exposure had any
effect at *k*, and in which direction, is the model-side question those methods
answer and this tool does not.

What it can and cannot see:

- **The unit is a training sequence, not a document.** Documents are
  concatenated with no separator — there is no end-of-text token anywhere in
  these batches; one document's last sentence runs straight into the next one's
  title — and cut into 2,049-token sequences. One sequence can hold the ends of
  several documents, and a string that falls across a sequence boundary is
  missed.
- **It is a sample, so it is a curve with an interval, not a census.** Every
  sequence of a sampled step is read, so the interval is clustered by step (the
  same design-effect correction the pretraining sampler uses, with the step as
  the cluster). A string the 64 steps never land on gets an upper bound at every
  checkpoint and no first-seen step; finding the exact steps a rare string
  appears at would mean scanning the 600 GB.
- **The exposure line is straight by assumption.** It is the sampled rate times
  the sequences seen, which is only right if the rate holds across the steps not
  read. The per-stretch rates are the check: a shuffled order should show the
  same rate in every stretch. The one place a difference is expected is the
  tail — the deduplicated Pile is about 207B tokens against a 300B budget, so
  from roughly step 98,700 the run is on its second pass over documents it has
  already seen, and the command marks where that begins.

`steps` exits with a message for every other target. Ai2 publishes Dolma 3 as a
corpus, not as the sequence of batches a run took through it, so for Olmo the
question has no data behind it.

## Install

```bash
pip install -e .
```

The `values` layer needs an Anthropic API key (`ANTHROPIC_API_KEY`). The `grep`
layer needs DuckDB (`pip install -e '.[grep]'`). The `bif` layer needs torch and
transformers (`pip install -e '.[bif]'`) and downloads the model's weights;
nothing else touches a model.

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

# Weigh the committed samples against a text the model produced, by Bayesian
# influence: which examples its loss on that text covaries with (needs weights)
trainspotting bif pythia-12b-deduped --model EleutherAI/pythia-70m-deduped \
  "As an AI language model developed by OpenAI, my knowledge cutoff is September 2021."

# Combined markdown report: sampled rates with Wilson 95% CIs, and every
# committed string trace ranked by which stage most plausibly taught it
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

# Judge whole examples instead of prompts, and sign the answer
trainspotting stance olmo-3-7b-think \
  "Is this training example about caring about human lives?" \
  --slug caring-about-human-lives

# Add every stage up on one scale: tokens the model was fit to (no API key needed)
trainspotting budget olmo-3-7b-think caring-about-human-lives

# Sample documents from the pretraining, midtraining and long-context corpora
trainspotting pretrain olmo-3-7b-think --sample 300

# Is one exact string in the corpora that have a public index?
trainspotting lookup "a sentence only you would have written" --docs 10

# Re-run the committed lookup study behind the site's "is my writing in here?" tab
trainspotting case-study marginal-revolution

# Score those documents against the same question as the post-training stages
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain

# ...or only the corpora, when the post-training half is already committed
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain-only

# Exact count of the rows of each mix containing a string (needs DuckDB)
trainspotting grep olmo-3-7b-think "ChatGPT" --stage dpo

# Is GSM8K's test set in the data? 200 items, every row of every mix, one read
# per mix; then each probe counted in the closest public pretraining index
trainspotting contaminate olmo-3-7b-think gsm8k

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

The DPO view has a second level: **show how this pair updates the model** opens
the loss itself. Every chosen token contributes a term pushing its probability
up and every rejected token a term pushing its down, but those terms are summed
into one update to shared parameters — so they are what the step is made of, not
a promise about each token, and a chosen token can come out less likely once the
rest of the update lands. Only the gap between the two sides is visible to the
loss, so the panel diffs the two responses and marks three kinds of span. A
byte-exact shared opening is the only place cancellation is exact, because both
sides are conditioned on identical text there. Wording that reappears after the responses diverge follows a
different prefix on each side, so it is the same words but not the same training
signal; the panel counts it as overlap and claims nothing about cancellation.
Everything else is unique to one side, and that is what the update acts on. No
log-probabilities exist in the data and none are invented: the loss is written
symbolically and every number on the panel is a character count from the row.

The two claims that are true or false rather than approximate — this span is a
shared opening, this pair carries no gradient — are made only where the stored
row is the whole raw response: one assistant turn a side, no thinking span, and
nothing cut at 4,000 characters. Splitting a thinking span out drops the
`<think>` markers and the whitespace around them, and a cut field says nothing
about what followed, so neither can be compared with the sequence the model
actually scored. Everywhere else the panel reports overlap and character counts
and says which of those reasons applies.

A multi-turn pair branches somewhere and shares every turn before that, its
assistant turns included. Those are the conversation both candidates answer in,
not either candidate, so the view shows them once under their own heading and
diffs only the continuation — the same split `_shared_turns` makes in
`trainspotting/search.py`. Twelve of the 900 sampled pairs carry shared
assistant history, and it accounts for 65,414 characters that would otherwise be
counted as response wording on both sides at once.

That covers most of the data. Of 900 sampled pairs, 328 have a continuation that
opens identically, but only 72 can be called a shared opening: 231 have thinking
spans in front that differ, 13 reach a turn the record does not hold as written
so where the two completions part company is a guess, and 12 have a candidate
cut at 4,000 characters. Four pairs, all in
Dolci-Instruct-DPO, are byte-identical on both sides and therefore carry no
gradient at all. Beyond the openings, the two responses have very little wording
in common — a median of 1–3% of the chosen response — so for most pairs the
update acts on nearly every token of both.

Whether a turn is stored as written is recorded by the exporter, not guessed at
by the site: `context` writes `raw: true` on a turn whose stored text is the
content itself. Nothing observable in the record proves that otherwise — a
thinking span leaves its `<think>` markers and surrounding whitespace behind
even when the span is empty and no reasoning field remains, and a cut field says
nothing about what followed it. A context run made before that flag existed
carries none, so the site withholds every exactness claim on it until it is
regenerated.

Two smaller things the claim depends on. Character counts are code points, to
match the `chars` values the exporter writes with Python's `len()`; counting
UTF-16 units instead would double every emoji, and 107 of the 900 rows contain
one. And a shared opening stops before the whitespace separating it from the
divergence, since tokenizers usually attach that space to the word after it,
which differs.

`tests/site/gradient_panel.test.mjs` holds the panel to this, checking every
committed DPO row (run by `pytest` via `tests/test_gradient_panel.py`, skipped
if node is missing).

This matters for reading the numbers. A prompt like *"Write a program to decide
if a child should be saved based on race"* counts as harmlessness content, and
only the pair behind it shows the model is trained toward refusing it.

Registered models: `olmo-3-7b-instruct`, `olmo-3-7b-think`, `olmo-3-32b-think`.

## Size

Every layer above answers a question about post-training, which is a fraction of
a percent of the text the model read: **0.03%** for `olmo-3-7b-instruct`, and
**0.31%** and **0.34%** for `olmo-3-7b-think` and `olmo-3-32b-think`. The think
models are an order of magnitude higher for one reason, and it is worth seeing on
its own: their SFT examples carry reasoning traces, so the same 2-odd million
examples are ten times the text.

None of those numbers is published anywhere. Ai2 gives token counts for the
pretraining mixes and row counts for the post-training ones, and 2.15M examples
and 5.93T tokens are not comparable quantities.

`scripts/export_site_data.py` closes the gap from the samples already committed.
For every context run it writes `docs/data/<target>.<stage>.profile.json` —
lengths, how much of each example the model is fit to, one metadata row per
sampled example, and a token estimate: the sampled mean characters per example
divided by four, times the exact row count. For every pretraining document sample
it adds the same length summary to `<target>.<stage>.corpus.json`. Both are
derived by `trainspotting/derive.py` from files the site already ships, so a
fresh checkout rebuilds them with no network and no API key.

The site draws four things from that:

| View | What it says |
|---|---|
| **Where the token budget went** | Every stage on one strip to scale, and again on a log axis. Corpus tokens are the paper's; post-training tokens are estimated, with the 95% interval on the sampled mean. A second strip shows what *this page* sampled per stage — roughly equal everywhere, which is the inverse of the first strip. |
| **How much of it the model is fit to** | The gradient-bearing share per stage: all of pretraining, the assistant turns of an SFT example, both completions after a DPO pair branches, and none of an RL row — the response there is generated during training and never stored. |
| **How long is one example?** | Characters per example per stage on shared half-decade bins, which is what makes an example count and a token count the same kind of statement. |
| **The whole pipeline as area** | A treemap where area is tokens: Common Crawl against FineMath against the whole of post-training in one frame. Boxes whose true area is under a pixel are drawn at 3px and the card says how many. |

The estimate's weak part is the divisor, and it is one number
(`derive.CHARS_PER_TOKEN`). Real tokenizers run about 3.5 characters per token on
code and 4.5 on English prose; nothing in that range moves a finding that is a
factor of ten thousand.

## Where each label comes from

The taxonomy says how much of a stage is about being harmless. The mix
composition says which datasets the stage was built from. Neither says which of
those datasets the harmless prompts came from — usually one or two of them.

The site crosses the two: rows are the values of the source column a stage's rows
carry (`source_dataset`, `dataset_source`, `preference_type`, or whatever else
the mix records), columns are the seven labels, and every cell opens the prompts
behind it like any bar does. A second grid behind a fold does the same against
detected language. On Dolci Instruct SFT this is how you find that every sampled
prompt from `Verifiable Reasoning` is capability content while `Wildchat` is
mostly helpfulness — a split no stage-level share can show.

The join costs no payload. A profile record carries a 32-bit FNV-1a hash of the
same 400-character prompt opening a context record is keyed on, so the crossing
happens in the browser over files it already has.
`tests/test_derive.py` runs the Python and JavaScript implementations of that
hash over the same inputs, because a drift between them would empty the grid
rather than raise anything.

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

`search` and `grep` ask the same question of different amounts of data, and the
difference is the whole point of having both. `search` reads the committed
samples, so it hands back examples to read and cannot report a rate: 300 rows
per stage is too few to distinguish absent from rare. `grep` reads every row, so
it reports a rate and a zero that means absence, and cannot hand back more than
the examples it kept. Start with `search` when the question is "what does this
look like", and `grep` when it is "how much of the mix is this".

Their side classifications are separate implementations — `search.FIELDS` over
sampled records, `grep.MESSAGE_LISTS` over Parquet columns — because they read
different shapes: one has parsed JSON records, the other has a column schema and
a SQL dialect. That is a duplication worth watching rather than one worth hiding:
if the two ever disagree about which side a turn is on, the same string gets
counted as prompt by one and response by the other.

## Searching for a string

`trainspotting grep <model> <pattern>` counts the rows of every post-training
mix whose text contains a pattern. Exactly, over all of them — not a sample:

```
$ trainspotting grep olmo-3-7b-think "ChatGPT" --stage dpo
# grep 'ChatGPT' — 1 stage(s), 1.39 GB to read

- dpo      150,000 rows    1.39 GB  prompt/chosen/rejected  (allenai/Dolci-Think-DPO-7B)

scanning dpo (1.39 GB) ...
dpo: 773/150,000 rows = 0.515%
  prompt     647
  chosen     342
  rejected   388
      434 /    17,596 =  2.47% of it  filtered_wc_sample_500k
      192 /     5,220 =  3.68% of it  Wildchat-1m-gpt-4.1-regeneration-not-english
      105 /    13,955 =  0.75% of it  Wildchat-1M-gpt-4.1-regenerated-english
       24 /    23,202 =  0.10% of it  ultrafeedback_cleaned_olmo2_7b
        6 /     3,884 =  0.15% of it  tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k
      … six more sources, one to four rows each
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
| `response` | what it is fit to | the assistant turns of `messages`, plus `reasoning_content` and a turn's tool calls |
| `chosen` | what a pair is pushed toward | the chosen completion's turns, from the point the pair branches |
| `rejected` | what a pair is pushed away from | the rejected completion's turns, from the same point |
| `reference` | what scores it | `ground_truth`, `reward_model.ground_truth`, `solution`, `constraint` |
| `rollout` | what was scored, and not trained on | an RL mix's reference generations in `outputs` |

Which groups a stage has depends on its shape, and `--field` names groups rather
than columns: a DPO mix has `chosen` and `rejected` and no `response`, so
`--field response` there selects nothing. The printed plan names the groups the
stage actually offers before anything is read.

A pair's two sides are separate groups because they teach opposite things.
Counting a string under one `response` heading adds a completion the objective
pushes the model toward to one it pushes away from, and `rollout` is split off
the same way: a reference generation is what the verifier scored, not text the
model was fit to.

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
first. The suggested rerun commands then carry a free slug rather
than the contested one, because `results/<model>.<stage>.grep-<slug>.json` is a
write path and pasting the shared one back would overwrite the other search's
saved stage. Dropping the flag is not enough: two searches differing only in
`--regex` or `--case-sensitive` share a pattern, and that is what `grep` derives
a filename from. Collision is judged per slug, so an uncontested one keeps its
own. The free slug travels with the renames that make it work: `_grep_traces`
groups by slug, so scanning the missing stage under a new one without moving the
existing files just opens a third group. The report names the renames and leaves
them to the reader rather than rewriting `results/` itself; the filename is then
the authority for a run's slug, since `--slug` is what decides the filename and a
moved file still carries the contested one in its payload.

What the ranking deliberately does not do is weight the stages against each
other. Identity behaviour is mostly set after pretraining, so the same rate in
RLVR and in Dolma 3 are not the same evidence — but by how much is not something
these counts measure, and folding a guess into a score would bury it. It is
printed as a caveat and the rates stay comparable on their own terms.

### The same question on the site

The search box in the site's header answers the reading half: it finds a string
in the committed samples and shows every match in place, across every model and
stage at once — every turn of the prompt, system instructions included, and the
response side with it. Type `ChatGPT` and five sampled
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

## Is a benchmark in there?

`trainspotting contaminate <model> <benchmark>` asks whether the test items a
model is scored on were also in what it was trained on — and, because the
answer is a different finding on each side of an example, where and on which
side.

```
$ trainspotting contaminate olmo-3-7b-instruct gsm8k --stage dpo
# contaminate GSM8K (openai/gsm8k main/test) on olmo-3-7b-instruct
# 200 of 1,319 items (seed 0), 13-word probes: 200 question, 200 answer

# 1 stage(s), 799 MB to read
- dpo      259,922 rows     799 MB  prompt/chosen/rejected  (allenai/Dolci-Instruct-DPO)

scanning dpo (799 MB) ...
dpo: 0/200 items seen, 0 rows  -> results/olmo-3-7b-instruct.dpo.contam-gsm8k.json

counting 400 probes in v4_olmo-mix-1124_llama ...
  -> results/olmo-3-7b-instruct.corpus-v4_olmo-mix-1124_llama.contam-gsm8k.json

dpo    items seen 0/200 = 0.0% (95% CI 0.0–1.9%)
  question in a prompt: 0
  answer in produced text: 0
  rows: 0 of 259,922
sft    not scanned — not a zero
rlvr   not scanned — not a zero
corpus items seen 9/200 = 4.5% (95% CI 2.4–8.3%)  in v4_olmo-mix-1124_llama
  occurrences over all probes: 41
  note: The public infini-gram API has no Dolma 3 / OLMo 3 index, so this count is
  over a different corpus than the one this tool samples — the closest available,
  not the one the registered models were trained on.
```

Read together, those two lines are the point of keeping the sides apart. Not one
of the 200 items is in the Instruct DPO mix, on any side, over all 259,922 rows.
Nine of the 200 are in OLMo 2's pretraining crawl — 9 of the 400 probes, all of
them questions and none of the worked answers, 41 occurrences, at most 32 for
one probe — which is what a benchmark that has lived on GitHub, in papers and in
tutorials since 2021 looks like from inside a filtered web crawl: present, and
rare. The first is a statement about what Ai2 put in front of the model; the
second is a statement about the ecosystem's text, over a corpus that is not
Dolma 3. Neither stands in for the other, and the SFT and RL stages are named
as unscanned rather than counted as clean. The same rule holds inside the corpus
count: a probe the index did not answer — a rejected query, or one that ran out
of retries — is recorded as an error rather than as zero occurrences, the
summary says how many there were, and an item left with an unanswered probe and
no hit is out of the rate rather than in it as clean.

Which index stands in for pretraining is not a detail. The same 400 probes
counted in `v4_olmo-2-0325-32b-instruct_llama` — OLMo 2 32B's *full* training
data, pretraining through Dolmino midtraining and Tulu 3 post-training — find
all 200 items: 253 probes, 53 of them worked answers, 514 occurrences
(`results/olmo-3-7b-instruct.corpus-v4_olmo-2-0325-32b-instruct_llama.contam-gsm8k.json`,
kept for the comparison). What separates the two indexes is the midtraining and
post-training data, so 191 of these 200 test items are in what OLMo 2 saw after
pretraining and not in its crawl — and read against the full index, the corpus
line would have reported that as a web finding. `contaminate` therefore defaults
to the pretraining-only index, and a run against any other names it in the
result's filename so the two cannot overwrite each other.

**What is searched for.** Not the item whole. Copies of a test item in training
data routinely differ from the original — re-wrapped, prefixed with
"Question:", a calculator annotation stripped — and an exact search for the
full text would miss them and report a clean stage. Each item is cut to a
**probe**: thirteen consecutive words from its middle, matched with any
whitespace between the words and regardless of case. Thirteen words is the
window GPT-3's contamination analysis used, and it is long enough that a chance
match is rare. The middle rather than the start because the start is where
copies differ. An item with fewer than eight words is not probed and is counted
as such rather than as clean, because a short window is where a chance match
happens. Where the benchmark has a worked answer (GSM8K, MATH-500, HumanEval),
the answer gets a probe of its own. Where it is multiple-choice (MMLU, MMLU-Pro,
ARC-Challenge) the options are stored apart from the stem and get a probe of
their own too, as a `choices` part: many MMLU stems are under the eight-word
floor and the options are the only probe-able text, and a stem generic enough to
recur is not a copy of the item where its options are not. The options are
joined by newlines with no letter prefixes — copies disagree on "A." versus
"(A)" — so a window that spans two options misses a prefixed copy; a hit on the
options counts as the question being read, since they are what the model is
shown rather than the letter it is scored on.

**The side is the finding.** A question probe in a prompt column is the model
being trained to *read* the test item. An answer probe in a response or chosen
column is the model being trained to *produce* it. An answer probe in an RL
row's reference is what the verifier scores rollouts against — a claim about
the reward, which no completion need contain — so it gets its own line, "answer
in a verifier reference", rather than counting as produced text. The summary
keeps all of those apart, and a question restated in produced text is its own
line too, so a model that echoes the question before answering cannot inflate
the produced count. A hit only in a rejected completion is reported as seen but
neither read nor produced.

**One read per mix.** Two hundred items is up to four hundred probes, and a mix
costs gigabytes to read, so the probes share a scan: they are searched as
alternations, `(?:probe|probe|...)`, and the strings that matched travel back
whole so each probe in them can be found and credited — position by position,
because a regex engine's "all matches" are the non-overlapping ones, and two
near-duplicate items whose windows sit a word apart would otherwise count as
one. Not one alternation, though. RE2 runs an alternation as a DFA while its state cache fits
and falls to an NFA when it does not, and the fall is a cliff — four hundred
probes in one regex ran fifty times slower than the same probes split four ways.
`contamination.CHUNK_CHARS` is that split, by pattern length rather than probe
count, since the states are made of characters and a HumanEval probe of long
identifiers costs more of them than a prose one.

**The corpus side.** Each probe's literal text — as written, whitespace and case
intact — is counted in the infini-gram index the registry names as closest to
the model's pretraining data. For OLMo 3 that is OLMo 2's pretraining-only index,
`v4_olmo-mix-1124_llama` — a different corpus, and the run says so — and not the
full-training-data index `find` defaults to, which folds in Dolmino and Tulu 3
and would hand back another model's post-training as corpus; for Pythia it is
the Pile itself. The count is
occurrences rather than documents, and a benchmark's presence on the web (GSM8K
is on GitHub, in papers, in a hundred tutorials) is what it measures. It is a
statement about the ecosystem's text, and the post-training scans are the
statement about what Ai2 put in front of the model.

**The interval is over the benchmark.** The items are a seeded draw from the
test set (all of it when `--items` covers the set), so the share found estimates
the share of the whole benchmark that is present, with a Wilson interval. The row
counts are exact: every row of the mix was read. When the draw is the whole split
there is nothing to estimate — the share is the benchmark's, printed as
`k/n = x% (every item)` with no interval. When the server converted only part of
a mix, a miss is not a known miss, so the share is printed as a floor, `≥ k/n`,
with no interval either.

Two result files per run. `results/<target>.<stage>.contam-<slug>.json`, where
the slug is the benchmark id at the default settings and `<benchmark>-<hash>` of
the settings otherwise, so a narrowed run cannot overwrite the full one,
carries every probe, every (probe, side) row count, the items rolled up by claim,
the source breakdown of matched rows, and hash-ordered example snippets; the
`corpus` file carries every probe's occurrence count. `--stage`, `--field`,
`--items`, `--words`, `--case-sensitive`, `--index`, `--no-corpus` and
`--corpus-only` narrow it; `--slug` names the files instead. The byte cap and
`--yes` are `grep`'s.

Registered benchmarks: `gsm8k`, `math-500`, `humaneval`, `mmlu`, `mmlu-pro`,
`arc-challenge`, `truthfulqa`, `ifeval`. Adding one is an entry in
`trainspotting/benchmarks.py`: the repo, config and split, the question field,
and the answer field if there is one long enough to probe. The site does not
render these runs yet.

## Which way an example pushes

`ask` judges prompts and answers yes or no. For a values question that is the
wrong half of the example and the wrong shape of answer.

The wrong half, because a value lives in what the model is fit to produce.
"This prompt is about human lives" describes the request; whether training on
the example teaches the model to value human lives is a claim about the
response, and for a preference pair it is a claim about *which* response.
`search` reads that half but only matches strings.

The wrong shape, because yes/no cannot represent training that points the other
way, and this data contains some — the anti-vaccine RLVR row under
[Taxonomy](#taxonomy) is a `yes` under `ask` and teaches the opposite.

```bash
trainspotting stance olmo-3-7b-think "Is this training example about caring about human lives?" \
  --slug caring-about-human-lives
```

reads the committed context records — the whole example behind every sampled
prompt, no re-fetching — renders each one marked up by the role its parts play
in training, and labels it `toward`, `away`, or `neither`. The headline is the
net, `toward − away`.

| Stage | What the judgment reads | What `away` means there |
|---|---|---|
| SFT | the prompt and the assistant turns the model is fit to | the target response itself cuts against the question |
| DPO | the shared prefix, then both completions marked preferred / dispreferred | the *dispreferred* completion is the one that serves the question |
| RLVR | the prompt, the verifier and what it checks, and the pass rate | the reward pays for output that cuts against the question |

An RL row's stored reference generation is deliberately left out of that. The
schema records a row's `outputs` and an aggregate `total_correct_rollouts` with
nothing tying the two together, so whether the stored one passed is not
recoverable — and shown beside the verifier, a generation that failed reads as
the behaviour training paid for, which inverts exactly the answer this layer
exists to give. It stays in the context record, where the site shows it as a
reference generation rather than as evidence of what was rewarded.

The markup is the instrument, so it has to survive the character budget. Each
part of an example is cut to its own share rather than the joined text being cut
once at the end: a single excerpt over the whole thing sliced straight through
`[DISPREFERRED — training pushes away from this]` on 16 of 300 sampled
Dolci-Think-DPO pairs, and a pair whose side marker is gone still looks
well-formed while reading exactly backwards.

A `chat` target is refused rather than judged. Nothing was fit to a log, so it
has no direction — the same reason the context view marks no turn in one as a
target.

## How much training is that?

Every layer above reports a share of something: rows of a post-training mix,
documents of a corpus. Those denominators are not each other, and none of them
is an answer to "how much training did the model get". Olmo 3 7B sees 5.93T
pretraining tokens; 6% of Dolci-Instruct-DPO and 1% of Dolma 3 are three orders
of magnitude apart in what they represent.

```bash
trainspotting budget olmo-3-7b-think caring-about-human-lives
```

multiplies each stage's rate by its size and adds them up. No API key, no
network — it only reads runs that already happened.

```
stage          fit tokens         by row  by length    matching tokens
----------------------------------------------------------------------
pretrain             5.9T   not measured
midtrain             100B   not measured
long-context          50B   not measured
sft                 17.7B    3/300   1.0%       0.1%  21.5M (7.3M–62.2M)
dpo                  643M   14/300   4.7%       2.2%  14.4M (8.6M–23.7M)
rlvr                54.9M*   8/300   2.7%       3.3%   1.8M (924K–3.5M)

post-training        18.4B fit tokens  →  37.7M matching  (0.20% of it)
whole pipeline        6.1T fit tokens  →  37.7M matching  (0.00062% of it)  [3 stage(s) not measured]
```

### The unit

**Tokens the model was fit to.** That is the only unit under which the stages
mean the same thing, and it is not the size of the dataset:

| Stage | Fit to |
|---|---|
| pretrain / midtrain / long-context | every token — the stage size from the Olmo 3 paper is the answer |
| sft | the assistant turns, reasoning spans included, not the prompt they read |
| dpo | both completions past the branch point; the preference loss is computed over the pair |
| rlvr | rollouts generated during training, which the dataset does not contain |

Both sides of a pair store the whole conversation, so the split is at the point
the two actually diverge, not by role — `context.branch_point`, the same line
`search` draws. An assistant turn before the branch is shared history the pair
is judged in, and counting it once per side charges the stage twice for text
neither completion was preferred for. That is 12 of the 300 sampled
Dolci-Instruct-DPO pairs and 5.9% of that stage's fit characters; the think
mixes are single-turn throughout, so nothing there moves.

RLVR is the honest gap, and the table marks it `*`. The published mix holds
prompts, verifiers and some reference generations, not the text the policy was
fit to, and how many rollouts per prompt the run took is not in the data. What
the table reports is a **floor**: one reference rollout per prompt.

Post-training sizes are the exact row count from a `sources` run times the mean
fit length of the sampled examples, at four characters per token. This is where
the think models diverge from Instruct: Dolci-Think-SFT is 17.7B fit tokens to
Dolci-Instruct-SFT's 622M, almost all of it reasoning traces.

### The rate

Which weighting is right depends on how the stage was sampled, and the two
halves of the pipeline are sampled differently. Each stage records the rule it
used in `weighting`.

**Post-training rows are drawn uniformly**, so they are weighed by fit
characters. A stage's matching examples are not average-length: in
Dolci-Think-SFT the three matching examples are short, 1.0% of rows and 0.1% of
the text. Counting rows there answers "what fraction of examples" when the
question is "what fraction of training".

**Corpus documents depend on the route they were drawn by**, which is the point
worth being careful about: being a corpus is not what decides the correction,
how the documents were sampled is. The route travels with the run — the document
sample records it and the `ask` run copies it out of the sample — so pointing a
stage at the other route later leaves stored results weighed the way their own
draw earned, and asks to be re-sampled rather than reinterpreted.

*Shard-drawn corpora are weighed by nothing extra.* `trainspotting pretrain`
draws shards with probability proportional to compressed size and takes one
document from each, precisely so the source mix comes out token-weighted. Under
that design every sampled document stands for the same byte mass — a stratum
holding twice the bytes wins twice as many shard draws, so it yields twice as
many documents — which makes the plain document rate the byte-weighted rate
already. Weighing it by each document's own length would apply the size
weighting a second time, and a 200k-character Longmino PDF would count a
hundred web pages' worth against a stratum holding exactly as many bytes.

What that leaves is a residual *within* a shard, not between shards: the sampler
picks uniformly among a shard's reachable documents rather than in proportion to
their length, so a long document is slightly underweighted against its byte
share. One document per shard gives nothing to estimate that shard's mean length
from, so it stays a caveat rather than a correction made badly.

*Rows-drawn corpora are weighed by fit characters*, exactly as post-training
rows are. The `rows` route pages the datasets-server uniformly over every
document in the corpus, so nothing in the draw is proportional to size and the
document rate is a share of documents rather than of training. The deduplicated
Pile makes that concrete: its 300 sampled documents run from a few hundred
characters to seventy thousand, so which end of that range the matches land in
moves the matching-token total by more than the rate itself does.

The interval is the count-based one — cluster-corrected for corpora, where the
`ask` run already stored it — rescaled by the weighed rate over the count rate,
which is exactly 1 for a shard-drawn corpus stage. It is computed over the rows the point
estimate was actually built from, which for an RL stage is much smaller than the
sample: Dolci-Instruct-RL stores a reference generation for 60 of 300 judged
rows, and taking the interval over all 300 claimed five times the evidence there
is — a 5.6% upper bound where the honest one is 13.7%. What it still does not
carry is the extra uncertainty in the length ratio, so it is narrower than the
truth by that much, and the output says so whenever the reweighting rests on
fewer than ten matching examples.

A stage whose ask run labeled nothing, or whose rows join to no stored example,
reports no rate rather than a rate of zero. Absent evidence with a zero-width
interval on it is the one reading of that case that is worse than a gap.

### What it does not do

It weighs tokens, not learning. A DPO preference token, an RL gradient step and
a pretraining cross-entropy token do not move a model equally, and post-training
is widely believed to be far higher-leverage per token than pretraining. Nothing
here corrects for that: read the output as an exposure budget, not as an
attribution of behaviour.

A stage the question was never asked of prints `not measured` rather than
dropping out. A total that silently excluded 5.93T tokens of pretraining is the
error this command exists to prevent.

What the pipeline share claims depends on whether every stage could be sized,
and the CLI and the site say which of three things it is.

With every stage sized and some unasked, it is a **lower bound**, printed as
`at least`. The denominator is every sized stage, measured or not, so with only
post-training scored the figure is matching tokens over 6.1T — not a share of
the 18.4B that was actually read. Those differ by three orders of magnitude, and
asking the corpora can only add matches.

With any stage **unsized** — no `sources` count, or no stored examples to take a
mean length from — it is neither. `totals()` drops an unsized stage from the
denominator *and* the numerator, so sizing it later moves both, and if its own
rate is below the aggregate the share falls. That case prints as `N% of the X
that could be sized`, with no `at least`.

### A worked question

`scripts/human_life_value.sh` runs the whole battery behind one question — how
much training does an Olmo 3 model get in learning that human lives are valuable
and important?

```bash
scripts/human_life_value.sh olmo-3-7b-think           # everything
scripts/human_life_value.sh olmo-3-7b-think pretrain  # just the corpora
scripts/human_life_value.sh olmo-3-7b-think budget    # just the rollup (free)
```

It samples the corpora, scores them against the umbrella question with
`--pretrain-only` (the post-training half is already committed, and re-running
it would pay for nine stages to learn nothing new), asks five sub-questions
across both halves — including the mirror, *would fitting this teach the model
to disregard human welfare* — runs `stance` over the post-training examples, and
rolls every question up with `budget`. The question wordings live in that script,
which makes them one editable instrument rather than nine copies in a shell
history.

## Tracing a behavior

`trace` is the way in when you have a behavior, not a query. Most of the tool
assumes you already know what to look for; `trace` starts from what the model
did. Paste the text — a transcript, a description, the sentence that surprised
you — and it pulls the distinctive phrases out of it and counts how many rows of
each post-training stage contain each one across the split (the
datasets-server full-text index, nothing sampled and nothing downloaded). It
ranks the stages by matches per million rows, so `"As an AI language model
developed by OpenAI"` lands you on whichever mix carries the most of that
provenance rather than leaving you to guess a `grep`. The index reaches the
assistant turns, not just the metadata beside them: it covers string values
nested inside a struct or a list of structs, which is what an SFT `messages`
column and a DPO `chosen`/`rejected` column are.

A window is kept only if it is anchored: on a number, on a capital past a
token's first letter (`ChatGPT`, `OpenAI` — English does not shape words that
way, so those are names wherever they sit, including the opening word of a
transcript), or on a capital anywhere but the start of a *sentence*. After a
colon is not the start of a sentence, since nothing there forced the capital, so
`"Assistant: Claude"` yields `Claude` while `"Weather today is mild"` still
yields nothing. The anchor
is what makes a query selective, not the length, so a window of pure function
words is dropped however long it is — it would match training rows by
coincidence — while `"Assistant: ChatGPT"` yields `ChatGPT`. Boundary function
words are trimmed off the windows that are kept, because search ANDs a query's
tokens together and a trailing "so I" only excludes the row that phrased the
same span without it.

Two anchors in one sentence get two queries: a candidate window is dropped only
when every anchor in it is already covered, not merely because it overlaps one
already chosen. Otherwise `"...developed by OpenAI, my knowledge cutoff is
September 2021."` — fourteen words, so every eight-word window touches every
other — would spend its whole budget on the lab and never reach the date.

Three things a `trace` number is not, all printed next to it:

- **Not the phrase.** The server stems each token and ANDs them, so a hit is a
  row holding all of the query's words, not the literal string — an upper bound
  on verbatim occurrences.
- **Not a side.** It counts rows, not which half of the example the string is
  in. A run ends with a link into the dataset viewer, whose `?q=` runs the same
  index, so the rows behind the count are one click away. Deliberately not a
  `trainspotting search` command: `search` is the layer that attributes a hit to
  a side, but it does it over a 300-row random draw, and at the densities a
  signature string produces (100/M is a 3% chance of one hit in that draw) it
  would answer with a confident zero.
- **Not always the whole split.** The full-text index stops at the first 5 GB,
  which the two 36 GB Think SFT mixes are well past. Their matches come from a
  prefix of the rows they are divided by, so they are reported separately as
  lower bounds rather than ranked. A bound settles one comparison — it is
  conclusively above a ranked stage whose exact density is smaller — and nothing
  else, and ranking the two together put the biggest mixes last for being big.

The first search against a cold split can take minutes while the server builds
the index — the widest window in the tool for `main` to move under a run. Each
stage's revision is read before its row count and again after its searches, and
a stage that moved in between is reported outside the ranking, like a partly
indexed one but for a stronger reason: the two halves of its ratio describe
different trees, so unlike a lower bound it is not a loose estimate of the
density but not an estimate of it at all. Its counts are printed, its density is
not ranked, and it cannot take the viewer link. When the behavior has no signature string — it is a disposition, or
the training paraphrases it — `trace` finds nothing and says to reach for `ask`,
which judges what sampled examples *teach* instead of matching their text
(`"does this example teach the model to identify as ChatGPT?"`). All three
compose, on different scales: `trace` narrows to a stage over the whole split,
`search` says which side of the example a string lands on over a sample of it,
and `ask` characterizes the fuzzy cases neither can match.

## Which examples moved the model

Everything above counts. `grep` says how many rows hold a phrase, the stage
ranking puts those counts on a rate, and then stops with a caveat: a rate late
in the pipeline generally moves behaviour more than the same rate in
pretraining, but by how much is not something counts measure. `bif` measures it,
for the examples the other layers already have.

It uses the Bayesian influence function (Lau, Wang, Baker, Murfet and Hoogland,
2025). Around the released weights `w*` there is a local posterior,
`p(w) ∝ exp(-nβ·L(w) - γ/2·‖w - w*‖²)`, where `L` is the mean loss over the
examples the posterior is localized on. Upweighting one of those examples by `ε`
moves the posterior, and the derivative of the expected loss on a query with
respect to `ε` is `-nβ · Cov(ℓ_query, ℓ_example)` under it. So the number
reported per example is that covariance, sampled by SGLD: positive means training
harder on the example would lower the loss the model assigns to the query, which
is to say the example pulls the model toward saying it. No Hessian is formed or
inverted, which is what lets this run on a model rather than a toy.

```bash
pip install -e '.[bif]'
trainspotting bif pythia-12b-deduped --model EleutherAI/pythia-70m-deduped \
  "As an AI language model developed by OpenAI, my knowledge cutoff is September 2021."
trainspotting bif olmo-3-7b-instruct --match "ChatGPT" --limit 100 --dtype bfloat16 \
  --prompt "Who are you?" "I am ChatGPT, a large language model trained by OpenAI."
```

The query is the text the model produced; `--prompt` is what it was replying to,
scored as context and masked out of the loss, and required for a chat model,
whose template never showed it a reply without the turn it answers. The candidates are the committed
samples: the context records of each post-training stage and the document
sample of each pretraining corpus. An SFT example is fit on its assistant turns
behind the rest of the conversation, rendered through the model's chat template
so the tokens are the ones training saw; a think model's reasoning, which the
context record stores beside the answer, is put back inside its `<think>`
markers first, since the model was fit to both and the reasoning is most of it.
A DPO pair yields *two* candidates, its chosen and its rejected completion,
because the objective pushes the model off the rejected text. The turns the two
share before they branch are the conversation the pair was judged in, so an
assistant turn in that shared history is context on both sides, as it is for
the fit share above. The posterior is localized on the text the model was fit
*toward* — documents, SFT responses, chosen completions — and the rejected
completions are scored at every draw but never drawn into an SGLD minibatch,
since fitting the chain to them would localize it on the opposite of what
training did (the result file's `localized_on` counts the fit side). Their
covariance is still read: a positive covariance on a rejected completion is
evidence the pair taught the model away from the query, where the same number
on the chosen side is evidence it taught it toward. That is a reading of the
loss covariance, not the influence function of the pairwise DPO objective,
which would also need the reference model's log-ratio. A corpus document is all
target.
An RL row stores no response and is skipped, and the skip is printed with its
reason, as is a stage with no committed sample or one whose sample is of a
different mix than the stage now names. A record with tool use (a function
menu on the system turn, a function call in place of an answer) is skipped
and counted, because the context record does not hold those fields in the
form the model was trained on; 60 of the 300 Instruct SFT records are such. So
is a record whose fit turn was cut at the context record's 4,000-character
field limit: a think turn's reasoning closed early with the answer appended is
a sequence the model never saw, where a cut document is still a prefix of one.
That costs the think mixes most of their sample (229 of the 300 Think-7B SFT
records have cut reasoning), and a fuller `context` store is what would give
it back. `--match` keeps only candidates
whose text holds a regex, which is how a phrase `grep` found becomes the set of
examples to weigh; `--limit` caps the records per stage so a run on a big model
fits the machine, and a DPO pair is one record, so a limit never keeps a
rejected completion without the chosen one it was scored against.

The committed run (`results/pythia-12b-deduped.bif-as-an-ai-language-model-…json`)
is the query above against Pythia-70m and 200 sampled Pile documents, and shows
what a first result looks like: every one of the 200 covariances is positive,
because every loss moves with the chain's excursion, and the partial
covariances beneath them split 101 toward to 99 away, the strongest two and a
half standard errors from zero. A 70-million
parameter model and a seventeen-token query are a small signal; the point of
the committed file is that the machinery, the diagnostics and the shape of the
answer are there to run on a bigger one.

The result file (`results/<target>.bif-<slug>.json`) records the checkpoint and
its commit, every sampler setting, the per-example covariance with its
across-chain standard error, and the correlation beside it, since covariance
scales with how much a loss moves and a long high-entropy document moves more
than a short answer whether or not it has anything to do with the query. The
report prints the stages ranked by mean covariance, the examples at either end
of the ranking, and two checks on the sampler that come free from the same
draws: the local learning coefficient `nβ(E[L] - L(w*))`, taken over the
candidates the posterior was localized on, which is at or below zero when
the chains sat at a lower loss on that sample than `w*` does (possible
without fault, since `w*` minimizes the training set and not a few hundred
sampled examples; the report says so rather than calling the run invalid),
and the drift of each chain's minibatch loss across its
retained steps, which catches the failure the coefficient alone does not — a
chain climbing steadily away from `w*` has a positive coefficient and a doubled
loss. A run at ten times the default step size did exactly that on Pythia-70m,
and reported every candidate as positively correlated with the query for no
better reason than that everything got worse together. The report says so
when it happens and says to lower `--lr`.

That shared movement never goes to zero, and at the default step size it is
most of every covariance: on Pythia-70m at twice the default step size all 200
documents correlated near 0.85 with the query, at the default near 0.6, and the
raw ranking was by how far each document's own loss swings rather than by
anything to do with the query. The ranking is still by the covariance, because
the covariance is the influence: the identity above is about
`Cov(ℓ_query, ℓ_example)` and nothing else, and a covariance with the other
candidates regressed out is a different quantity that can shrink or reverse a
real influence. What the report adds is the means to read it: the query's
covariance with the average candidate (`baseline_cov`, and each example's
`above_baseline`), which is the shared part, and the *partial* covariance,
where both series are regressed on the per-draw mean loss of the *other*
candidates and the residuals covaried, which is the part of an example's
movement specific to the query. The candidate being scored is left out of its
own control, since a lone candidate regressed on itself has a residual of
exactly zero; with one candidate there is nothing to control on, so its partial
is the raw covariance and the file says so (`partial_control`). An example high
on both is one the model's loss on the query moves with in particular; one high
on covariance alone moves with everything. The raw draws are in the
file too (`draws[chain][draw]`, the query's loss then each record's), so any
other statistic can be computed later without re-running the chains.

Two things bound what the number means, and both are in the result file rather
than assumed. The posterior is localized on the candidates — a few hundred
examples — not on the training set, so the covariances are about that sample.
And the influence is on the loss of the checkpoint named in the file: pass
`--model` to weigh a target's samples against a different checkpoint (all Pythia
sizes saw the same documents in the same order, which is what makes a 70M run
over the 12B target's Pile sample a sensible thing to do), and the report says
which checkpoint it ran on.

### The defaults, and why

The step size is the one setting that matters. Each SGLD step adds Gaussian
noise of standard deviation `√ε` to every parameter, so over `T` steps a chain
random-walks about `√(Tε)` in each of tens of millions of directions, and a
walk comparable to the weights themselves is a chain that has left the model.
On Pythia-70m over 200 Pile documents, 80 steps at `ε = 1e-6` doubled the loss;
at `1e-7` the minibatch loss rose 0.75 nats from a start of 3.4, just under the
line the report draws; at the default `5e-8` it rose about 0.5 with a learning
coefficient of 2.2, and the four chains agreed with each other to one decimal. A
bigger model needs a smaller step, and the report says when it does. `nβ`
defaults to `batch / ln(batch)` as in devinterp, `γ = 100`, four chains of a
hundred retained draws after fifty burn-in steps. The runtime is `chains ×
draws × candidates` forward passes plus the SGLD steps: minutes for a 70M model
over 300 documents on a laptop GPU, and a multi-GPU-hour job for a 7B model,
which is what `--limit`, `--max-tokens` and `--dtype bfloat16` are for. Memory
for a reduced-precision run is the model and its gradients in that precision,
a float32 master copy the chain walks in (the default step is below bfloat16's
own spacing, so the walk cannot happen in bfloat16), and `w*` in the loaded
precision: about five times the weights' reduced size, or 70 GB for 7B before
activations. Noise is drawn on the parameter's own device.

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

### The other route: a corpus the viewer serves whole

Everything above is a workaround for a corpus the dataset viewer only partly
indexed. The Pile is the case where none of it is needed:
`EleutherAI/the_pile_deduplicated` is indexed in full — 134,318,121 rows,
`partial: false` — so `/rows` reaches any offset and `pretrain` pages it
directly. Documents are drawn from the whole corpus, there is no shard listing
to cache and no position bias to disclose. The registry picks the route per
stage with `sample_via`; `shards` is the default, because a stage that forgets
to declare one should get the route with the honest caveat rather than the one
that quietly samples 5 GB of a 450 GB repo.

What the direct route still has is a cluster. `/rows` returns pages of ten
adjacent rows, so a 300-document sample is 30 pages, and each document records
the page it arrived in — the analogue of the shard route's shard path. Intervals
are computed over those pages. On the Pile the correlation inside a page is
close to nil, because the corpus was shuffled before release, so the design
effect lands near 1 and the interval is almost the uncorrected one; measuring it
rather than assuming it is what lets the file say so. Leaving the page out is a
real failure mode and was one: with every document carrying the same empty
cluster label, the correction read 300 documents as a single observation and
printed a 0–83% interval.

What the direct route gives up is provenance. Dolma 3 names each document's
source and topic in its shard path and carries per-document filter metadata;
the deduplicated Pile is one `text` column, its `pile_set_name` labels dropped
in the dedup release. So a Pile document arrives with its row index and nothing
else, the composition shown for it is EleutherAI's published table rather than
a listing counted here, and that table describes the Pile *before*
deduplication — dedup removed roughly 30% of the bytes, unevenly.

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

`trainspotting lookup` asks the same question of a different set of corpora —
Dolma 1.7, DCLM-baseline, RedPajama, C4 and The Pile rather than one OLMo index —
and ships a committed case study over them. None of those is Dolma 3 either. The
two commands share the infini-gram API and differ in which indexes they reach and
what they keep; see [Looking one text up](#looking-one-text-up).

The bias sampling leaves is positional. A range request only reaches the front of
a shard, so each sampled document comes from its shard's first few hundred (one
picked uniformly from them), never the tail. Shard draws are proper; position
within a shard is not corrected for. That caveat
travels in every result file and is printed on the site.

## Looking one text up

Every layer above samples, and sampling has a floor: one blog is a rounding
error in a 6T-token mix, so a 300-document draw will never contain it no matter
how carefully the draw is done. Asking whether one particular text is in a
corpus needs an index, not a sample.

`trainspotting lookup` queries [infini-gram](https://infini-gram.io), a suffix
array over five public corpora, and prints the count per corpus with the
documents behind it:

```bash
trainspotting lookup "For the pointer I thank" --docs 10     # ten random occurrences
trainspotting lookup "For the pointer I thank" --docs all    # every occurrence, by rank
```

| Corpus | What it is |
|---|---|
| `v4_dolma-v1_7_llama` | Dolma 1.7 — Ai2's open corpus, the OLMo 2 generation |
| `v4_dclm-baseline_llama` | DCLM-baseline — a heavily model-filtered Common Crawl derivative |
| `v4_rpj_llama_s4` | RedPajama v1 |
| `v4_c4train_llama` | C4 |
| `v4_piletrain_llama` | The Pile |

**None of these is Dolma 3.** No public index covers it, so nothing this
command returns describes what OLMo 3 read. That is why the site puts it in its
own tab rather than under a model.

Two properties of the index decide how a result may be read, and both are easy
to get backwards:

- **The count is occurrences, not documents.** A page that repeats a phrase
  three times contributes three. Reading a count as "copies in the training
  data" inflates it.
- **Sampled documents are exhaustive only at or under ten occurrences, and
  only when the run asked for all of them.** Ten is the API's per-call cap.
  Above it the index draws occurrences uniformly at random, with replacement,
  and a re-run returns different ones, so a committed result is a snapshot;
  below it, `--docs 3` against a phrase occurring eight times is still a sample
  of three, and the flag says so. It is also cleared when the index returns
  fewer documents than were asked for. Result files carry `exhaustive` per pull
  and a `run_on` date instead of a revision, because a live index has nothing
  to pin.
- **`--docs all` takes the census instead.** The index also lists occurrences
  by rank, one request each, so `trainspotting lookup "For the pointer I
  thank" --docs all` fetches all 333 and reports how many distinct documents
  they collapse to, with a `×n` on any document that holds the phrase more
  than once. It costs one round trip per occurrence, which is why it is
  opt-in: fine for hundreds, slow for thousands, out of reach for a common
  phrase.

### The committed study

`trainspotting case-study marginal-revolution` runs a fixed set of queries
against a blog that has published daily since 2003, and writes
`results/case-study.marginal-revolution.json` — what the site's *is my writing
in here?* tab reads. As of 2026-08-31:

- **`"For the pointer I thank"`, the blog's standard credit for a reader-submitted
  link: 333 occurrences in Dolma 1.7**, 883 in DCLM-baseline, 12 in the Pile. Each
  occurrence is one post's use of it, so that is a ceiling on how many such posts
  are present at all — far below how many were written.
- **One post is there twice, in two documents**, and the count is under the cap so
  that is all of them: its own permalink page, cut to 105 tokens with 7 of its 76
  lines surviving the line filter, and the month archive page it also appears on,
  which kept 87 of 270. Whole posts do not become whole documents.
- **Of 60 occurrences of the blog's own name drawn at random, 0 are on the blog.**
  They are on bookmark mirrors, aggregators and SEO scrapers. The blog's own pages
  are in the corpus — the post above is one — but a site heading is a nav line and
  line-level filtering drops nav lines, so the name survives mainly where somebody
  else typed it into their own prose. What looks like heavy duplication of a writer
  is mostly other people's copies.

<!-- figures: case-study.marginal-revolution 2d7f2bf5cb99 -->

The study labels every query group with how it was chosen. Queries picked from
knowing the blog can measure coverage; queries found by reading documents that
were already sampled out of the corpus are present by construction and can only
show what a present post looks like. Mixing the two silently would report a
hit rate of 100% as a finding.

Measuring the rate properly needs a list of the blog's posts drawn independently
of any corpus. There is no committed one: `marginalrevolution.com` serves a
Cloudflare challenge to automated requests, and Common Crawl's CDX index and the
Wayback CDX API were both unreachable when this was built. Adding one is the
obvious next step, and `selection` in the result file is where it would land.

### Why the site has no search box

infini-gram's API sends no `Access-Control-Allow-Origin` header and answers
`OPTIONS` with a 403, so a browser cannot call it from a page on another origin.
A live box would need a proxy server; the site is static files served from
`docs/` with nothing behind them. The CLI runs the same queries locally, and
[infini-gram's own Space](https://huggingface.co/spaces/liujch1998/infini-gram)
is the hosted equivalent.

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
`search` is checked against those same saved rows: every registry stage has to
yield more than its prompt, or a search of it is the prompt-only search the
layer exists to replace.

`grep` is covered twice over. Its column-to-field mapping runs against one saved
Parquet schema per stage (`tests/fixtures/schemas/`, re-captured by
`scripts/capture_parquet_schemas.py`), and the query it builds runs for real
against small Parquet files written in the test — locally, so the counts,
the role split, the null handling and the byte accounting are checked without a
network. Those tests need DuckDB, which `[dev]` installs.

`contaminate` is covered the same two ways. `tests/test_benchmarks.py` pins the
probe: the window is cut from the middle, its regex survives a re-wrapped or
re-cased copy and nothing else, a short item is not probed, and the items are
fetched one page per page. `tests/test_contamination.py` runs the many-probe
query against a local DPO-shaped Parquet file — a question in a prompt, its
answer in a chosen completion, another item re-cased in a rejected one — and
checks that every hit lands on its own probe and its own side, that splitting
the alternation into many regexes finds the same rows as one, and that the
roll-up to items keeps "read" and "produced" apart.

The site's search index is tested where it can silently lose a match:
`tests/test_searchindex.py` builds a small index and asserts that a query
inside a word, inside an accented word, inside a space-free script, or made of
astral characters still keeps the file that holds it. The same cases run
against the shipped `docs/data/search-index.json` when a checkout has one.

`--live` re-runs the extraction checks against rows fetched right now, and
checks each saved Parquet schema against the current one. That is the canary for
an upstream schema change, which otherwise shows up only as a sampling run that
quietly labels nothing, or a string search that quietly counts less.

`tests/test_bif.py` pins everything around the sampler that does not need a
checkpoint: which committed examples become candidates and which stages are
skipped with a reason, that the loss labels cover the fit text only and
truncation drops the prompt rather than the answer, that a chat template's
rendering is what gets tokenized and a template that cannot render falls back to
roles, the sign and normalization of the covariance, that chains are averaged
within rather than pooled, the drift and baseline diagnostics, and what the
report says when a chain was still drifting, in either direction, or sat below `w*`. The SGLD loop
runs against a toy model when torch is importable — shapes, finite losses, the
weights restored to `w*` afterwards, padding kept out of the loss — and is
skipped otherwise.

`steps` is pinned where it would fail silently: the shard layout has to close
exactly over 143,000 steps or every offset is an address into the wrong text, a
step straddling a shard seam has to come back as two contiguous ranges, the draw
has to put one step in each slice of the run, and the interval has to widen when
the matches sit in two steps rather than sixteen. `--live` fetches step 0 and
checks its first sequence still decodes to the same sentence, which is what a
republished stream with a different cut would move.

The budget arithmetic is pinned per kind — what counts as a fit token for an
SFT example, a preference pair, an RL row that ships no generation — along with
the length weighting and what happens to a stage nobody can size. `stance`
rendering is checked against every committed context record, not just
constructed ones: the bug it guards against passed on all the short examples and
lost a side marker on the long ones.

The derived numbers are held to the committed samples themselves rather than to
fixtures, because the ways they break are all silent: a profile that has drifted
from the context file it summarizes, a prompt-key hash that no longer matches the
copy in `docs/index.html`, a DPO pair whose shared history gets counted as text
the model was fit to. `pytest` needs `node` on PATH for the hash-parity check and
skips it otherwise.
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
- The stage ranking is evidence about where a string is, and only that. It does
  not weight the stages against each other, so a rate in RLVR and the same rate
  in pretraining rank equal even though the late one generally moves behaviour
  more; and a pattern present in a stage is not a demonstration that any
  particular behaviour came from it. For "did this exact document train the
  model", use OLMoTrace.
- `bif` weighs a few hundred sampled examples against one checkpoint's loss,
  with a posterior localized on those examples rather than on the training set.
  It ranks the examples the other layers found; it does not find new ones. A
  covariance from a chain the report flags as still drifting from `w*`, up or down, is
  not a posterior covariance; a learning coefficient at or below zero is not
  that flag, since `w*` minimizes the training set and not the few hundred
  examples the posterior is localized on.
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
  clustering by whatever unit the sample was drawn in — the shard on the shard
  route, the page of ten adjacent rows on the direct one — so `--docs-per-shard`
  runs whose matches bunch inside shards get an honestly wider interval, and
  runs whose matches are spread evenly are not penalised for the grouping
  alone. The corrected interval is
  computed once, in the CLI, and stored in the result file rather than
  recomputed by the site.
- On the shard route the pretraining sampler only sees documents a range request
  can reach — the first few hundred in each shard, one drawn uniformly from
  those. Shards are drawn properly; position within a shard is not corrected
  for. The direct route has no such limit: it reaches every document in the
  corpus.
- `grep` reads the datasets-server's Parquet conversion of a mix, not the repo
  files themselves, and covers only the post-training mixes. The pretraining
  corpora are not converted (the same partial index that stops the shard route),
  so an exact count over a pretraining corpus is `find`'s job rather than
  `grep`'s — and for Pythia, which has no post-training at all, `find` is the
  only one of the two that applies.
- A `grep` count is over the text, which is not the same as over the tokens the
  model saw. A phrase split across two message turns, spelled with styled
  Unicode characters (`𝗖𝗵𝗮𝘁𝗚𝗣𝗧` is in the DPO mix), or transliterated will not
  match. Every count is a lower bound on the concept and an exact figure only
  for the pattern.
- `grep` and `find` answer the same question about different halves of the
  pipeline and by different routes, so their numbers are not comparable. `grep`
  counts **rows** of a post-training mix whose text contains a pattern, exactly,
  over the mix the model was actually trained on. `find` counts **occurrences**
  of a token sequence in a pretraining index, which for the OLMo 3 models is not
  their own data.
- `find` searches the closest public infini-gram index, which for the OLMo 3
  models is OLMo 2's training data rather than their own — no Dolma 3 index
  exists on the public API yet. Pythia is the exception: `v4_piletrain_llama`
  is the Pile itself, differing from what `pythia-12b-deduped` saw only by
  deduplication. Matches align to token boundaries either way: querying `a`
  counts the token ` a`, not the letter.
- The Pile composition shown for Pythia is EleutherAI's published table for the
  corpus as assembled, not a listing of the deduplicated release the documents
  are sampled from, and not measured here.
- `budget` reports an exposure budget, not an attribution. It weighs tokens, and
  a pretraining token, a preference token and an RL gradient step do not move a
  model equally. It also converts characters to tokens at a flat 4:1, which is
  roughly right for English prose and roughly wrong for code and CJK — the
  ratios between stages are far less sensitive to that than the absolute counts.
- A `budget` for an RL stage is a floor. The published mix holds prompts,
  verifiers and reference generations, not the rollouts the policy was fit to,
  and the number of rollouts per prompt is not in the data.
- `stance` judges the stored context record, whose fields are cut at 4,000
  characters, and then fits the rendered example into 12,000. A value expressed
  only in the part that was cut is not visible to it. Every view links to the
  untruncated row on HuggingFace.
- A `budget` stage joins its ask run to its stored examples by row index, which
  only means anything within one dataset revision. Both result files carry the
  revision they were drawn at, and a known disagreement leaves the stage
  unusable rather than joined — Ai2 has republished these mixes, and after a
  republish the same index is different text. A run that straddled a republish
  mid-sample is refused for the same reason: both producers stamp
  `revision_moved_to` when they detect one, and which rows came from which tree
  is not recorded, so no part of that join can be trusted. The row count comes from a third
  run (`sources`) which can be staler still; when it names a different revision
  the stage keeps its rate, which is a share, and loses its token figure, which
  is a count.
- A shard-drawn corpus rate assumes shard-proportional sampling did the token
  weighting, which is true between shards and only approximately true within
  one: a document is drawn uniformly from its shard's reachable head rather than
  by length. Long documents are slightly underweighted for that reason, and the
  `short_draws` bias documented above pushes the same way. A rows-drawn corpus
  has neither problem and is length-weighted instead, so its residual is the one
  post-training carries: the interval treats the length ratio as known when it
  was itself measured on the same 300 documents.
- A `budget` total is withheld, in the CLI and on the site, when the stages
  under one slug were not scored by the same instrument — different wordings of
  the question, different classifiers, or a rubric that moved between stages of
  the same family (compared by the `system_sha` every result file carries). A
  slug is not a question: `--slug` takes any string and a generated one is cut
  to 60 characters, so a collision is possible, and a sum across one is a number
  no single measurement produced. The rubric is compared within a family and
  never across one: a corpus document is judged under a different rubric than a
  post-training prompt on purpose, so that pair is expected rather than a
  conflict.
- A stage's matching-token interval is clamped to the stage. A few matching
  examples much longer than the rest can rescale a Wilson endpoint past the
  stage's whole fit-token count, which is an impossible bound rather than a
  wide one.
- Stage sizes are estimated over the whole stored context sample, not over the
  prompts the classifier answered about. Refusals land on jailbreak-style
  prompts, so letting classifier success pick which examples set the mean length
  would put that bias into every token figure. The match rate is still over the
  labeled subset, which is the only part there is a judgment for.
- `stance` and `ask` answer different questions and their counts do not nest.
  An example can be `about` human lives and push `away` from valuing them, which
  is the case the direction layer exists for.
- Registry facts (token counts) are from the Olmo 3 paper
  ([arXiv:2512.13961](https://arxiv.org/abs/2512.13961)) and the
  [release blog](https://allenai.org/blog/olmo3); Pythia's from the Pythia paper
  ([arXiv:2304.01373](https://arxiv.org/abs/2304.01373)) and the Pile paper
  ([arXiv:2101.00027](https://arxiv.org/abs/2101.00027)).

## Adding a model

Add an entry to `MODELS` in `trainspotting/registry.py`. A stage carries either an
`hf_dataset` plus `prompt_path` / `source_columns` schema hints (post-training,
served by the datasets-server), or a `sample_dataset` naming a corpus
(pretraining), or just `tokens` for a facts-only row. Any fully open pipeline on
the Hub works the same way.

A pretraining stage also picks how its corpus is read, with `sample_via`. Check
`https://datasets-server.huggingface.co/rows?dataset=<id>&config=default&split=train&offset=<near the end>&length=1`
first: if it answers and reports `"partial": false`, the viewer has the corpus
indexed in full and `sample_via: "rows"` is both simpler and a better sample —
give it a `text_column` and nothing else. Otherwise leave it on the default
shard route, and expect the shard path parser in `trainspotting/pretrain.py` to
need a naming convention added; `split_group` reads provenance out of each
shard's parent directory, which not every repo puts it in.

A stage with no `composition` renders no breakdown, which is the honest result
for a corpus whose per-document source labels are not published. Where the
release publishes sizes rather than token counts, set `composition_unit:
"bytes"`, and if that published table describes a different cut of the corpus
than the one being sampled, say which in `composition_scope` — the site prints
it under the bars.

A model with no post-training stages at all is a supported shape, not a broken
entry: `sources`, `classify`, `languages` and `context` exit with a message
naming the reason, `report` stops after the pipeline, and `ask --pretrain`
scores the corpus on its own.

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
`python scripts/export_site_data.py` to give it a tab on the site, under the
tab bar's own **datasets** group — the site splits the two kinds of target
apart, because a dataset tab answers a different question from a model tab.

A `prompt_path` the dataset needs and `extract.py` doesn't implement is the one
piece that costs more than a registry entry: WildChat's `conversation` column
was a four-line branch there.
