# Models, datasets and corpora

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
  check on py3langid rather than the only breakdown available. The dataset's
  own column says 56.2% English / 14.9% Chinese / 10.4% Russian. In the
  [committed detector run](data/wildchat-1m.chat.languages.json), 996 prompts
  remain from a requested 1,000-row draw: 55.8% English (556), 12.1% Chinese
  (121), 10.2% Russian (102), and 9.3% undetermined (93). These shares use the
  retained prompts as their denominator; the dataset-wide column and sampled
  detector also differ in coverage and whether they leave a language uncalled.

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
