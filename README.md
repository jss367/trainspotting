# trainspotting

Spot what's in a model's training data. Audits what a fully open model was
trained on — currently the OLMo 3 pipelines (Ai2), whose pretraining (Dolma 3)
and post-training (Dolci) data are public — and, with the same layers, any
dataset on its own. The tool answers eight kinds of question. The first five go
in increasing order of depth; the sixth is a lookup rather than an estimate; the
last two read a whole example rather than its prompt, and put the answer on a
scale the stages share:

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
7. **Direction** — which way an example pushes on a question: `toward`, `away`,
   or `neither`. A yes/no over prompts cannot say that a stage contains training
   pointing the other way, and this data has some. See
   [Which way an example pushes](#which-way-an-example-pushes).
8. **Budget** — every stage's rate times its size, in tokens the model was fit
   to, so the stages are on one scale and add up. A share of DPO rows and a
   share of Dolma 3 documents are different denominators; this is where they
   become one number. See [How much training is that?](#how-much-training-is-that).

Every one of those starts from something you can already name — a string to
search for, or a question you can already phrase. When you start from an
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

# Judge whole examples instead of prompts, and sign the answer
trainspotting stance olmo-3-7b-think \
  "Is this training example about caring about human lives?" \
  --slug caring-about-human-lives

# Add every stage up on one scale: tokens the model was fit to (no API key needed)
trainspotting budget olmo-3-7b-think caring-about-human-lives

# Sample documents from the pretraining, midtraining and long-context corpora
trainspotting pretrain olmo-3-7b-think --sample 300

# Score those documents against the same question as the post-training stages
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain

# ...or only the corpora, when the post-training half is already committed
trainspotting ask olmo-3-7b-think "..." --slug my-question --pretrain-only

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
the loss itself. Chosen tokens are pushed up, rejected tokens are pushed down,
and only the gap between the two is visible to the loss, so the panel diffs the
two responses and marks three kinds of span. A byte-exact shared opening is the
only place cancellation is exact, because both sides are conditioned on identical
text there. Wording that reappears after the responses diverge follows a
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

**Corpus documents are weighed by nothing extra.** `trainspotting pretrain`
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

The interval is the count-based one — cluster-corrected for corpora, where the
`ask` run already stored it — rescaled by the weighed rate over the count rate,
which is exactly 1 for a corpus stage. It is computed over the rows the point
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

The budget arithmetic is pinned per kind — what counts as a fit token for an
SFT example, a preference pair, an RL row that ships no generation — along with
the length weighting and what happens to a stage nobody can size. `stance`
rendering is checked against every committed context record, not just
constructed ones: the bug it guards against passed on all the short examples and
lost a side marker on the long ones.

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
- The corpus rate assumes shard-proportional sampling did the token weighting,
  which is true between shards and only approximately true within one: a
  document is drawn uniformly from its shard's reachable head rather than by
  length. Long documents are slightly underweighted for that reason, and the
  `short_draws` bias documented above pushes the same way.
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
