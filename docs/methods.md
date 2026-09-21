# Method, taxonomy, tests and caveats

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

Most labels come from the classifier reading the prompt. Where an RL row's
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
Across the three RL samples, 260 rows are settled by their verifier and 47 of
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

`tests/test_pairs.py` pins the preference-pair arithmetic where it would be
wrong and still look like a number: a multi-turn pair's shared history counted
on both sides (which cancels out of the difference and not out of the two
totals), a tie scored as an answer the length rule got wrong, a think stage's
reasoning and answer halves that do not add back up to the gap printed above
them, a signed distribution binned through `derive.histogram` — which drops
everything at or below zero, so the rejected side's half of the chart simply
would not be there — and a `pairs.json` that has drifted from the context
sample the site serves beside it. The last is checked against the committed
samples rather than a fixture, for the same reason the derived numbers are.

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

`tests/test_bif.py` checks candidate selection, loss masking, covariance
arithmetic, conservative diagnostics, and withholding unsupported conclusions.
The sampler tests use a tiny CPU model. A dedicated CI job installs torch and
runs the exact Gaussian validation; the regular dependency-light test job
continues to exercise the rest of the command without model downloads.

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

- The values layer classifies **prompts**. For RL stages the values are also
  carried by the reward, which the prompt text does not show. Where that reward
  is a constraint checker the label comes from it instead of from the prompt
  (see [Taxonomy](#taxonomy)); where it is an LLM judge the rubric is not
  published with the dataset, so those rows are still labeled from the prompt
  alone. The `sources` layer's reward-type breakdown and the `context` layer's
  verifier view are the complement.
- `pairs` measures what is *available* to be fit from a preference stage
  without reading either answer — the length asymmetry and the generator
  asymmetry — not what any trained policy fit. The generator rule in particular
  is read off the same rows it is scored on, so it is a ceiling for that sample
  and carries `fitted: true` rather than an interval. And on Dolci the asymmetry
  is partly by design: `delta_learning` pairs a strong generator against a weak
  one on purpose, so a rate of 1.0 there is the recipe rather than a defect in
  it. What the layer adds is the price — how much of the label a length rule
  also recovers.
- `context` records which sampled rows arrived with a cell the datasets-server
  had shortened, because `chars` is otherwise read as a field's true length and
  for a cut cell it is the length of what arrived. Cuts land on the longest
  cells, which is exactly what `pairs` measures, so a sample committed before
  the check existed reports `truncated_rows: null` — unknown rather than none.
  `pairs` counts only the cuts that land in `chosen` or `rejected`: those are
  the cells its lengths come from, and a row shortened in the standalone
  `prompt` cell has both measured sides whole.
- The stage ranking is evidence about where a string is, and only that. It does
  not weight the stages against each other, so a rate in RL and the same rate
  in pretraining rank equal even though the late one generally moves behaviour
  more; and a pattern present in a stage is not a demonstration that any
  particular behaviour came from it. For "did this exact document train the
  model", use OLMoTrace.
- `bif` is experimental standalone-text sensitivity on Pythia-70m, not training
  attribution. Its Gaussian validation does not validate a language model. The
  committed language-model run is inconclusive and its ranking is withheld.
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
- Sampled estimates come with Wilson 95% intervals in `report`; the default
  1,000 post-training samples give roughly ±3 percentage points in the worst
  case. Actual intervals use each run's labeled count. Post-training intervals
  assume independent draws.
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

## Redacted credentials in sampled text

Stored examples replace Discord bot tokens with `[REDACTED_DISCORD_TOKEN]`.
The result writer and site exporter apply this targeted redaction because public
training data can contain credentials. Dataset revisions, row identifiers,
original character counts, and existing classifier judgments remain unchanged;
the marker changes only the displayed text. Other types of credentials are not
covered by this detector.
