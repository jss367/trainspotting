"""What the tool can be pointed at: models, and datasets on their own.

A **model** maps to an ordered list of stages. Pretraining-scale stages carry
facts only (the corpora are terabytes; we don't compute over them). Post-training
stages carry a HuggingFace dataset ID plus schema hints so the generic code can
pull prompts and source tags out of heterogeneous Dolci schemas.

A **dataset** is a target with no pipeline around it — one samplable dataset and
nothing else. Every layer below `facts` only ever needed a dataset ID and how to
read a prompt out of a row; requiring a training pipeline around that was an
accident of starting with Olmo. `resolve` hands both kinds back in the same
shape, so a dataset is a one-stage target and no command has to know which it
got.

Not every model has all of these. Pythia has a pretraining stage and nothing
else, because EleutherAI never post-trained it — so `post_training_stages` comes
back empty and every layer built on prompts sits out. That is a fact about the
model worth showing, not a target to exclude.

Model facts sourced from the Olmo 3 paper (arXiv:2512.13961) and the Ai2 release
blog; Pythia's from the Pythia paper (arXiv:2304.01373) and the Pile paper
(arXiv:2101.00027).
"""

# Schema hints:
#   prompt_path: how to pull the user prompt out of a row.
#     "messages"        -> first role=="user" content in row["messages"]
#     "chosen_messages" -> first role=="user" content in row["chosen"]
#     "prompt"          -> row["prompt"] (string, or list of chat messages)
#     "conversation"    -> first role=="user" content in row["conversation"]
#   source_columns: columns whose value counts describe the mix composition.
#     Every name must exist on the dataset. `sources` renders each column
#     independently and skips one the dataset doesn't have, so a stale name
#     is invisible — it drops a breakdown from the site rather than erroring.
#
# Base (pretraining-scale) stages additionally carry:
#   hf: dataset ID for linking only — hf_dataset stays None so the generic
#       prompt-sampling code never tries to pull rows from a multi-TB corpus.
#   composition: [{name, tokens}] per-source token counts from the mix's own
#       dataset card / the Olmo 3 paper (Table 4), not computed by us.
#   sample_dataset: the repo the pretrain sampler reads. Distinct from `hf`,
#       which is only a link: this one has to be a repo the sampler can draw
#       documents from.
#   sample_via: how to draw from it — "shards" (default) walks the repo's
#       .jsonl.zst files by HTTP range request, "rows" pages the
#       datasets-server. See `sample_route`.
#   text_column: for a "rows" corpus, the column holding the document text.
#   sample_scope: what that repo is relative to what the model actually saw.
#   composition_unit: what the `composition` numbers count. Absent means
#       tokens, which is what the Dolma 3 mixes publish. "bytes" is for a
#       corpus whose own release states sizes rather than token counts.

# The 6T-1025 mix's own card: 5.93T tokens over 3.87B documents, upsampled
# from the ~9.3T-token Dolma 3 pool.
DOLMA3_MIX_COMPOSITION = [
    {"name": "Common Crawl (web)", "tokens": 4_510_000_000_000},
    {"name": "olmOCR science PDFs", "tokens": 805_000_000_000},
    {"name": "Stack-Edu (code)", "tokens": 409_000_000_000},
    {"name": "FineMath 3+ (math)", "tokens": 151_000_000_000},
    {"name": "arXiv (proof-pile)", "tokens": 50_900_000_000},
    {"name": "Wikipedia & Wikibooks", "tokens": 2_510_000_000},
]

# Olmo 3 Base 7B: 5.93T pretrain + 100B midtrain + 50B long-context.
OLMO3_7B_BASE_STAGES = [
    {
        "stage": "pretrain",
        "name": "Dolma 3 Mix",
        "tokens": 5_930_000_000_000,
        "note": (
            "5.93T tokens over 3.87B documents, sampled (with upsampling of the "
            "smaller sources) from the ~9.3T-token Dolma 3 pool. Three quarters "
            "of pretraining is Common Crawl web text."
        ),
        "hf": "allenai/dolma3_mix-6T-1025-7B",
        "sample_dataset": "allenai/dolma3_mix-6T-1025-7B",
        "sample_scope": (
            "The exact mix used to pretrain Olmo 3 7B. Some olmOCR science PDFs were redacted to `[REMOVED]` after training, so a few documents here show a placeholder where the model saw real text."
        ),
        "composition": DOLMA3_MIX_COMPOSITION,
        "hf_dataset": None,
    },
    {
        "stage": "midtrain",
        "name": "Dolma 3 Dolmino Mix",
        "tokens": 100_000_000_000,
        "note": (
            "Curated to boost math, code, QA, instruction following, and "
            "thinking — much of it synthetic (reasoning traces, generated QA, "
            "Flan/Tulu instructions) alongside the highest-quality web and PDF "
            "data from pretraining."
        ),
        "hf": "allenai/dolma3_dolmino_mix-100B-1025",
        "sample_dataset": "allenai/dolma3_dolmino_mix-100B-1025",
        "sample_scope": (
            "The 100B-token Dolmino mix itself."
        ),
        "composition": [
            {"name": "web pages", "tokens": 28_000_000_000},
            {"name": "code", "tokens": 20_000_000_000},
            {"name": "math", "tokens": 19_000_000_000},
            {"name": "QA", "tokens": 14_000_000_000},
            {"name": "thinking / reasoning traces", "tokens": 8_000_000_000},
            {"name": "instruction following", "tokens": 6_000_000_000},
            {"name": "science PDFs", "tokens": 5_000_000_000},
        ],
        "hf_dataset": None,
    },
    {
        "stage": "long-context",
        "name": "Dolma 3 Longmino Mix",
        "tokens": 50_000_000_000,
        "note": (
            "Extends context length: two thirds replays the midtraining mix, "
            "one third is long science PDFs (real and synthetic, 8K–64K tokens)."
        ),
        "hf": "allenai/dolma3_longmino_mix-50B-1025",
        "sample_dataset": "allenai/dolma3_longmino_mix-50B-1025",
        "sample_scope": (
            "The 50B-token Longmino mix the 7B run used."
        ),
        "composition": [
            {"name": "midtraining data (replay)", "tokens": 33_000_000_000},
            {"name": "long science PDFs (8K–64K)", "tokens": 8_930_000_000},
            {"name": "synthetic long PDFs", "tokens": 8_020_000_000},
        ],
        "hf_dataset": None,
    },
]

# Olmo 3 Base 32B: same Dolma 3 Mix recipe but 5.50T tokens of it, then two
# 100B Dolmino mixes trained separately and model-merged, then 100B long-context.
OLMO3_32B_BASE_STAGES = [
    {
        "stage": "pretrain",
        "name": "Dolma 3 Mix",
        "tokens": 5_500_000_000_000,
        "note": (
            "5.50T tokens of the same Dolma 3 Mix recipe as the 7B run "
            "(the paper's Table 4 mix applies to both); shares below are the "
            "mix's own proportions."
        ),
        "hf": "allenai/dolma3_mix-6T",
        "sample_dataset": "allenai/dolma3_mix-6T",
        "sample_scope": (
            "The primary Dolma 3 6T mix, used to pretrain Olmo 3 32B. No redactions."
        ),
        "composition": DOLMA3_MIX_COMPOSITION,
        "hf_dataset": None,
    },
    {
        "stage": "midtrain",
        "name": "Dolma 3 Dolmino Mix ×2",
        "tokens": 200_000_000_000,
        "note": (
            "Two separate 100B Dolmino runs (web, code, math, QA, thinking, "
            "instruction, PDFs — heavily synthetic) whose checkpoints were "
            "model-merged before the long-context stage."
        ),
        "hf": "allenai/dolma3_dolmino_mix-100B-1125",
        "sample_dataset": "allenai/dolma3_dolmino_mix-100B-1125",
        "sample_scope": (
            "The 100B-token Dolmino mix itself."
        ),
        "hf_dataset": None,
    },
    {
        "stage": "long-context",
        "name": "Dolma 3 Longmino Mix",
        "tokens": 100_000_000_000,
        "note": "Extends context length: midtraining-data replay plus long science PDFs.",
        "hf": "allenai/dolma3_longmino_mix-100B-1125",
        "sample_dataset": "allenai/dolma3_longmino_mix-100B-1125",
        "sample_scope": (
            "The 100B-token Longmino mix."
        ),
        "hf_dataset": None,
    },
]

# --- Pythia (EleutherAI) ------------------------------------------------------
#
# The other end of the openness spectrum from Olmo: the corpus is public and the
# exact training order is published, but the pipeline stops at the base model.
# EleutherAI never post-trained Pythia, so there is no SFT, DPO or RL stage to
# register — `sources`, `classify` and `context` have nothing to run on, and the
# helpful/honest/harmless question this tool leads with is unanswerable here.
# What Pythia gives instead is a pretraining corpus that can be sampled
# *properly*: see `sample_via` below.
#
# GiB, as the Pile's own table reports sizes. Raw Size, not Effective Size —
# the released corpus holds each document once, and the epoch multipliers in
# the paper's other column describe a training schedule Pythia did not use.
_GIB = 1024**3

# The Pile's 22 components by raw size, from EleutherAI's own table
# (github.com/EleutherAI/the-pile, Table 1 of arXiv:2101.00027). Sums to
# 825.18 GiB, the "825 GiB" the paper's title rounds to.
PILE_COMPOSITION = [
    {"name": "Pile-CC (web)", "bytes": int(227.12 * _GIB)},
    {"name": "Books3", "bytes": int(100.96 * _GIB)},
    {"name": "GitHub", "bytes": int(95.16 * _GIB)},
    {"name": "PubMed Central", "bytes": int(90.27 * _GIB)},
    {"name": "OpenWebText2", "bytes": int(62.77 * _GIB)},
    {"name": "arXiv", "bytes": int(56.21 * _GIB)},
    {"name": "FreeLaw", "bytes": int(51.15 * _GIB)},
    {"name": "Stack Exchange", "bytes": int(32.20 * _GIB)},
    {"name": "USPTO Backgrounds", "bytes": int(22.90 * _GIB)},
    {"name": "PubMed Abstracts", "bytes": int(19.26 * _GIB)},
    {"name": "OpenSubtitles", "bytes": int(12.98 * _GIB)},
    {"name": "Gutenberg (PG-19)", "bytes": int(10.88 * _GIB)},
    {"name": "DM Mathematics", "bytes": int(7.75 * _GIB)},
    {"name": "Wikipedia (en)", "bytes": int(6.38 * _GIB)},
    {"name": "BookCorpus2", "bytes": int(6.30 * _GIB)},
    {"name": "Ubuntu IRC", "bytes": int(5.52 * _GIB)},
    {"name": "EuroParl", "bytes": int(4.59 * _GIB)},
    {"name": "HackerNews", "bytes": int(3.90 * _GIB)},
    {"name": "YouTube Subtitles", "bytes": int(3.73 * _GIB)},
    {"name": "PhilPapers", "bytes": int(2.38 * _GIB)},
    {"name": "NIH ExPorter", "bytes": int(1.89 * _GIB)},
    {"name": "Enron Emails", "bytes": int(0.88 * _GIB)},
]

# 143,000 steps x 1024 sequences x 2048 tokens. Every Pythia size saw this same
# token budget over this same corpus in this same order — the point of the suite.
PYTHIA_TOKENS = 143_000 * 1024 * 2048

PYTHIA_DEDUPED_STAGES = [
    {
        "stage": "pretrain",
        "name": "The Pile (deduplicated)",
        "tokens": PYTHIA_TOKENS,
        "note": (
            "22 curated sources — web, books, code, papers, law, patents, "
            "subtitles, email — assembled by EleutherAI in 2020 and released "
            "whole. The deduplicated release is about 207B tokens, so Pythia's "
            "299.9B-token budget is roughly 1.5 passes over it rather than one. "
            "There is no midtraining, no long-context stage and no "
            "post-training: EleutherAI built Pythia to study how a model "
            "changes during pretraining, and stopped at the base model."
        ),
        "hf": "EleutherAI/the_pile_deduplicated",
        "sample_dataset": "EleutherAI/the_pile_deduplicated",
        # The one corpus in the registry the datasets-server serves whole:
        # 134,318,121 rows, `partial: false`, so /rows reaches any offset and
        # the sample is uniform over documents. Every Dolma 3 corpus has to go
        # the shard route and carries a positional caveat this one does not.
        "sample_via": "rows",
        "text_column": "text",
        "sample_scope": (
            "The deduplicated Pile, which is what every `-deduped` Pythia saw. "
            "The plain Pythia models trained on the non-deduplicated Pile, whose "
            "original distribution is no longer downloadable."
        ),
        "composition": PILE_COMPOSITION,
        "composition_unit": "bytes",
        "composition_scope": (
            "EleutherAI published this breakdown for the Pile as assembled, "
            "before deduplication. Dedup removed about 30% of the bytes and "
            "did not remove them evenly, so these are the shares of the corpus "
            "this one was derived from, not of the corpus sampled below."
        ),
        "hf_dataset": None,
    },
]

MODELS = {
    "olmo-3-7b-instruct": {
        "hf_model": "allenai/Olmo-3-7B-Instruct",
        "stages": OLMO3_7B_BASE_STAGES
        + [
            {
                "stage": "sft",
                "name": "Dolci Instruct SFT",
                "hf_dataset": "allenai/Dolci-Instruct-SFT",
                "prompt_path": "messages",
                "source_columns": ["domain", "source_dataset"],
            },
            {
                "stage": "dpo",
                "name": "Dolci Instruct DPO",
                "hf_dataset": "allenai/Dolci-Instruct-DPO",
                "prompt_path": "chosen_messages",
                "source_columns": ["preference_type"],
            },
            {
                "stage": "rlvr",
                "name": "Dolci Instruct RL",
                "hf_dataset": "allenai/Dolci-Instruct-RL",
                "prompt_path": "prompt",
                "source_columns": ["dataset_source", "data_source"],
            },
        ],
    },
    "olmo-3-7b-think": {
        "hf_model": "allenai/Olmo-3-7B-Think",
        "stages": OLMO3_7B_BASE_STAGES
        + [
            {
                "stage": "sft",
                "name": "Dolci Think SFT 7B",
                "hf_dataset": "allenai/Dolci-Think-SFT-7B",
                "prompt_path": "messages",
                "source_columns": ["dataset_source"],
            },
            {
                "stage": "dpo",
                "name": "Dolci Think DPO 7B",
                "hf_dataset": "allenai/Dolci-Think-DPO-7B",
                "prompt_path": "prompt",
                "source_columns": ["dataset_source", "preference_type"],
            },
            {
                "stage": "rlvr",
                "name": "Dolci Think RL 7B",
                "hf_dataset": "allenai/Dolci-Think-RL-7B",
                "prompt_path": "prompt",
                "source_columns": ["dataset_source", "dataset"],
            },
        ],
    },
    "olmo-3-32b-think": {
        "hf_model": "allenai/Olmo-3-32B-Think",
        "stages": OLMO3_32B_BASE_STAGES
        + [
            {
                "stage": "sft",
                "name": "Dolci Think SFT 32B",
                "hf_dataset": "allenai/Dolci-Think-SFT-32B",
                "prompt_path": "messages",
                # This mix names the column `source`; the 7B SFT mix names the
                # same thing `dataset_source`. Listing only the latter left the
                # 32B SFT sources table empty, which reads as "this mix has no
                # source labels" rather than as a column-name mismatch. Only
                # columns the dataset actually has belong here — a name that
                # addresses nothing is silently dropped by `sources`, so it can
                # only hide the next mismatch.
                "source_columns": ["source"],
            },
            {
                "stage": "dpo",
                "name": "Dolci Think DPO 32B",
                "hf_dataset": "allenai/Dolci-Think-DPO-32B",
                "prompt_path": "prompt",
                # `dataset` here, `dataset_source` in the 7B DPO mix, same values.
                "source_columns": ["dataset", "preference_type"],
            },
            {
                "stage": "rlvr",
                "name": "Dolci Think RL 32B",
                "hf_dataset": "allenai/Dolci-Think-RL-32B",
                "prompt_path": "prompt",
                "source_columns": ["dataset_source", "dataset"],
            },
        ],
    },
    "pythia-12b-deduped": {
        "hf_model": "EleutherAI/pythia-12b-deduped",
        "stages": PYTHIA_DEDUPED_STAGES,
        # The one registered model whose pretraining corpus has a public
        # infini-gram index. See `infinigram_index`.
        "infinigram_index": "v4_piletrain_llama",
    },
}

# The infini-gram index that stands closest to a model's pretraining corpus, for
# a command that has to pick one without being told. No public index covers
# Dolma 3, so every OLMo 3 model falls back to an OLMo 2 index — a different
# corpus, and `infinigram.caveat_for` says so on every run that uses it. A model
# whose entry names its own index (Pythia) gets that instead.
#
# The pretraining-only index, not the full-training-data one `find` defaults to.
# `contaminate` measures the post-training side exactly, by reading the model's
# own mixes, and uses this index for the other side — the web. An index that
# folds in Dolmino midtraining and Tulu 3 post-training would hand back hits
# from another model's post-training and call them corpus: the GSM8K probe
# "Janet sells duck eggs" counts 2 in the full index and 0 in this one.
FALLBACK_INFINIGRAM_INDEX = "v4_olmo-mix-1124_llama"


def infinigram_index(target: dict) -> str | None:
    """The corpus index to search for a target, or None for a dataset.

    A dataset has no pretraining behind it, so there is nothing to search: a
    count over some corpus would describe a model nobody named.
    """
    if not target.get("is_model"):
        return None
    return target.get("infinigram_index", FALLBACK_INFINIGRAM_INDEX)

# The shape of the training example a prompt was drawn from. It decides what
# `context` stores behind the prompt and whether `classify` may read a label off
# the row's verifier, and for a dataset it is also the stage token in the result
# filenames (results/wildchat-1m.chat.labels.json).
#
# A model stage takes its kind from its own `stage` name; the three post-training
# stage names are kinds. A dataset has no pipeline position to borrow one from,
# so it states its kind outright — and gets `chat` when it is a conversation log
# rather than anything a model was fit to.
KINDS = ("sft", "dpo", "rlvr", "chat")

# Datasets explorable in their own right, with no model wrapped around them.
# Same schema hints a post-training stage carries, plus the kind and a note.
DATASETS = {
    "wildchat-1m": {
        "name": "WildChat-1M",
        "hf_dataset": "allenai/WildChat-1M",
        "kind": "chat",
        "prompt_path": "conversation",
        "source_columns": ["model", "language", "country", "redacted"],
        "note": (
            "Real conversations between people and ChatGPT, collected by Ai2 "
            "in exchange for free access. Not a training mix: nothing was fit "
            "to these responses. Olmo's Dolci mixes draw on it — the Instruct "
            "SFT mix takes 302,406 of its prompts, the Think mixes take "
            "regenerated variants — so it is what a large slice of Olmo's "
            "post-training data started as. 837,989 conversations, and `toxic` "
            "is false on every one HuggingFace's stats API reached: the toxic "
            "conversations are held back in the gated WildChat-1M-Full."
        ),
    },
}

for _key, _d in DATASETS.items():
    # A kind outside the set would fall through context.build's dispatch to the
    # RL branch and store an empty verifier record for every sampled row.
    assert _d["kind"] in KINDS, f"{_key}: unknown kind {_d['kind']!r}"


def resolve(name: str) -> dict:
    """A target — model or dataset — in the one shape every command reads.

    Both carry `stages`; a dataset's is a single post-training stage built from
    its own fields, so `post_training_stages` and everything downstream of it
    work on a dataset without a special case. `is_model` is what the two things
    that genuinely differ read: `facts` has no pipeline to print for a dataset,
    and the site's cross-model compare has nothing to put one on an axis with.
    """
    key = name.lower()
    if key in MODELS:
        return {"target": key, "is_model": True, **MODELS[key]}
    if key in DATASETS:
        d = DATASETS[key]
        return {
            "target": key,
            "is_model": False,
            "hf_model": None,
            "name": d["name"],
            "note": d.get("note"),
            "stages": [
                {
                    "stage": d["kind"],
                    "kind": d["kind"],
                    "name": d["name"],
                    "note": d.get("note"),
                    "hf_dataset": d["hf_dataset"],
                    "prompt_path": d["prompt_path"],
                    "source_columns": d["source_columns"],
                }
            ],
        }
    raise KeyError(
        f"Unknown model or dataset {name!r}."
        f" Models: {', '.join(sorted(MODELS))}."
        f" Datasets: {', '.join(sorted(DATASETS))}."
    )


def targets() -> list[str]:
    """Every name `resolve` accepts, models first."""
    return sorted(MODELS) + sorted(DATASETS)


def stage_kind(stage: dict) -> str:
    """The shape of this stage's training examples.

    A model stage's pipeline position is its kind; a dataset states one. Reading
    it through here is what lets a dataset be sampled by a stage name that is not
    one of sft/dpo/rlvr.
    """
    return stage.get("kind") or stage["stage"]


def post_training_stages(target: dict) -> list[dict]:
    return [s for s in target["stages"] if s.get("hf_dataset")]


def pretrain_stages(target: dict) -> list[dict]:
    """Stages sampled as a corpus of documents rather than as training examples.

    What separates these from `post_training_stages` is not the transport but
    the shape of what comes back: a corpus document has no prompt, no response
    and no verifier, so none of the layers built on those apply to it. Two
    stages here can still be read by different routes — see `sample_route`.
    """
    return [s for s in target["stages"] if s.get("sample_dataset")]


def sample_route(stage: dict) -> str:
    """How to draw documents from a pretraining stage's corpus.

    "shards" reads the repo's .jsonl.zst files by HTTP range request. It exists
    because the datasets-server indexes only the first ~5 GB of a Dolma 3 repo
    and those shards are ordered by topic cluster, so paging /rows tours
    whichever clusters sort first. The cost is a positional bias the result
    files state on every run: a range request only reaches a shard's head.

    "rows" pages the datasets-server, and is *better* where it is available —
    uniform over the whole corpus, no shard listing, no position caveat. It is
    only correct for a repo the server has indexed in full (`partial: false`),
    which of the registered corpora is the deduplicated Pile alone.

    Defaulting to shards keeps the choice explicit at the one place it is safe
    to make: a stage that forgets to declare it gets the route with the honest
    caveat rather than the one that silently samples 5 GB of a 450 GB corpus.
    """
    return stage.get("sample_via", "shards")
