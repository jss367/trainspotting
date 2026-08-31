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

Model facts sourced from the Olmo 3 paper (arXiv:2512.13961) and the Ai2 release
blog.
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
#   sample_dataset: the repo of .jsonl.zst shards the pretrain sampler reads by
#       HTTP range request. Distinct from `hf`, which is only a link: this one
#       has to be a repo whose shard layout the sampler understands.
#   sample_scope: what that repo is relative to what the model actually saw.

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
}

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
    """Stages whose corpora are shard repos rather than datasets-server datasets."""
    return [s for s in target["stages"] if s.get("sample_dataset")]
