"""Hardcoded per-model training-pipeline registry.

Each model maps to an ordered list of stages. Pretraining-scale stages carry
facts only (the corpora are terabytes; we don't compute over them). Post-training
stages carry a HuggingFace dataset ID plus schema hints so the generic code can
pull prompts and source tags out of heterogeneous Dolci schemas.

Facts sourced from the Olmo 3 paper (arXiv:2512.13961) and the Ai2 release blog.
"""

# Schema hints:
#   prompt_path: how to pull the user prompt out of a row.
#     "messages"        -> first role=="user" content in row["messages"]
#     "chosen_messages" -> first role=="user" content in row["chosen"]
#     "prompt"          -> row["prompt"] (string, or list of chat messages)
#   source_columns: columns whose value counts describe the mix composition.

#   sample_dataset: a Dolma 3 repo of .jsonl.zst shards that the pretrain
#     sampler reads by HTTP range request. These stages keep hf_dataset=None
#     because the datasets-server layers (sources, context) cannot serve them:
#     its index covers only the first ~5 GB and the shards are topic-ordered.
#   sample_scope: what the sampled repo is relative to what the model saw.

SEVEN_B_MIX_CAVEAT = (
    "The exact mix used to pretrain Olmo 3 7B. Some olmOCR science PDFs were "
    "redacted to `[REMOVED]` after training, so a few documents here show a "
    "placeholder where the model saw real text."
)


def olmo3_pretrain_stages(pretrain_mix: str, pretrain_scope: str) -> list[dict]:
    """The three pre-post-training stages, differing only in which 6T mix was used.

    Olmo 3 published one pretraining mix per model size; midtraining and the
    long-context extension are shared.
    """
    return [
        {
            "stage": "pretrain",
            "name": "Dolma 3 Mix",
            "tokens": 5_900_000_000_000,
            "note": "Sampled from the ~9.3T-token Dolma 3 pool (web, olmOCR science PDFs, code, math, encyclopedic).",
            "hf_dataset": None,
            "sample_dataset": pretrain_mix,
            "sample_scope": pretrain_scope,
        },
        {
            "stage": "midtrain",
            "name": "Dolma 3 Dolmino Mix",
            "tokens": 100_000_000_000,
            "note": "Sampled from a ~2.2T-token pool of math, science, code, IF, and reading-comprehension data.",
            "hf_dataset": None,
            "sample_dataset": "allenai/dolma3_dolmino_mix-100B-1125",
            "sample_scope": "The 100B-token Dolmino mix itself.",
        },
        {
            "stage": "long-context",
            "name": "Dolma 3 Longmino Mix",
            "tokens": 50_000_000_000,
            "note": "Long-context extension mix.",
            "hf_dataset": None,
            "sample_dataset": "allenai/dolma3_longmino_mix-100B-1125",
            "sample_scope": "The 100B-token Longmino mix; the 7B run drew 50B tokens from it.",
        },
    ]


OLMO3_PRETRAIN_STAGES = olmo3_pretrain_stages(
    "allenai/dolma3_mix-6T-1025-7B", SEVEN_B_MIX_CAVEAT
)
OLMO3_32B_PRETRAIN_STAGES = olmo3_pretrain_stages(
    "allenai/dolma3_mix-6T",
    "The primary Dolma 3 6T mix, used to pretrain Olmo 3 32B. No redactions.",
)

MODELS = {
    "olmo-3-7b-instruct": {
        "hf_model": "allenai/Olmo-3-7B-Instruct",
        "stages": OLMO3_PRETRAIN_STAGES
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
        "stages": OLMO3_PRETRAIN_STAGES
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
        "stages": OLMO3_32B_PRETRAIN_STAGES
        + [
            {
                "stage": "sft",
                "name": "Dolci Think SFT 32B",
                "hf_dataset": "allenai/Dolci-Think-SFT-32B",
                "prompt_path": "messages",
                "source_columns": ["dataset_source"],
            },
            {
                "stage": "dpo",
                "name": "Dolci Think DPO 32B",
                "hf_dataset": "allenai/Dolci-Think-DPO-32B",
                "prompt_path": "prompt",
                "source_columns": ["dataset_source", "preference_type"],
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


def get_model(name: str) -> dict:
    key = name.lower()
    if key not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(sorted(MODELS))}")
    return MODELS[key]


def post_training_stages(model: dict) -> list[dict]:
    return [s for s in model["stages"] if s.get("hf_dataset")]


def pretrain_stages(model: dict) -> list[dict]:
    """Stages whose corpora are shard repos rather than datasets-server datasets."""
    return [s for s in model["stages"] if s.get("sample_dataset")]
