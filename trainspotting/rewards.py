"""What each RLVR mix's reward actually checks.

The RL datasets don't label their reward functions — each row only names the
mix it came from. But every mix has exactly one verifier, so a mix→verifier
table turns those names into exact reward-type counts. This module is the
single source of that table: the context layer uses it to explain one row,
and the site (via the reward-kinds.json export) uses it to roll a whole
stage's dataset_source counts up into "what fraction of RL is scored by what".

Kind ids double as display strings because committed context records already
store them verbatim; renaming one would orphan every drill-down built before
the rename.
"""

KINDS = {
    "exact answer match": {
        "explain": (
            "The final answer is extracted from the response and compared to the "
            "ground truth below. Reward 1 on a match, 0 otherwise."
        ),
    },
    "unit tests": {
        "explain": (
            "The response's code is executed against test cases. Reward is the "
            "fraction of tests that pass."
        ),
        "gt_label": "test cases the answer is run against",
    },
    "constraint checker": {
        "explain": (
            "A program checks the response against the constraints listed below. "
            "Reward 1 when every constraint holds, 0 otherwise."
        ),
        "gt_label": "checker configuration",
    },
    "LLM judge": {
        "explain": (
            "No rule can grade a free-form answer, so a judge model compares the "
            "response to the stored reference answer and rewards a match. The "
            "grading itself is another model call."
        ),
        "gt_label": "reference answer the judge compares against",
    },
    "unknown": {
        "explain": "This mix's reward function isn't identifiable from the row's own fields.",
    },
}

# Every dataset_source value observed across the three models' RL mixes,
# mapped to its verifier and its subject matter. Exact strings, because the
# same underlying mix appears under different filtered/renamed ids per model.
MIXES = {
    # Dolci Instruct RL
    "allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered": ("LLM judge", "general knowledge"),
    "allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered": ("constraint checker", "instruction following"),
    "hamishivi/omega-combined-no-boxed_filtered": ("exact answer match", "math"),
    "hamishivi/rlvr_acecoder_filtered_filtered": ("unit tests", "code"),
    "hamishivi/polaris_53k": ("exact answer match", "math"),
    "hamishivi/rlvr_orz_math_57k_collected_filtered": ("exact answer match", "math"),
    "hamishivi/MathSub-30K_filtered": ("exact answer match", "math"),
    "hamishivi/DAPO-Math-17k-Processed_filtered": ("exact answer match", "math"),
    # Dolci Think RL 7B
    "hamishivi/math_rlvr_mixture_dpo": ("exact answer match", "math"),
    "hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter": ("constraint checker", "instruction following"),
    "hamishivi/code_rlvr_mixture_dpo": ("unit tests", "code"),
    "hamishivi/rlvr_general_mix": ("LLM judge", "general knowledge"),
    # Dolci Think RL 32B (where it differs from 7B)
    "saurabh5/code_rlvr_mixture_dpo": ("unit tests", "code"),
}

# Fallback for mixes not yet in the table (a future Dolci release), matched
# against every source-ish field on the row. Order matters: the more specific
# needles come first so "code_rlvr" wins before "rlvr" could mean anything.
NEEDLES = [
    (("if_multi_constraints", "constraint"), "constraint checker"),
    (("acecoder", "code_rlvr", "python"), "unit tests"),
    (("math", "omega", "polaris", "orz", "dapo", "gsm"), "exact answer match"),
    (("general_mix", "general-mix", "wildchat", "chat"), "LLM judge"),
]


def kind_for(row: dict) -> str:
    """The verifier kind for one RL row: exact mix lookup, then needle fallback."""
    for k in ("dataset_source", "data_source", "original_dataset"):
        hit = MIXES.get(row.get(k) or "")
        if hit:
            return hit[0]
    tags = " ".join(
        str(row.get(k) or "").lower()
        for k in ("dataset_source", "data_source", "original_dataset", "ability", "constraint_type")
    )
    for needles, kind in NEEDLES:
        if any(n in tags for n in needles):
            return kind
    return "unknown"


def site_export() -> dict:
    """The table as the site consumes it (reward-kinds.json)."""
    return {
        "kinds": KINDS,
        "mixes": {name: {"kind": kind, "subject": subject} for name, (kind, subject) in MIXES.items()},
    }
