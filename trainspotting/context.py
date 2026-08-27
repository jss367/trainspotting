"""Pull the full training context out of a sampled row.

A prompt is half of a training example. What the model is actually trained
toward differs by stage, so each stage yields a different record:

    sft   the target conversation — the model is fit to the assistant turns
    dpo   a preferred and a dispreferred response, plus where each came from
    rlvr  no stored response at all — a verifier, what it checks, and how often
          rollouts from the reference model passed it

Records are keyed by the same prompt text the classifier saw, so the site can
join them onto committed label and ask results without re-running any model.
"""

from . import extract

MAX_TEXT = 4000  # per field; the full row stays one click away on HuggingFace


def _text(value) -> dict:
    """A text field plus its true length, so truncation is visible and lengths stay honest."""
    s = "" if value is None else str(value)
    return {"text": s[:MAX_TEXT], "chars": len(s)}


def _split_think(text: str) -> tuple[str | None, str]:
    """Separate a thinking span from the answer it precedes.

    Think models put most of their length in the reasoning, so splitting before
    truncation is what keeps the answer visible at all.
    """
    i = text.find("</think>")
    if i < 0:
        return None, text
    head = text[:i].lstrip()
    if head.startswith("<think>"):
        head = head[len("<think>") :]
    return head.strip(), text[i + len("</think>") :].strip()


def _turns(messages) -> list[dict]:
    out = []
    for m in messages or []:
        if not (isinstance(m, dict) and m.get("content")):
            continue
        reasoning, answer = _split_think(str(m["content"]))
        turn = {"role": m.get("role", "?"), **_text(answer)}
        if reasoning:
            turn["reasoning"] = _text(reasoning)
        out.append(turn)
    return out


def _meta(row: dict, keys: list[str]) -> dict:
    return {k: row[k] for k in keys if row.get(k) not in (None, "", [], {})}


# What the reward actually checks, by which mix the prompt came from. The raw
# dataset_source travels with the record so the inference stays checkable.
REWARD_KINDS = [
    (
        ("if_multi_constraints", "constraint"),
        "constraint checker",
        "A program checks the response against the constraints listed below. "
        "Reward 1 when every constraint holds, 0 otherwise.",
    ),
    (
        ("acecoder", "code_rlvr", "python"),
        "unit tests",
        "The response's code is executed against test cases. Reward is the fraction "
        "of tests that pass.",
    ),
    (
        ("math", "omega", "polaris", "orz", "dapo", "gsm"),
        "exact answer match",
        "The final answer is extracted from the response and compared to the ground "
        "truth below. Reward 1 on a match, 0 otherwise.",
    ),
    (
        ("general_mix", "general-mix", "wildchat", "chat"),
        "LLM judge",
        "A judge model scores the response. The rubric it uses is not published with "
        "the dataset, so the prompt text is all this tool can show you.",
    ),
]


def _reward(row: dict) -> dict:
    tags = " ".join(
        str(row.get(k) or "").lower()
        for k in ("dataset_source", "data_source", "original_dataset", "ability", "constraint_type")
    )
    kind, explain = "unknown", "This mix's reward function isn't identifiable from the row's own fields."
    for needles, k, e in REWARD_KINDS:
        if any(n in tags for n in needles):
            kind, explain = k, e
            break
    rm = row.get("reward_model") or {}
    gt = row.get("ground_truth")
    if isinstance(gt, list):
        gt = next((g for g in gt if g), None)
    return {
        "kind": kind,
        "explain": explain,
        "style": rm.get("style"),
        "ground_truth": _text(gt or rm.get("ground_truth")) if (gt or rm.get("ground_truth")) else None,
        "solution": _text(row["solution"]) if row.get("solution") else None,
        "constraint": _text(row["constraint"]) if row.get("constraint") else None,
        "constraint_type": row.get("constraint_type"),
    }


def build(row: dict, stage: str, prompt_path: str, prompt: str, row_index: int) -> dict:
    """One stage-appropriate context record for an already-sampled row.

    `prompt` is the classifier's copy (the join key); `prompt_full` is the same
    prompt cut at this module's larger cap, so the site can show text the
    classifier's copy had already lost.
    """
    rec = {
        "prompt": prompt,
        "prompt_full": _text(extract.extract_prompt(row, prompt_path, MAX_TEXT) or prompt),
        "row": row_index,
        "id": row.get("id") or row.get("prompt_id") or row.get("custom_id"),
    }
    if stage == "sft":
        rec["kind"] = "sft"
        rec["turns"] = _turns(row.get("messages"))
        rec["meta"] = _meta(row, ["source_dataset", "domain", "dataset_source"])
    elif stage == "dpo":
        rec["kind"] = "dpo"
        rec["chosen"] = {"model": row.get("chosen_model"), "turns": _turns(row.get("chosen"))}
        rec["rejected"] = {"model": row.get("rejected_model"), "turns": _turns(row.get("rejected"))}
        rec["meta"] = _meta(row, ["preference_type", "dataset_source"])
    else:
        rec["kind"] = "rlvr"
        rec["reward"] = _reward(row)
        outputs = [o for o in (row.get("outputs") or []) if o]
        rec["rollouts"] = {
            "total": row.get("total_rollouts"),
            "correct": row.get("total_correct_rollouts"),
            "passrate": row.get("passrate"),
            "sample": _text(outputs[0]) if outputs else None,
        }
        rec["meta"] = _meta(
            row, ["dataset_source", "data_source", "ability", "difficulty", "setting_name"]
        )
    return rec
