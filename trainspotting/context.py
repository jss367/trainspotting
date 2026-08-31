"""Pull the full training context out of a sampled row.

A prompt is half of a training example. What the model is actually trained
toward differs by the kind of example, so each kind yields a different record:

    sft   the target conversation — the model is fit to the assistant turns
    dpo   a preferred and a dispreferred response, plus where each came from
    rlvr  no stored response at all — a verifier, what it checks, and how often
          rollouts from the reference model passed it
    chat  a conversation log, where nothing was trained on anything — the other
          side of the exchange is what was said, not a target

Records are keyed by the same prompt text the classifier saw, so the site can
join them onto committed label and ask results without re-running any model.
"""

from trainspotting import rewards, search

MAX_TEXT = 4000  # per field; the full row stays one click away on HuggingFace
KEY_CHARS = 400  # prompt prefix that joins a context record to a labeled prompt


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
        content = str(m["content"])
        reasoning, answer = _split_think(content)
        turn = {"role": m.get("role", "?"), **_text(answer)}
        if reasoning:
            turn["reasoning"] = _text(reasoning)
        # Whether what is stored is the turn as it was written. Splitting a
        # thinking span out drops the <think> markers and the whitespace around
        # them, and long fields are cut, so a turn that went through either can no
        # longer be compared byte for byte with the sequence the model was scored
        # on. The absence of a reasoning field does not say this on its own: a
        # turn whose thinking span was empty loses its markers and keeps no field
        # to show it. Nor does `content` alone — a message can carry output beside
        # it, a separate reasoning field or tool calls or a refusal, which
        # `search` reads as part of the turn and this record does not keep.
        # Anything claiming two turns are identical needs this, not a guess.
        beside_content = ("reasoning_content",) + search.STRUCTURED_TURN_FIELDS + search.INPUT_TURN_FIELDS
        if turn["text"] == content and not any(m.get(k) for k in beside_content):
            turn["raw"] = True
        out.append(turn)
    return out


def _meta(row: dict, keys: list[str]) -> dict:
    return {k: row[k] for k in keys if row.get(k) not in (None, "", [], {})}


def _reward(row: dict) -> dict:
    # What the reward checks comes from the mix→verifier table in rewards.py.
    # The raw dataset_source travels with the record so the inference stays
    # checkable; the site re-derives the explanation from the kind, so the
    # baked text here only serves offline readers of the JSON.
    kind = rewards.kind_for(row)
    explain = rewards.KINDS[kind]["explain"]
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


def build(row: dict, kind: str, prompt: str, row_index: int) -> dict:
    """One kind-appropriate context record for an already-sampled row.

    `kind` is `registry.stage_kind` of the stage the row came from — the shape
    of the training example, which for a model stage is its pipeline position
    and for a standalone dataset is what the registry declares it to be.

    `row` is the join: a result record stores the same index, and the site looks
    the context up by it. `key` is the older prompt-prefix join, kept for runs
    committed before result records carried a row — two rows sharing a
    400-character opening collapse to the first of them under it, which a
    curated mix mostly gets away with and a chat log does not.
    """
    rec = {
        "key": prompt[:KEY_CHARS],
        "prompt_full": _text(prompt),
        "row": row_index,
        "id": row.get("id") or row.get("prompt_id") or row.get("custom_id"),
    }
    if kind == "sft":
        rec["kind"] = "sft"
        rec["turns"] = _turns(row.get("messages"))
        rec["meta"] = _meta(row, ["source_dataset", "domain", "dataset_source"])
    elif kind == "chat":
        # A log, not a training example: no turn here is a target, so the record
        # is the exchange and the metadata the collector recorded around it.
        rec["kind"] = "chat"
        rec["turns"] = _turns(row.get("conversation"))
        rec["meta"] = _meta(
            row, ["model", "language", "country", "state", "turn", "toxic", "redacted"]
        )
    elif kind == "dpo":
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
