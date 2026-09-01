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


# Output a message can carry beside its `content`. `search` reads these as part
# of the turn; this record keeps none of them, so a turn holding any is not
# stored whole however well its text matches.
BESIDE_CONTENT = ("reasoning_content",) + search.STRUCTURED_TURN_FIELDS + search.INPUT_TURN_FIELDS


def _turns(messages) -> list[dict]:
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        # `is not None` rather than truthiness: a falsy content that is not
        # absent, `""` or a bare `0`, is still what the turn said.
        raw_content = m.get("content")
        content = "" if raw_content is None else str(raw_content)
        omitted = [k for k in BESIDE_CONTENT if m.get(k)]
        # Every message in the list is kept, including one that is nothing but a
        # tool call and one that is empty. Both are turns in the sequence the
        # model was scored on — a message with no content still contributes its
        # role header and end-of-turn token — and `_shared_turns` branches at one
        # that appears on a single side. Dropping either would close the gap it
        # leaves: two completions differing only there would read as the same
        # conversation, and the answers behind them as a shared opening. They are
        # kept, empty, so the turn counts stay aligned and nothing about them is
        # marked as stored whole.
        reasoning, answer = _split_think(content)
        turn = {"role": m.get("role", "?"), **_text(answer)}
        if reasoning:
            turn["reasoning"] = _text(reasoning)
        if omitted:
            turn["omitted"] = omitted
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
        # Structured content — a list of parts, a dict — survives `str()` as a
        # Python repr, which is a serialization of the turn and not the text the
        # model was scored on, so only a string (or an absent content, faithfully
        # empty) can be stored as written.
        stored_as_written = raw_content is None or isinstance(raw_content, str)
        if stored_as_written and turn["text"] == content and not omitted:
            turn["raw"] = True
        elif stored_as_written:
            # Not stored as written, so the halves above no longer add up to
            # what the model read: the `<think>` markers and the whitespace
            # around them are gone, and a long field is cut. `derive` measures
            # a turn by summing those halves and would understate it — by 0.04%
            # of the Think SFT characters and 0.16% of the DPO ones, measured.
            # `raw` says whether that happened; this says by how much. Keyed off
            # the same flag rather than off a non-empty reasoning field, because
            # an empty thinking span loses its markers and keeps no field to
            # show it — a 125-character turn opening `<think>\n\n</think>` was
            # measured as 106.
            turn["chars_raw"] = len(content)
        out.append(turn)
    return out


def _turn_key(turn: dict) -> tuple:
    """A stored turn reduced to what makes it the same turn as another.

    `text` is cut at MAX_TEXT and `chars` is not, so two different turns that
    agree on their first 4,000 characters still differ here.
    """
    reasoning = turn.get("reasoning") or {}
    return (
        turn.get("role"),
        turn.get("text"),
        turn.get("chars"),
        reasoning.get("text"),
        reasoning.get("chars"),
    )


def branch_point(chosen: list[dict], rejected: list[dict]) -> int:
    """How many leading turns of a stored pair are shared history.

    A multi-turn pair branches somewhere and shares everything before it,
    assistant turns included. Those shared turns are the conversation the pair
    is judged in, not either candidate answer, so splitting the pair by role
    puts earlier assistant turns on both sides — text neither completion is
    being preferred for. `search` draws the same line on raw rows
    (`search._shared_turns`); this is it for the records `context` stores.

    The last turn of each side is its candidate answer by definition and is
    never shared however far the two lists agree, so a pair whose completions
    are identical branches at the final turn rather than having no completions
    at all.

    12 of the 300 sampled Dolci-Instruct-DPO pairs are multi-turn, and counting
    their shared history on both sides inflates the stage's fit characters by
    5.9%. The think mixes are single-turn throughout, so nothing there moves.
    """
    n = 0
    for a, b in zip(chosen or [], rejected or []):
        if _turn_key(a) != _turn_key(b):
            break
        n += 1
    return min(n, max(0, len(chosen or []) - 1), max(0, len(rejected or []) - 1))


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
    # Every accepted answer, not the first of them. `ground_truth` is a list on
    # the RL mixes that accept more than one form of the answer, and keeping
    # only its first element made the rest unfindable on the site while
    # `search` and `grep` both matched them — a hit the whole-mix count could
    # prove was there and the drill-down said was not.
    #
    # `search.flatten` rather than a join written here, because "how does a cell
    # become searchable text" is one question and this is the third layer to ask
    # it. It joins scalars with newlines and renders anything structured as
    # JSON, so the record holds what a search would have matched against.
    gt = row.get("ground_truth")
    if gt is None or gt == [] or gt == "":
        gt = rm.get("ground_truth")
    ground_truth = search.flatten(gt)
    return {
        "kind": kind,
        "explain": explain,
        "style": rm.get("style"),
        "ground_truth": _text(ground_truth) if ground_truth else None,
        "solution": _text(row["solution"]) if row.get("solution") else None,
        "constraint": _text(row["constraint"]) if row.get("constraint") else None,
        "constraint_type": row.get("constraint_type"),
    }


def build(row: dict, kind: str, prompt: str, row_index: int, source_columns=()) -> dict:
    """One kind-appropriate context record for an already-sampled row.

    `kind` is `registry.stage_kind` of the stage the row came from — the shape
    of the training example, which for a model stage is its pipeline position
    and for a standalone dataset is what the registry declares it to be.

    `source_columns` is the stage's own provenance columns from the registry.
    The per-kind lists below are the columns these mixes happen to use, and a
    mix that names its provenance something else — Dolci Think 32B calls it
    `source` where the 7B mixes call it `source_dataset` — dropped out of the
    record entirely, which the site reads as a stage that records nothing about
    where its examples came from. The registry already knows the answer, so ask
    it rather than growing the list every time a mix is added.

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
        rec["meta"] = _meta(row, [*source_columns, "source_dataset", "domain", "dataset_source"])
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
        rec["meta"] = _meta(row, [*source_columns, "preference_type", "dataset_source"])
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
            row,
            [*source_columns, "dataset_source", "data_source", "ability", "difficulty", "setting_name"],
        )
    return rec
