"""Find a string anywhere in a training example, and say which side it landed on.

The classify, ask and languages layers all read the prompt. A behaviour like
claiming to be ChatGPT is not in the prompt: it is in what the model is fit to —
the assistant turns of an SFT example, the chosen completion of a DPO pair — so
a prompt-only search reports zero for a string the model was trained to say.

Every hit therefore carries the side of the example it landed on, because the
same string means opposite things on different sides. "I am ChatGPT" in a DPO
chosen completion trains the model toward saying it; in the rejected completion
it trains the model away from it. A count that adds the two together is worse
than no count.

The column shapes come from the same place `context.py` reads them, one stage at
a time:

    sft   messages       -> prompt (user/system turns), response (assistant turns)
    dpo   chosen/rejected-> prompt (the shared turns), chosen, rejected
    rlvr  prompt         -> prompt, verifier (what is checked), rollout (reference
                            generations — no response is stored for these rows)
"""

import json
import re

PAD = 100  # snippet characters kept either side of a hit

# Message columns that can hold what the model produced instead of `content`: an
# OpenAI-style tool call, a function call, a refusal string. They are their own
# columns in the WildChat-derived and Instruct SFT schemas, so a search reading
# `content` alone would report zero for a tool name rather than "not in this
# sample". `functions` is the menu of tools a turn was offered rather than
# anything the model said, so it is searched as prompt text whatever role the
# turn carries.
STRUCTURED_TURN_FIELDS = ("tool_calls", "function_call", "function_calls", "refusal")
INPUT_TURN_FIELDS = ("functions",)

# Every side a stage can produce, in reporting order. Sides are named for what
# the example does with the text, not for the column it sits in: `chosen` and
# `rejected` are both assistant turns, and that is the whole distinction.
SIDES = {
    "sft": ("prompt", "response"),
    "dpo": ("prompt", "chosen", "rejected"),
    "rlvr": ("prompt", "verifier", "rollout"),
}


def _flatten(value) -> str:
    """A cell's searchable text. Cells arrive as strings, lists of strings
    (`ground_truth`, `outputs`), structured payloads (a tool call), or None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in value):
        return "\n".join(str(v) for v in value)
    if isinstance(value, (list, dict)):
        # A tool call is a name, its arguments and their values. Rendering it as
        # JSON makes all three searchable, and keeps the field names visible so
        # a hit can be read.
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _turns(messages):
    """(turn index, role, text) for each piece of text a message turn carries.

    A turn is more than its `content`. `reasoning_content` and the structured
    output columns are yielded as their own entries, named for the column they
    came from, because text that lives only there would otherwise never be
    searched — and a hit in a tool call is a different reading of the example
    from a hit in the prose.
    """
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        if m.get("content"):
            yield i, role, str(m["content"])
        if m.get("reasoning_content"):
            yield i, f"{role}/reasoning", str(m["reasoning_content"])
        for name in STRUCTURED_TURN_FIELDS + INPUT_TURN_FIELDS:
            if m.get(name):
                yield i, f"{role}/{name}", _flatten(m[name])


def _side_of(role: str, response_side: str) -> str:
    """Which side of the example a turn's text belongs to.

    The role decides it — an assistant turn is what the model is fit to — except
    for the tool definitions a turn offers: those are given to the model, not
    produced by it, whichever turn they hang off.
    """
    if role.endswith(tuple(f"/{name}" for name in INPUT_TURN_FIELDS)):
        return "prompt"
    return response_side if role.startswith("assistant") else "prompt"


def fields(row: dict, stage: str) -> list[dict]:
    """Every searchable text field of one row, tagged with its side and role.

    Reasoning spans are left inside the assistant content they are part of
    (unlike `context.build`, which folds them away for display): a model that
    says it is ChatGPT while thinking still said it.
    """
    out: list[dict] = []
    seen_prompt: set[str] = set()

    def add(side, role, value, turn=None):
        text = _flatten(value)
        if not text.strip():
            return
        if side == "prompt":
            # A DPO row repeats its prompt on both sides, and some RL rows carry
            # it as both `prompt` and `source_prompt`. Counting the same text
            # twice would double every prompt hit in those stages.
            if text in seen_prompt:
                return
            seen_prompt.add(text)
        out.append({"side": side, "role": role, "turn": turn, "text": text})

    if stage == "sft":
        for i, role, text in _turns(row.get("messages")):
            add(_side_of(role, "response"), role, text, i)
    elif stage == "dpo":
        if isinstance(row.get("prompt"), list):
            for i, role, text in _turns(row["prompt"]):
                add("prompt", role, text, i)
        else:
            add("prompt", "prompt", row.get("prompt"))
        for side in ("chosen", "rejected"):
            for i, role, text in _turns(row.get(side)):
                add(_side_of(role, side), role, text, i)
    else:
        if isinstance(row.get("prompt"), list):
            for i, role, text in _turns(row["prompt"]):
                add("prompt", role, text, i)
        else:
            add("prompt", "prompt", row.get("prompt"))
        for i, role, text in _turns(row.get("source_prompt")):
            add("prompt", role, text, i)
        # What the verifier scores against. Not a response — no RL row stores
        # one — but it is the rest of what the example teaches, and a string in
        # the answer key is a different finding from the same string in the
        # question.
        for name in ("ground_truth", "solution", "constraint"):
            add("verifier", name, row.get(name))
        add("verifier", "reward_model.ground_truth", (row.get("reward_model") or {}).get("ground_truth"))
        # Reference-model generations stored with the row. The model is not fit
        # to these — they are what the rollouts scored — so they get their own
        # side rather than being reported as a response.
        for i, output in enumerate(row.get("outputs") or []):
            add("rollout", "output", output, i)
    return out


def snippet(text: str, match: re.Match, pad: int = PAD) -> str:
    """The match plus its surroundings, with elisions marked."""
    lo = max(0, match.start() - pad)
    hi = min(len(text), match.end() + pad)
    return ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def search_row(row: dict, stage: str, pattern: re.Pattern) -> list[dict]:
    """One hit record per matching field of a row; empty if the row does not match."""
    hits = []
    for field in fields(row, stage):
        matches = list(pattern.finditer(field["text"]))
        if matches:
            hits.append(
                {
                    "side": field["side"],
                    "role": field["role"],
                    "turn": field["turn"],
                    "count": len(matches),
                    "chars": len(field["text"]),
                    "snippet": snippet(field["text"], matches[0]),
                }
            )
    return hits


def side_counts(records: list[dict], stage: str) -> dict[str, int]:
    """Matching rows per side — rows, not hits, so a string repeated forty times
    in one response counts once."""
    counts = {side: 0 for side in SIDES[stage]}
    for rec in records:
        for side in {h["side"] for h in rec["hits"]}:
            counts[side] = counts.get(side, 0) + 1
    return counts


def pair_split(records: list[dict]) -> dict[str, int]:
    """How DPO rows divide by which completion holds the string.

    A hit on both sides is the uninformative case — the pair says nothing about
    the string, because it appears whichever way the model answers — and reading
    it as evidence either way is the mistake this breakdown exists to prevent.
    """
    split = {"chosen_only": 0, "rejected_only": 0, "both": 0}
    for rec in records:
        sides = {h["side"] for h in rec["hits"]}
        chosen, rejected = "chosen" in sides, "rejected" in sides
        if chosen and rejected:
            split["both"] += 1
        elif chosen:
            split["chosen_only"] += 1
        elif rejected:
            split["rejected_only"] += 1
    return split
