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

import re

PAD = 100  # snippet characters kept either side of a hit

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
    (`ground_truth`, `outputs`), or None."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_flatten(v) for v in value)
    return str(value)


def _turns(messages):
    """(turn index, role, text) for each message turn that carries text.

    `reasoning_content` is yielded as its own turn rather than folded into the
    content: it is a separate column in the WildChat-derived mixes, so text that
    lives only there would otherwise never be searched.
    """
    for i, m in enumerate(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        if m.get("content"):
            yield i, role, str(m["content"])
        if m.get("reasoning_content"):
            yield i, f"{role}/reasoning", str(m["reasoning_content"])


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
            add("response" if role.startswith("assistant") else "prompt", role, text, i)
    elif stage == "dpo":
        if isinstance(row.get("prompt"), list):
            for i, role, text in _turns(row["prompt"]):
                add("prompt", role, text, i)
        else:
            add("prompt", "prompt", row.get("prompt"))
        for side in ("chosen", "rejected"):
            for i, role, text in _turns(row.get(side)):
                add(side if role.startswith("assistant") else "prompt", role, text, i)
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
