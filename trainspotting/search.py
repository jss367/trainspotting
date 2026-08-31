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

# The row columns `fields` reads, per stage. A shortened cell only costs a search
# something when it is one of these: an RL row carries `input_ids`,
# `attention_mask` and `labels`, which are long enough to be what the server cuts
# and hold no text this searches, so a row cut there is still a confirmed
# non-match rather than one that could not be read.
COLUMNS = {
    "sft": ("messages",),
    "dpo": ("prompt", "chosen", "rejected"),
    "rlvr": (
        "prompt",
        "source_prompt",
        "ground_truth",
        "solution",
        "constraint",
        "reward_model",
        "outputs",
    ),
}

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


def _turns(messages, start: int = 0):
    """(turn index, role, text) for each piece of text a message turn carries,
    from turn `start` onward. Indices stay absolute, so a turn keeps the
    position it has in the conversation however the caller sliced it.

    A turn is more than its `content`. `reasoning_content` and the structured
    output columns are yielded as their own entries, named for the column they
    came from, because text that lives only there would otherwise never be
    searched — and a hit in a tool call is a different reading of the example
    from a hit in the prose.
    """
    for i, m in enumerate(messages or []):
        if i < start or not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        if m.get("content"):
            yield i, role, str(m["content"])
        if m.get("reasoning_content"):
            yield i, f"{role}/reasoning", str(m["reasoning_content"])
        for name in STRUCTURED_TURN_FIELDS + INPUT_TURN_FIELDS:
            if m.get(name):
                yield i, f"{role}/{name}", _flatten(m[name])


def _turn_text(m: dict) -> tuple:
    """A turn reduced to the text a search can see, for comparing two branches.

    Only the searchable columns: the Instruct DPO schema hangs per-branch
    metadata off every turn (timestamps, an OpenAI id, the sampling
    temperature), so two turns holding the same words are the same turn here
    even when those differ.
    """
    return (
        m.get("role"),
        _flatten(m.get("content")),
        _flatten(m.get("reasoning_content")),
        *(_flatten(m.get(name)) for name in STRUCTURED_TURN_FIELDS + INPUT_TURN_FIELDS),
    )


def _shared_turns(chosen, rejected) -> int:
    """How many leading turns the two completions of a pair hold in common.

    A multi-turn pair branches at some point and shares everything before it,
    assistant turns included. Those shared turns are the conversation the pair
    is judged in, not either candidate answer: attributing them by role would
    report a string in the shared history as a hit on both completions, which
    `pair_split` then calls `both` — the code for "the pair says nothing about
    this string", claimed here about text neither completion contains.
    """
    n = 0
    for a, b in zip(chosen or [], rejected or []):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            break
        if _turn_text(a) != _turn_text(b):
            break
        n += 1
    return n


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

    def add(side, role, value, turn=None):
        text = _flatten(value)
        if text.strip():
            out.append({"side": side, "role": role, "turn": turn, "text": text})

    if stage == "sft":
        for i, role, text in _turns(row.get("messages")):
            add(_side_of(role, "response"), role, text, i)
    elif stage == "dpo":
        chosen, rejected = row.get("chosen"), row.get("rejected")
        # A pair's last turn is its candidate answer by definition, so it is
        # never shared history however far the two lists agree. Without this a
        # pair whose completions are identical would have every turn read as
        # prompt text and report neither side — when what it actually shows is
        # the string in both candidates, which is what `both` is for.
        shared = min(
            _shared_turns(chosen, rejected),
            max(0, len(chosen or []) - 1),
            max(0, len(rejected or []) - 1),
        )
        # Everything before the branch point is the conversation both
        # completions answer in — prompt text, read once, off the chosen side.
        prefix = set()
        for i, role, text in _turns((chosen or [])[:shared]):
            add("prompt", role, text, i)
            prefix.add(text)
        # The `prompt` column repeats those opening turns verbatim, so it is
        # added only when it says something they do not. Dropping it by text
        # rather than by position keeps a later turn that happens to repeat it:
        # a conversation that says "continue" twice said it twice.
        if isinstance(row.get("prompt"), list):
            for i, role, text in _turns(row["prompt"]):
                if text not in prefix:
                    add("prompt", role, text, i)
        elif _flatten(row.get("prompt")) not in prefix:
            add("prompt", "prompt", row.get("prompt"))
        seen_after = set()
        for side, turns in (("chosen", chosen), ("rejected", rejected)):
            for i, role, text in _turns(turns, start=shared):
                attributed = _side_of(role, side)
                if attributed == "prompt":
                    # Context past the branch point still appears in both
                    # branches. The same turn in the same position is one turn,
                    # not one per branch.
                    if (i, text) in seen_after:
                        continue
                    seen_after.add((i, text))
                add(attributed, role, text, i)
    else:
        column: set[tuple] = set()
        if isinstance(row.get("prompt"), list):
            for i, role, text in _turns(row["prompt"]):
                add("prompt", role, text, i)
                column.add((i, text))
        else:
            add("prompt", "prompt", row.get("prompt"))
            column.add((None, _flatten(row.get("prompt"))))
        for i, role, text in _turns(row.get("source_prompt")):
            # Some RL rows carry the same conversation in both columns. A turn
            # is a copy when it holds the same text at the same position, or
            # when it repeats a string-valued `prompt` outright — not merely
            # when some other turn happened to say the same words.
            if (i, text) in column or (None, text) in column:
                continue
            add("prompt", role, text, i)
        # What the verifier scores against. Not a response — no RL row stores
        # one — but it is the rest of what the example teaches, and a string in
        # the answer key is a different finding from the same string in the
        # question.
        for name in ("ground_truth", "solution", "constraint"):
            add("verifier", name, row.get(name))
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict):
            # Every field of it, not just the ground truth: `style` names what
            # the verifier does and is text a search can legitimately be looking
            # for. `COLUMNS` declares the whole cell searched, so reading part
            # of it would also let a truncation of the unread part censor a row
            # for nothing.
            for name, value in sorted(reward_model.items()):
                add("verifier", f"reward_model.{name}", value)
        else:
            add("verifier", "reward_model", reward_model)
        # Reference-model generations stored with the row. The model is not fit
        # to these — they are what the rollouts scored — so they get their own
        # side rather than being reported as a response.
        for i, output in enumerate(row.get("outputs") or []):
            add("rollout", "output", output, i)
    return out


def truncated_columns(stage: str, truncated_cells) -> list[str]:
    """Which of the cells the server shortened this stage actually searches."""
    return sorted(set(truncated_cells or ()) & set(COLUMNS[stage]))


def snippet(text: str, match: re.Match, pad: int = PAD) -> str:
    """The match plus its surroundings, with elisions marked."""
    lo = max(0, match.start() - pad)
    hi = min(len(text), match.end() + pad)
    return ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def search_row(row: dict, stage: str, pattern: re.Pattern) -> list[dict]:
    """One hit record per matching field of a row; empty if the row does not match."""
    hits = []
    for field in fields(row, stage):
        matches = pattern.finditer(field["text"])
        first = next(matches, None)
        if first is None:
            continue
        # Counted off the iterator, keeping only the first match. A pattern that
        # can match the empty string — `.*?`, `(?:)`, an empty pattern — matches
        # once per character, and holding all of those match objects at once is
        # megabytes per field on a response that runs to 200k characters.
        count = 1 + sum(1 for _ in matches)
        hits.append(
            {
                "side": field["side"],
                "role": field["role"],
                "turn": field["turn"],
                "count": count,
                "chars": len(field["text"]),
                "snippet": snippet(field["text"], first),
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

    `unknown` is the same mistake one step further back: a row that matches on
    one completion while the server shortened the other cannot be called
    exclusive, because the text nobody read could hold the string too. A record
    says which of its columns were cut in `truncated`.
    """
    split = {"chosen_only": 0, "rejected_only": 0, "both": 0, "unknown": 0}
    for rec in records:
        sides = {h["side"] for h in rec["hits"]}
        cut = set(rec.get("truncated") or ())
        chosen, rejected = "chosen" in sides, "rejected" in sides
        if chosen and rejected:
            split["both"] += 1
        elif chosen:
            split["unknown" if "rejected" in cut else "chosen_only"] += 1
        elif rejected:
            split["unknown" if "chosen" in cut else "rejected_only"] += 1
    return split
