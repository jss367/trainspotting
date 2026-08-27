"""Pull prompts and responses out of heterogeneous Dolci row schemas."""

MAX_CLASSIFY_CHARS = 1500   # what the classifier sees — enough to judge intent
MAX_STORE_CHARS = 12000     # what results/ keeps for drill-down display


def clip(text: str) -> str:
    text = str(text)
    if len(text) <= MAX_STORE_CHARS:
        return text
    return text[:MAX_STORE_CHARS] + "\n…[truncated]"


def _first(messages, role: str) -> str | None:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == role and m.get("content"):
            return str(m["content"])
    return None


def extract_prompt(row: dict, prompt_path: str) -> str | None:
    """Full user-prompt text (untruncated)."""
    if prompt_path == "messages":
        text = _first(row.get("messages"), "user")
    elif prompt_path == "chosen_messages":
        text = _first(row.get("chosen"), "user")
    elif prompt_path == "prompt":
        p = row.get("prompt")
        if isinstance(p, list):  # some RL mixes store prompt as chat messages
            text = _first(p, "user")
        else:
            text = p
        if not text:
            text = _first(row.get("source_prompt"), "user")
    else:
        raise ValueError(f"Unknown prompt_path {prompt_path!r}")
    if not text:
        return None
    text = str(text).strip()
    return text or None


def extract_responses(row: dict, prompt_path: str) -> dict:
    """The completions the example trains on, where the row has them.

    SFT rows -> {"response": ...} (the target the model is trained to imitate).
    DPO rows -> {"chosen": ..., "rejected": ...} (the preference pair).
    RL rows have no completion — the signal is the reward — so this is empty.
    """
    out = {}
    if prompt_path == "messages":
        r = _first(row.get("messages"), "assistant")
        if r:
            out["response"] = r
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if isinstance(chosen, list):
        chosen = _first(chosen, "assistant")
    if isinstance(rejected, list):
        rejected = _first(rejected, "assistant")
    if chosen:
        out["chosen"] = str(chosen)
    if rejected:
        out["rejected"] = str(rejected)
    return out
