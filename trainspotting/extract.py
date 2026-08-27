"""Pull a plain-text user prompt out of heterogeneous Dolci row schemas."""

MAX_CHARS = 1500  # enough to classify intent; keeps classifier calls cheap


def _first_user(messages) -> str | None:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


def extract_prompt(row: dict, prompt_path: str, max_chars: int = MAX_CHARS) -> str | None:
    if prompt_path == "messages":
        text = _first_user(row.get("messages"))
    elif prompt_path == "chosen_messages":
        text = _first_user(row.get("chosen"))
    elif prompt_path == "prompt":
        p = row.get("prompt")
        if isinstance(p, list):  # some RL mixes store prompt as chat messages
            text = _first_user(p)
        else:
            text = p
        if not text:
            text = _first_user(row.get("source_prompt"))
    else:
        raise ValueError(f"Unknown prompt_path {prompt_path!r}")
    if not text:
        return None
    text = str(text).strip()
    return text[:max_chars] if text else None
