"""Pull a plain-text user prompt out of heterogeneous Dolci row schemas."""

MAX_CLASSIFY_CHARS = 1500   # what the classifier sees — enough to judge intent
MAX_STORE_CHARS = 12000     # what results/ keeps for drill-down display

# Corpus documents get a far bigger budget than prompts. A prompt states its
# intent in its opening line; a web page or a paper does not, and the
# long-context mixes hold documents past 200k characters, so judging the first
# 1,500 would be judging the nav bar and the abstract.
#
# Deliberately equal to MAX_STORE_CHARS: the classifier must judge exactly the
# text the site displays, or "read the matched documents and check they mean
# what you think" stops being true.
MAX_DOCUMENT_CHARS = MAX_STORE_CHARS
EXCERPT_MARKER = "\n\n[…]\n\n"


def clip(text: str) -> str:
    text = str(text)
    if len(text) <= MAX_STORE_CHARS:
        return text
    return text[:MAX_STORE_CHARS] + "\n…[truncated]"


def excerpt(text: str, budget: int = MAX_DOCUMENT_CHARS, parts: int = 3) -> str:
    """A document reduced to `budget` characters, sampled across its whole length.

    Truncation would judge every long document by its opening boilerplate, which
    for a corpus document is the least representative part of it. Taking evenly
    spaced spans instead means the middle and end get a vote, and the elisions
    are marked so the classifier can see it is reading an excerpt.
    """
    text = str(text)
    if len(text) <= budget:
        return text
    # The markers count against the budget. Leaving them out overflowed it by 14
    # characters, which the classifier then truncated away — so the site showed
    # 14 characters more than was judged, breaking the one property this budget
    # exists to guarantee.
    span = (budget - len(EXCERPT_MARKER) * (parts - 1)) // parts
    step = (len(text) - span) // (parts - 1)
    return EXCERPT_MARKER.join(text[i * step : i * step + span] for i in range(parts))


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
