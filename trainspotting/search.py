"""Locate a phrase in the training examples the datasets-server matched.

The /search endpoint answers "how many rows mention these words" exactly, but
its index is word-based: it returns every row containing all the query's words,
not just the exact phrase, and it says nothing about *where* in the example the
words appear. Both halves matter for behavior attribution — "I am ChatGPT" in
an assistant turn teaches the opposite of the same words in a DPO rejected
response — so this module re-reads each matched row locally and reports which
trainable field held the phrase, with a snippet around every occurrence.
"""

from trainspotting import extract

SNIPPET_CONTEXT = 80   # characters kept either side of a match
MAX_SNIPPETS = 3       # per row; one is usually enough to judge it


def texts(row: dict, stage: str):
    """Yield (where, text) for each trainable text field of a row.

    `where` names the field's role in the training example, because that decides
    what a match teaches: sft responses and dpo chosen turns are fit toward,
    dpo rejected turns are pushed away from, rlvr rollouts are merely sampled
    and verifier fields are never seen by the model at all. Source-label and id
    columns are deliberately absent — a source named "chatgpt_synthetic" should
    show up in the source breakdown, not masquerade as example text.
    """
    if stage == "sft":
        for m in row.get("messages") or []:
            if isinstance(m, dict) and m.get("content"):
                role = m.get("role")
                yield ("response" if role == "assistant" else "prompt", str(m["content"]))
    elif stage == "dpo":
        # Instruct-DPO stores the shared prompt as the user turns of `chosen`
        # (duplicated in `rejected`); Think-DPO also carries it as row["prompt"].
        # Read it from one place so a prompt hit is one hit, not three.
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            yield ("prompt", prompt)
        for side in ("chosen", "rejected"):
            for m in row.get(side) or []:
                if not (isinstance(m, dict) and m.get("content")):
                    continue
                if m.get("role") == "assistant":
                    yield (side, str(m["content"]))
                elif side == "chosen" and not (isinstance(prompt, str) and prompt.strip()):
                    yield ("prompt", str(m["content"]))
    elif stage == "rlvr":
        prompt = extract.extract_prompt(row, "prompt")
        if prompt:
            yield ("prompt", prompt)
        for out in row.get("outputs") or []:
            if out:
                yield ("rollout", str(out))
        gt = row.get("ground_truth")
        for v in gt if isinstance(gt, list) else [gt]:
            if v:
                yield ("verifier", str(v))
        for key in ("solution", "constraint"):
            if row.get(key):
                yield ("verifier", str(row[key]))
    else:
        raise ValueError(f"Unknown stage {stage!r}")


def _occurrences(text: str, needle: str) -> list[int]:
    hay, needle = text.lower(), needle.lower()
    out, i = [], hay.find(needle)
    while i >= 0:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def _snippet(text: str, start: int, length: int) -> str:
    lo, hi = max(0, start - SNIPPET_CONTEXT), start + length + SNIPPET_CONTEXT
    return (
        ("…" if lo else "")
        + text[lo:start]
        + "«" + text[start : start + length] + "»"
        + text[start + length : hi]
        + ("…" if hi < len(text) else "")
    ).replace("\n", " ")


def find_matches(row: dict, stage: str, query: str) -> dict:
    """Where a matched row actually holds the query, with snippets.

    `exact` is whether the query appears verbatim (case-insensitive) somewhere
    in the row's trainable text. When it does not — the index matched the words
    scattered across the row — the snippets fall back to the query's longest
    word, so the record still shows what the index saw in it.
    """
    where: list[str] = []
    snippets: list[str] = []

    def scan(needle: str) -> bool:
        found = False
        for w, text in texts(row, stage):
            for i in _occurrences(text, needle):
                found = True
                if w not in where:
                    where.append(w)
                if len(snippets) < MAX_SNIPPETS:
                    snippets.append(_snippet(text, i, len(needle)))
        return found

    exact = scan(query)
    if not exact:
        terms = sorted(query.split(), key=len, reverse=True)
        if terms:
            scan(terms[0])
    return {"exact": exact, "where": where, "snippets": snippets}
