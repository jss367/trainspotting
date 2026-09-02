"""Benchmark test sets, and how to cut a probe out of one item.

Every other layer starts from a string or a question the caller already has.
This one starts from a benchmark: the question is whether the items a model is
scored on were also in what it was trained on, and if so at which stage and on
which side of the example. A test question in an SFT prompt column means the
model was trained to read it; its worked answer in a response column means the
model was fit to produce it. Those are different findings, and the same
prompt/response split `grep` makes is what tells them apart.

An item is not searched for whole. Copies of a benchmark item in training data
routinely differ from the original in whitespace, in a stripped annotation, or in
a prefix ("Question: ") — so an exact search for the full text misses them and
reports a clean stage. What is searched for is a **probe**: a window of
consecutive words from the middle of the item, matched with any whitespace
between the words, as whole words at either end, and regardless of case.
Thirteen words is the window GPT-3's
contamination analysis used (Brown et al. 2020, appendix C), and it is long
enough that a chance match is rare: thirteen consecutive words from a maths
word problem do not recur in unrelated text.

Two things a probe cannot do, stated here because a clean result is read as
absence:

  * A paraphrased or translated item does not match. A probe finds verbatim and
    near-verbatim copies, which is what "contamination" usually means, and
    nothing looser.
  * An item shorter than the minimum word count is not probed at all, and is
    counted as such rather than as clean. Short items are where a window would
    match by chance.

Each entry names the text field that holds the question and, where one exists,
the field that holds a worked answer long enough to probe. A multiple-choice
benchmark's answer is a letter, so it has no answer field — but its options are
text, stored apart from the stem, and they are probed as a third part,
`choices`. Many MMLU stems are under the eight-word floor and the options are
the only probe-able text; and a stem generic enough to recur ("Which of the
following is true?") is not a copy of the item where its options are not. The
options are joined by newlines with no letter prefixes, so a probe window that
spans two options misses a copy that carries "A." / "(B)" between them; a copy
of the whole item still hits on the stem, and the window is worth having for
the stems that cannot be probed at all.
"""

import random
import re

from . import hf

# Rows the datasets-server returns per page, and its maximum.
PAGE = 100

# Words in a probe, and the fewest an item may have to be probed at all.
WORDS = 13
MIN_WORDS = 8

BENCHMARKS = {
    "gsm8k": {
        "name": "GSM8K",
        "hf_dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question": "question",
        "choices": None,
        "answer": "answer",
        # The worked answers carry calculator annotations, `<<16-3-4=9>>`,
        # that almost every re-release strips. A probe that crossed one would
        # miss every stripped copy, which is most of them.
        "clean": {"answer": r"<<[^>]*>>"},
        "note": "Grade-school maths word problems; the 1,319-item test split.",
    },
    "math-500": {
        "name": "MATH-500",
        "hf_dataset": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "question": "problem",
        "choices": None,
        "answer": "solution",
        "note": "The 500-problem subset of Hendrycks MATH that PRM800K held out.",
    },
    "humaneval": {
        "name": "HumanEval",
        "hf_dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "split": "test",
        "question": "prompt",
        "choices": None,
        "answer": "canonical_solution",
        "note": (
            "164 Python function stubs with docstrings. The prompt is code, so a "
            "probe is a run of code tokens; whitespace between them is still free."
        ),
    },
    "mmlu": {
        "name": "MMLU",
        "hf_dataset": "cais/mmlu",
        "config": "all",
        "split": "test",
        "question": "question",
        "choices": "choices",
        "answer": None,
        "note": "14,042 multiple-choice questions over 57 subjects. The answer is a letter.",
    },
    "mmlu-pro": {
        "name": "MMLU-Pro",
        "hf_dataset": "TIGER-Lab/MMLU-Pro",
        "config": "default",
        "split": "test",
        "question": "question",
        "choices": "options",
        "answer": None,
        "note": "12,032 ten-option questions; the answer is a letter.",
    },
    "arc-challenge": {
        "name": "ARC-Challenge",
        "hf_dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "question": "question",
        # A struct column: `{"text": [...], "label": ["A", "B", ...]}`.
        "choices": "choices.text",
        "answer": None,
        "note": "1,172 grade-school science questions; the answer is a letter.",
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "hf_dataset": "truthfulqa/truthful_qa",
        "config": "generation",
        "split": "validation",
        "question": "question",
        "choices": None,
        "answer": "best_answer",
        "note": (
            "817 questions built to elicit common misconceptions. The best answers "
            "are one sentence, so most are under the probe minimum and only the "
            "questions are searched."
        ),
    },
    "ifeval": {
        "name": "IFEval",
        "hf_dataset": "google/IFEval",
        "config": "default",
        "split": "train",
        "question": "prompt",
        "choices": None,
        "answer": None,
        "note": (
            "541 instruction-following prompts. The whole benchmark is prompts, and "
            "its one split is named `train` although it is the evaluation set."
        ),
    },
}


def resolve(name: str) -> dict:
    key = name.lower()
    if key not in BENCHMARKS:
        raise KeyError(
            f"Unknown benchmark {name!r}. Known: {', '.join(sorted(BENCHMARKS))}."
        )
    return {"id": key, **BENCHMARKS[key]}


def parts(spec: dict) -> list[str]:
    """The item fields this benchmark can be probed on, question first."""
    return ["question"] + [p for p in ("choices", "answer") if spec.get(p)]


def column(spec: dict, part: str) -> str:
    """The dataset column a part is read from — the head of a dotted path.

    The server reports truncation per column, so a part inside a struct
    (`choices.text`) is cut when its column is.
    """
    return spec[part].split(".")[0]


def _cell(row: dict, path: str):
    value = row
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def item_text(spec: dict, row: dict, part: str) -> str | None:
    """The text of one part of an item, cleaned as the benchmark asks.

    A `choices` part is a list of option strings and comes back as one text:
    the options in order, joined by single newlines, with no letter prefixes.
    The benchmarks store none, and which prefix a copy carries — "A.", "(A)",
    "A)" — is the one thing copies do not agree on. The cost is stated in the
    module docstring: a window spanning two options misses a prefixed copy.
    """
    value = _cell(row, spec[part])
    if isinstance(value, list):
        if not value or not all(isinstance(v, str) for v in value):
            return None
        value = "\n".join(value)
    if not isinstance(value, str):
        return None
    strip = (spec.get("clean") or {}).get(part)
    if strip:
        value = re.sub(strip, "", value)
    return value


_WORD = re.compile(r"\S+")
# A word character as RE2 counts one for `\b`: ASCII only, unlike Python's `\w`.
_EDGE = re.compile(r"[0-9A-Za-z_]")


def probe(text: str, words: int = WORDS, min_words: int = MIN_WORDS) -> dict | None:
    """A window of `words` consecutive words from the middle of `text`, or None.

    Two forms of the same window come back. `literal` is the text as written,
    whitespace included, for an index that matches exact strings. `regex` is
    the words escaped and joined by `\\s+`, for a scan that should not care how
    a copy was re-wrapped, with a word boundary at each end that is a word
    character so the window is matched as whole words. Both are cut from the
    same words, so a hit on either is a hit on the same span of the item.

    The middle rather than the start, because the start is where copies differ:
    a "Question:" prefix, a stripped title, a renumbered problem. An item with
    fewer than `min_words` words is not probed — a short window matches by
    chance, and a chance match reads as contamination. For the same reason a
    `words` below `min_words` is refused outright rather than cut: the floor
    is on the window, and admitting the item only to cut a one-word probe
    from it would defeat it.
    """
    if words < min_words:
        raise ValueError(
            f"a probe needs at least {min_words} words, got {words}: a shorter "
            "window matches by chance, and a chance match reads as contamination"
        )
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if len(spans) < min_words:
        return None
    k = min(words, len(spans))
    start = (len(spans) - k) // 2
    window = spans[start : start + k]
    first, last = window[0][0], window[-1][1]
    regex = r"\s+".join(re.escape(text[a:b]) for a, b in window)
    # The window's ends are whole words, not prefixes or suffixes of one: a
    # window ending in "market" is not in "marketplace", nor one starting in
    # "train" in "restrain". RE2 has no lookaround, so the only edge assertion
    # is `\b`, and `\b` demands a word character on the inside — put next to
    # "$2", "market?" or a closing paren it would ask for one past the
    # punctuation and miss the exact copy. So each end is anchored only when
    # its own character is a word character, in RE2's ASCII sense: Python's
    # `\w` admits "é", RE2's `\b` does not, and anchoring on a letter RE2 does
    # not count as one would fail on the copy that matters.
    if _EDGE.match(text[first]):
        regex = r"\b" + regex
    if _EDGE.match(text[last - 1]):
        regex = regex + r"\b"
    return {
        "literal": text[first:last],
        "regex": regex,
        "words": k,
    }


def pick_indices(total: int, n: int, seed: int) -> list[int]:
    """Which items to probe: all of them if `n` covers the set, else a seeded draw.

    Sorted, so the rows are fetched in page order and a result file lists items
    in the order the benchmark does.
    """
    if n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def fetch_items(spec: dict, indices: list[int]) -> list[dict]:
    """The rows at `indices`, by page, each with its index and any cut cells.

    A cell the server shortened cannot be probed from its middle — the middle
    may be the part that was cut — so the truncation travels with the row and
    the caller skips that part rather than probing a fragment.
    """
    by_page: dict[int, list[int]] = {}
    for i in indices:
        by_page.setdefault(i // PAGE, []).append(i)
    out = []
    for page, wanted in sorted(by_page.items()):
        j = hf.rows_page(
            spec["hf_dataset"], page * PAGE, PAGE, spec["config"], spec["split"]
        )
        rows = {r["row_idx"]: r for r in j.get("rows", [])}
        for i in wanted:
            r = rows.get(i)
            if r is None:
                continue
            out.append({
                "index": i,
                "row": r["row"],
                "truncated": list(r.get("truncated_cells") or []),
            })
    return out


def total_items(spec: dict) -> int:
    return hf.num_rows(spec["hf_dataset"], spec["config"], spec["split"])
