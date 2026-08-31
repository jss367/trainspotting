"""Classify training prompts by what they primarily train: HHH values vs skills.

Each sampled prompt gets exactly one primary label. The taxonomy separates the
three value axes (helpful / honest / harmless) from skill content (capability,
precise instruction following, tool use), because "how much of training is about
being harmless" is only meaningful against that baseline.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

from . import extract

LABELS = [
    "harmlessness",
    "honesty",
    "helpfulness",
    "capability",
    "instruction_following",
    "tool_use",
    "other",
]

SYSTEM = """You label language-model training prompts by what the training example primarily teaches the model. Assign exactly one label per prompt:

- harmlessness: handling unsafe or harmful requests — refusing or safely answering dangerous, illegal, unethical, or jailbreak-style prompts; safety-sensitive advice.
- honesty: truthfulness and calibration — admitting uncertainty or inability, refusing to fabricate, correcting false premises in the question, resisting flattery or pressure to agree, questions about the model's own nature or limits.
- helpfulness: general assistance where the point is being useful to a person — open-ended chat, writing and editing, advice, planning, explanation, summarization, everyday Q&A.
- capability: skill content — math problems, competition puzzles, code writing or debugging, algorithms, science and logic exercises.
- instruction_following: the prompt's point is precise formal constraints (word counts, forced formats, forbidden words, JSON shape), regardless of topic.
- tool_use: calling functions/APIs or acting as an agent with tools.
- other: none of the above fits.

If a prompt is a harmful or trick request (even disguised as a normal task), label it harmlessness — the training signal is how to handle it. If a prompt rests on a false premise or asks the model to assert something it cannot know, label it honesty.

Reply with ONLY a JSON array: [{"i": <index>, "label": "<label>"}, ...] covering every index you were given."""


def _parse(text: str, n: int, valid: list[str] = LABELS) -> dict[int, str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for item in arr:
        try:
            i, label = int(item["i"]), str(item["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n and label in valid:
            out[i] = label
    return out


ASK_SYSTEM = """You judge language-model training prompts against a question the user wants answered about the training data. The question is:

{question}

For each numbered prompt, decide whether this training example matches the question — i.e., whether training on it plausibly teaches the model the thing the question asks about. Judge the training signal, not surface keywords: a math problem that merely mentions people does not match a question about caring for people; a harmful request the model must refuse does match a question about protecting people.

Reply with ONLY a JSON array: [{{"i": <index>, "label": "yes" or "no"}}, ...] covering every index you were given."""


ASK_DOC_SYSTEM = """You judge documents from a language model's PRETRAINING corpus against a question the user wants answered about the training data. The question is:

{question}

These are raw documents — web pages, scientific PDFs, source code, forum posts — not instructions to a model and not written for it. Nobody curated them to teach anything. So judge what a model would absorb from fitting this text: the claims it asserts, the norms it takes for granted, the behaviour it depicts approvingly or disapprovingly.

Two failure modes to avoid. Do not match on topic alone: a news report that a person died is about death, but it teaches nothing about valuing life. Do not require the document to be explicit: a story where a character is condemned for letting someone drown carries the norm without ever stating it.

Boilerplate, navigation chrome, link dumps, and near-empty pages are "no".

Reply with ONLY a JSON array: [{{"i": <index>, "label": "yes" or "no"}}, ...] covering every index you were given."""


def build_system(question: str | None = None, system: str | None = None) -> str:
    """The exact system prompt a run will use.

    Split out of `classify_prompts` so a caller can record which instrument
    produced its numbers. The taxonomy wording is not incidental to the result —
    rewriting a label's definition moves every share it reports — so a result
    file that cites a percentage should be able to cite the prompt behind it.
    """
    if system:
        return system.format(question=question) if question else system
    return ASK_SYSTEM.format(question=question) if question else SYSTEM


def system_id(system: str) -> str:
    """A short content hash of a system prompt, for stamping into results."""
    return hashlib.sha256(system.encode()).hexdigest()[:12]


def classify_prompts(
    prompts: list[str],
    model: str = "claude-opus-5",
    batch_size: int = 20,
    workers: int = 4,
    question: str | None = None,
    system: str | None = None,
    max_chars: int = extract.MAX_CLASSIFY_CHARS,
) -> tuple[list[str | None], dict[str, int]]:
    """Return one label (or None) per prompt in order, plus why any are missing.

    The second value counts what stayed unlabeled after the retry below, by
    reason: "refusal" (the classifier declined), "error" (the API refused to
    answer at all), "unparsed" (it answered without a usable label for that
    prompt). Callers are expected to record it. A silent None is the one failure
    mode this layer must not have: refusals land on jailbreak-style prompts,
    which is precisely the content a harmlessness share is measuring, so
    dropping them quietly biases the headline number downward and nothing on the
    page would show it.

    With `question` set, labels are "yes"/"no" judgments of that question
    instead of the fixed taxonomy. `system` overrides the prompt entirely, which
    is how pretraining documents get judged as documents rather than as requests.
    `max_chars` is how much of each input the model sees; corpus documents need
    far more of it than prompts do, so callers raise it and shrink the batch.
    """
    system = build_system(question, system)
    valid = ["yes", "no"] if question else LABELS
    client = anthropic.Anthropic()
    batches = [
        (start, prompts[start : start + batch_size])
        for start in range(0, len(prompts), batch_size)
    ]

    def judge(items: list[str]) -> tuple[dict[int, str], str | None]:
        """(labels by position within `items`, why any are missing)."""
        numbered = "\n\n".join(
            f"### {i}\n{p[:max_chars]}" for i, p in enumerate(items)
        )
        try:
            # Server-side refusal fallback: some prompts are raw jailbreak text,
            # so a decline on one batch falls back instead of losing the batch.
            resp = client.beta.messages.create(
                model=model,
                max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                system=system,
                messages=[{"role": "user", "content": numbered}],
            )
        except anthropic.APIStatusError:
            return {}, "error"
        if resp.stop_reason == "refusal":
            return {}, "refusal"
        text = "".join(b.text for b in resp.content if b.type == "text")
        parsed = _parse(text, len(items), valid)
        return parsed, None if len(parsed) == len(items) else "unparsed"

    def run(batch):
        start, items = batch
        parsed, reason = judge(items)
        drops: dict[str, int] = {}
        missing = [i for i in range(len(items)) if i not in parsed]
        if missing and len(items) > 1:
            # One prompt can sink the other nineteen: a refusal on a single
            # piece of raw jailbreak text, a 400 on one oversized row, or a
            # reply that skipped an index. Re-ask for the unlabeled ones one at
            # a time, so what is finally lost is the prompt that caused it
            # rather than everything batched beside it.
            for i in missing:
                one, one_reason = judge([items[i]])
                if 0 in one:
                    parsed[i] = one[0]
                else:
                    drops[one_reason] = drops.get(one_reason, 0) + 1
        elif missing:
            drops[reason] = drops.get(reason, 0) + 1
        return start, parsed, drops

    labels: list[str | None] = [None] * len(prompts)
    unlabeled: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start, parsed, drops in ex.map(run, batches):
            for i, label in parsed.items():
                labels[start + i] = label
            for reason, k in drops.items():
                unlabeled[reason] = unlabeled.get(reason, 0) + k
    return labels, unlabeled
