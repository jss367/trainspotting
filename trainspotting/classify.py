"""Classify training prompts by what they primarily train: HHH values vs skills.

Each sampled prompt gets exactly one primary label. The taxonomy separates the
three value axes (helpful / honest / harmless) from skill content (capability,
precise instruction following, tool use), because "how much of training is about
being harmless" is only meaningful against that baseline.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

from . import extract, rewards

LABELS = [
    "harmlessness",
    "honesty",
    "helpfulness",
    "capability",
    "instruction_following",
    "tool_use",
    "other",
]

# Labels a row's own verifier settles, so no model is asked about the prompt.
# An RLVR example teaches whatever its reward pays for. When the reward is a
# program checking IFEval-style constraints, the example trains instruction
# following whatever the prompt is about — including a jailbreak the verifier is
# perfectly happy to see answered, which reading the prompt alone scores as
# harmlessness content and counts toward the opposite of what training does.
VERIFIER_LABELS = {"constraint checker": "instruction_following"}
# A kind that no longer exists would disable the rule silently, and the
# labels would quietly go back to being read off the prompt.
assert set(VERIFIER_LABELS) <= set(rewards.KINDS), sorted(set(VERIFIER_LABELS) - set(rewards.KINDS))


def verifier_label(row: dict, stage: str) -> str | None:
    """The label this row's verifier fixes, or None to ask the classifier."""
    if stage != "rlvr":
        return None
    return VERIFIER_LABELS.get(rewards.kind_for(row))


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


def classify_prompts(
    prompts: list[str],
    model: str = "claude-opus-5",
    batch_size: int = 20,
    workers: int = 4,
    question: str | None = None,
    system: str | None = None,
    max_chars: int = extract.MAX_CLASSIFY_CHARS,
) -> list[str | None]:
    """Return one label (or None) per prompt, preserving order.

    With `question` set, labels are "yes"/"no" judgments of that question
    instead of the fixed taxonomy. `system` overrides the prompt entirely, which
    is how pretraining documents get judged as documents rather than as requests.
    `max_chars` is how much of each input the model sees; corpus documents need
    far more of it than prompts do, so callers raise it and shrink the batch.
    """
    if not prompts:
        return []  # nothing to ask about; don't demand a key to say so
    if system:
        system = system.format(question=question) if question else system
    else:
        system = ASK_SYSTEM.format(question=question) if question else SYSTEM
    valid = ["yes", "no"] if question else LABELS
    client = anthropic.Anthropic()
    batches = [
        (start, prompts[start : start + batch_size])
        for start in range(0, len(prompts), batch_size)
    ]

    def run(batch):
        start, items = batch
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
            return start, {}
        if resp.stop_reason == "refusal":
            return start, {}
        text = "".join(b.text for b in resp.content if b.type == "text")
        return start, _parse(text, len(items), valid)

    labels: list[str | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start, parsed in ex.map(run, batches):
            for i, label in parsed.items():
                labels[start + i] = label
    return labels
