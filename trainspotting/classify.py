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


def verifier_label(row: dict, kind: str) -> str | None:
    """The label this row's verifier fixes, or None to ask the classifier.

    `kind` is `registry.stage_kind` of the stage the row came from. Only an RL
    example has a verifier; every other kind is read off the prompt.
    """
    if kind != "rlvr":
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


# A chat log is not training data. The rubric above labels a prompt by what
# fitting the example would teach — which is why it sends a jailbreak attempt to
# `harmlessness`, "the training signal is how to handle it". Nothing was fit to a
# WildChat conversation, so there is no such signal to read, and only the opening
# user turn is passed anyway: whether the assistant refused or complied is not
# even on screen. Applied there, the model rubric reports a training property the
# data does not have.
#
# So label what the person asked for instead. Same seven labels, so a dataset's
# card and a model stage's card still stack up against each other — what changes
# is the claim each bar makes.
CHAT_SYSTEM = """You label prompts that real people sent to a chatbot, by what the person was asking for. Nothing was trained on these conversations; you are describing the request, not a training signal. You see only the opening user turn, so judge that — not how the assistant did or should have answered. Assign exactly one label per prompt:

- harmlessness: the person is asking for something unsafe, harmful, illegal, or unethical, or is trying to jailbreak the assistant. This means the request was safety-relevant; it says nothing about what came back.
- honesty: the request presses on what is true or knowable — asking for facts the assistant cannot know, resting on a false premise, pushing it to agree, or asking about the assistant's own nature or limits.
- helpfulness: ordinary assistance — open-ended chat, writing and editing, advice, planning, explanation, summarization, everyday questions.
- capability: skill work — math, code writing or debugging, algorithms, science and logic problems.
- instruction_following: the point of the request is a precise formal constraint (word count, forced format, forbidden words, JSON shape), whatever the topic.
- tool_use: asking the assistant to call functions or APIs, or to act as an agent with tools.
- other: none of the above fits.

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


ASK_CHAT_SYSTEM = """You judge prompts that real people sent to a chatbot against a question the user wants answered about the data. The question is:

{question}

Nothing was trained on these conversations, so there is no training signal to judge — judge the request itself: what the person wanted, what they were willing to ask for, what they took for granted. You see only the opening user turn, so the assistant's answer is not evidence either way.

Do not match on topic alone: mentioning a subject is not the same as the request being about it.

Reply with ONLY a JSON array: [{{"i": <index>, "label": "yes" or "no"}}, ...] covering every index you were given."""

# Which rubric a stage's prompts are judged under. Only `chat` needs one of its
# own; every model stage is a training example and reads correctly under the
# default. Returning None means "let classify_prompts pick", which keeps the
# default path in exactly one place.
SYSTEMS = {"chat": (CHAT_SYSTEM, ASK_CHAT_SYSTEM)}


def system_for(kind: str, question: str | None = None) -> str | None:
    """The system prompt for this kind of example, or None for the default."""
    pair = SYSTEMS.get(kind)
    if not pair:
        return None
    return pair[1] if question else pair[0]


# HTTP statuses a single prompt can plausibly cause: the request was malformed
# or too large for this batch. Everything else — auth, quota, rate limits,
# overload, server errors — is a property of the run, and re-asking one prompt
# at a time would only multiply it.
PER_PROMPT_STATUSES = {400, 413, 422}


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
    valid: list[str] | None = None,
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

    `valid` overrides the answer set. Anything outside it is dropped rather than
    recorded, so a rubric with three answers has to say so here — otherwise the
    parser silently discards every one of them and the run reports a sample it
    never labeled.
    """
    if not prompts:
        return [], {}  # nothing to ask about; don't demand a key to say so
    system = build_system(question, system)
    valid = valid or (["yes", "no"] if question else LABELS)
    client = anthropic.Anthropic()
    batches = [
        (start, prompts[start : start + batch_size])
        for start in range(0, len(prompts), batch_size)
    ]

    def judge(items: list[str]) -> tuple[dict[int, str], str | None, bool]:
        """(labels by position within `items`, why any are missing, retry singly?).

        The third value says whether one prompt in the batch could plausibly be
        the cause. A refusal or a skipped index could be; a 401, an exhausted
        quota, or a persistent 429 could not — those are about the run.
        """
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
        except anthropic.APIStatusError as exc:
            return {}, "error", getattr(exc, "status_code", None) in PER_PROMPT_STATUSES
        if resp.stop_reason == "refusal":
            return {}, "refusal", True
        text = "".join(b.text for b in resp.content if b.type == "text")
        parsed = _parse(text, len(items), valid)
        return parsed, None if len(parsed) == len(items) else "unparsed", True

    def run(batch):
        start, items = batch
        parsed, reason, per_prompt = judge(items)
        drops: dict[str, int] = {}
        missing = [i for i in range(len(items)) if i not in parsed]
        if missing and len(items) > 1 and per_prompt:
            # One prompt can sink the other nineteen: a refusal on a single
            # piece of raw jailbreak text, a 400 on one oversized row, or a
            # reply that skipped an index. Re-ask for the unlabeled ones one at
            # a time, so what is finally lost is the prompt that caused it
            # rather than everything batched beside it.
            for done, i in enumerate(missing):
                one, one_reason, one_per_prompt = judge([items[i]])
                if 0 in one:
                    parsed[i] = one[0]
                    continue
                drops[one_reason] = drops.get(one_reason, 0) + 1
                if not one_per_prompt:
                    # The run started failing partway through the retry — a 401,
                    # a quota, a rate limit. Asking about the remaining prompts
                    # individually would get the same answer, slower, while
                    # deepening the limit that caused it. Count them and stop.
                    rest = len(missing) - done - 1
                    if rest:
                        drops[one_reason] += rest
                    break
        elif missing:
            # Either a one-item batch, or a failure no single prompt could have
            # caused. Fanning out the second kind turns one doomed request into
            # twenty, across every worker, which delays the failure and deepens
            # the rate limit that caused it without recovering a single label.
            drops[reason] = drops.get(reason, 0) + len(missing)
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
