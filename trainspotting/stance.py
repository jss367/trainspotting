"""Which way a training example pushes on a question, not whether it is about it.

`ask` judges prompts. For a values question that is the wrong half of the
example and the wrong shape of answer.

The wrong half, because a value lives in what the model is fit to produce. "This
prompt is about human lives" is a claim about the topic of the request; whether
training on the example teaches the model to value human lives is a claim about
the response, and for a preference pair it is a claim about *which* response.
`search` already reads that half — it scans response columns and reports the DPO
side of every hit — but it matches strings, so it can only find a value someone
can write a regex for.

The wrong shape, because yes/no cannot represent training that pushes the other
way, and this data contains some. The README's worked example is an RLVR row
whose verifier pays the model 54% of the time for delivering an anti-vaccine
speech in the right format. Under `ask` that row is a match: it is certainly
"about" human welfare. Counted as evidence that the model was taught to care
about people, it points backwards.

So this layer reads the whole stored example and answers with a direction:

    toward   fitting this example teaches the model the thing asked about
    away     fitting it teaches the opposite
    neither  the example does not bear on the question either way

and the headline is the net, `toward − away`.

## What it reads

The committed context records, not a fresh sample. `context` already stores the
whole example behind every sampled prompt, keyed to the rows `classify` and
`ask` labeled, so a stance run costs API calls and no downloads, and lands on
exactly the examples someone can click through to on the site.

It costs more per example than `ask` does: a prompt is 1,500 characters and a
DPO pair is two completions plus the conversation in front of them. Batches are
small for that reason.

## Per kind

| Kind | What the judgment reads | What `away` means there |
|---|---|---|
| sft | the prompt and the assistant turns the model is fit to | the target response itself cuts against the question |
| dpo | the shared prefix, then both completions, marked preferred / dispreferred | the *dispreferred* completion is the one that serves the question — training pushes away from it |
| rlvr | the prompt, the verifier and what it checks, and the pass rate | the reward pays for output that cuts against the question |
| chat | the conversation | nothing was fit to it; `chat` targets are refused rather than judged |

A chat log is not a training example and has no direction, which is the same
reason `context` marks no turn in it as a target. Asking this question of one
would invent a training signal the data does not have, so a `chat` target is an
error rather than a run.
"""

from . import context, extract, rewards

STANCES = ["toward", "away", "neither"]

# How much of one rendered example the judge reads. The context records are
# already cut at 4,000 characters a field, so this bounds how many of those
# fields stack up in a single input — and a DPO pair stacks three of them.
MAX_EXAMPLE = 12000
# Whole examples, not prompts: a batch of four is already an order of magnitude
# more input than `ask` sends in a batch of twenty.
BATCH = 4


SYSTEM = """You judge which way a language-model training example pushes on a question. The question is:

{question}

You are shown one whole training example at a time, marked up by what role each part plays in training. Judge the training signal — what fitting this example teaches the model — not the topic of the text. Answer with exactly one of:

- toward: fitting this example makes the model more likely to behave the way the question describes.
- away: fitting this example makes the model LESS likely to behave that way — it rewards, prefers, or demonstrates the opposite.
- neither: the example does not bear on the question in either direction.

How to read each markup:

- [PROMPT] is context the model reads, not text it is trained to produce. What the prompt asks for is not by itself the training signal; a harmful request answered with a refusal pushes toward safety, not away from it.
- [TARGET] is text the model is trained to produce. This is where the signal usually is.
- [PREFERRED] and [DISPREFERRED] are the two sides of a preference pair. Training pushes the model toward the preferred one and away from the other. If the DISPREFERRED completion is the one that serves the question, that is "away" — the model is being trained out of it. If both sides do the same thing, the pair teaches nothing about it: "neither".
- [REWARD] describes a verifier that scores generated output. The example teaches whatever the verifier pays for. A verifier that only checks formatting pays for the content the prompt requested, whatever that content is — so a formatting reward on a request for harmful content pushes "away" from protecting people, not "toward" it.

"neither" is the right answer most of the time, and much more often than "toward". Do not stretch: a math problem set in a hospital is "neither", and so is a factual passage that merely mentions death or illness.

Reply with ONLY a JSON array: [{{"i": <index>, "label": "toward" | "away" | "neither"}}, ...] covering every index you were given."""


def _turn_text(turn: dict) -> str:
    """One stored turn as text, reasoning span included.

    The reasoning is folded away in the context *view* because it is most of the
    length and buries the answer. It is not folded away here: the model was fit
    to it, and a value expressed only while thinking was still trained in.
    """
    reasoning = (turn.get("reasoning") or {}).get("text")
    body = turn.get("text", "")
    return f"<think>{reasoning}</think>\n{body}" if reasoning else body


# Below this, `extract.excerpt` has no room to take three spans and their
# elision markers, and would produce a span of nonsense or a negative one.
MIN_EXCERPT = 300


def _cut(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    if cap < MIN_EXCERPT:
        return text[: max(0, cap)]
    return extract.excerpt(text, cap)


def _allocate(sizes: list[int], room: int) -> list[int]:
    """Split `room` characters between parts of the given sizes.

    Equal shares, with whatever a short part does not need handed back to the
    others. A preference pair is usually one long completion and one short one,
    and a proportional split would cut the long one twice as hard for no reason
    while the short one leaves its budget unused.
    """
    caps = [0] * len(sizes)
    pending = list(range(len(sizes)))
    while pending:
        share = room // len(pending)
        fits = [i for i in pending if sizes[i] <= share]
        if not fits:
            for i in pending:
                caps[i] = share
            return caps
        for i in fits:
            caps[i] = sizes[i]
            room -= sizes[i]
            pending.remove(i)
    return caps


def _fit(parts: list[tuple[str, str]], budget: int) -> str:
    """Render `(marker, body)` pairs inside a character budget.

    Each body is cut to its own share rather than the joined text being cut once
    at the end. Cutting the joined text is what a single `excerpt` over the whole
    example does, and on a long preference pair it slices straight through
    "[DISPREFERRED — training pushes away from this]" — which is the one line
    that decides the answer. Sixteen of 300 sampled Dolci-Think-DPO pairs lost a
    side marker that way.
    """
    parts = [(marker, body) for marker, body in parts if body.strip()]
    if not parts:
        return ""
    overhead = sum(len(marker) + 3 for marker, _ in parts)
    caps = _allocate([len(b) for _, b in parts], max(0, budget - overhead))
    return "\n\n".join(f"{marker}\n{_cut(body, cap)}" for (marker, body), cap in zip(parts, caps))


def _turn_parts(turns: list[dict], target_role: str = "assistant") -> list[tuple[str, str]]:
    return [
        (
            f"[{'TARGET' if t.get('role') == target_role else 'PROMPT'}: {t.get('role', '?')}]",
            _turn_text(t),
        )
        for t in turns or []
    ]


def _side_parts(rec: dict) -> list[tuple[str, str]]:
    """A preference pair: the history both sides share, then each completion.

    `context` stores each side as the whole conversation, so the shared history
    is on both. Rendering both copies would spend the budget saying the same
    thing twice and bury the difference the pair is about.

    The split is at the branch point, not by role. A multi-turn pair shares
    assistant turns too, and marking those PREFERRED / DISPREFERRED tells the
    judge the pair was preferred for text that is identical on both sides — the
    one thing that would make it read a direction into a pair that has none. 12
    of the 300 sampled Dolci-Instruct-DPO pairs are multi-turn.
    """
    chosen = (rec.get("chosen") or {}).get("turns", [])
    rejected = (rec.get("rejected") or {}).get("turns", [])
    shared = context.branch_point(chosen, rejected)
    prefix = [
        (f"[PROMPT: {t.get('role', '?')}]", _turn_text(t))
        for t in chosen[:shared]
    ]
    sides = [
        (f"[{label}]", "\n\n".join(
            _turn_text(t) for t in turns[shared:] if t.get("role") == "assistant"
        ))
        for turns, label in (
            (chosen, "PREFERRED — training pushes toward this"),
            (rejected, "DISPREFERRED — training pushes away from this"),
        )
    ]
    return prefix + sides


def _reward_parts(rec: dict) -> list[tuple[str, str]]:
    """The verifier as the judge needs to see it: what it checks, and how often
    the reference model satisfied it.

    The pass rate is part of the signal, not decoration. A constraint checker
    that reference rollouts satisfied half the time was paying out for that
    content half the time.
    """
    r = rec.get("reward") or {}
    kind = r.get("kind") or "unknown"
    explain = r.get("explain") or rewards.KINDS.get(kind, {}).get("explain", "")
    parts = [("[REWARD]", f"{kind}: {explain}")]
    for field in ("ground_truth", "solution", "constraint"):
        value = r.get(field)
        if value and value.get("text"):
            parts.append((f"[REWARD: {field}]", value["text"]))
    rollouts = rec.get("rollouts") or {}
    if rollouts.get("passrate") is not None:
        parts.append(
            (
                "[REWARD: reference rollouts]",
                f"{rollouts.get('correct')} of {rollouts.get('total')} passed"
                f" ({rollouts['passrate']:.0%})",
            )
        )
    if (rollouts.get("sample") or {}).get("text"):
        parts.append(
            ("[REWARD: one generation the verifier scored]", rollouts["sample"]["text"])
        )
    return parts


def render(rec: dict) -> str:
    """A stored context record as the marked-up text the judge reads."""
    kind = rec.get("kind")
    if kind == "sft":
        parts = _turn_parts(rec.get("turns", []))
    elif kind == "dpo":
        parts = _side_parts(rec)
    elif kind == "rlvr":
        prompt = (rec.get("prompt_full") or {}).get("text", "")
        parts = [("[PROMPT]", prompt)] + _reward_parts(rec)
    else:
        # A log. `cmd_stance` refuses these upstream; this is the shape-check
        # that keeps a new kind from silently rendering as an empty string.
        raise ValueError(f"no stance rubric for kind {kind!r}")
    return _fit(parts, MAX_EXAMPLE)


def net(counts: dict[str, int]) -> int:
    """toward − away. The headline: how much of the stage pushes which way."""
    return counts.get("toward", 0) - counts.get("away", 0)
