"""Helpers every command shares: stamps, sampling, result files, slugs, formatting."""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone

from .. import extract, hf, registry
from ..redact import redact_credentials


def _fmt_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    return f"{n:,}"


# Sentinel for "look the revision up now". Distinct from None, which is a
# caller saying it knows the revision and the answer is "unknown" — reusing an
# old result file that predates this field, say. Collapsing the two would stamp
# today's `main` onto rows drawn from a revision nobody recorded.
_RESOLVE = object()


def _stamp(dataset: str | None = None, revision=_RESOLVE) -> dict:
    """Provenance every result file carries: when it was written, and which
    commit of the dataset it was computed over.

    A dataset id alone does not identify what was counted. `main` moves — Ai2
    has republished these mixes — so a file reporting "8.1% harmlessness, n=300"
    without a revision cannot be checked later, or told apart from the same
    figure over different rows.

    Callers that draw rows resolve the revision *before* the draw and pass it
    here, so the stamp names the tree the rows actually came from; a lookup
    after a long labeling run could name a revision published while it ran. A
    lookup that fails records null rather than failing a run whose API calls are
    already paid for.
    """
    out = {"generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    if dataset:
        out["revision"] = hf.dataset_revision(dataset) if revision is _RESOLVE else revision
    return out


def _unlabeled_note(labels: list, reasons: dict[str, int]) -> str:
    """One line naming what the classifier never labeled, for stderr."""
    n = sum(1 for label in labels if label is None)
    if not n:
        return ""
    detail = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
    return f"  [{n} unlabeled ({detail or 'reason unrecorded'})]"


def _select_stages(args, stages_of, family):
    """The stages a command runs over: every one of `family`, narrowed by `--stage`.

    `family` names the group in the error message, so asking for a pretraining
    stage by a post-training name fails with the right suggestion.

    A target with none of the family at all is an error too, not an empty loop:
    a dataset has no pretraining corpora behind it, and `pretrain wildchat-1m`
    exiting silently having written nothing reads exactly like a sample that
    came back empty.
    """
    target = registry.resolve(args.target)
    stages = stages_of(target)
    if not stages:
        sys.exit(f"{args.target} has no {family} stages")
    if getattr(args, "stage", None):
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no {family} stage {args.stage!r} for {args.target}")
    return stages


def _sample_rows(stage, sample, seed):
    """(index, row, prompt) for each row of a deterministic (sample, seed) draw
    that has one.

    Rows carrying no user prompt drop out here, so the result is usually shorter
    than `sample`. The row travels with its prompt because part of what an
    example teaches is in the row rather than the text — an RL row's verifier
    settles its taxonomy label outright.

    The index is the row's absolute position in the split, and it is what every
    result record stores to address its training example. Joining on the prompt
    instead cannot tell two rows apart that open with the same 400 characters,
    which is rare in a curated mix and routine in a chat log: WildChat repeats
    the same Midjourney prompt-generator preamble before different conversations.
    """
    print(f"sampling {sample} rows from {stage['hf_dataset']} ...", file=sys.stderr)
    rows = hf.sample_rows_with_index(stage["hf_dataset"], sample, seed=seed)
    triples = (
        (i, r, extract.extract_prompt(r, stage["prompt_path"])) for i, r in rows
    )
    return [(i, r, p) for i, r, p in triples if p]


def _sample_prompts(stage, sample, seed):
    """(index, prompt), for the callers with no use for the row itself."""
    return [(i, p) for i, _, p in _sample_rows(stage, sample, seed)]


def _write_json(path, payload):
    """Write a result file, creating results/ if this is the first one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact_credentials(json.dumps(payload, indent=2)))
    return path


def _counts(records):
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return counts


def _print_match_rate(stage, k, n, lo, hi, path, note=""):
    print(
        f"{stage}: {k}/{n} match = {k / n * 100 if n else 0:.1f}%"
        f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%) -> {path}{note}",
        file=sys.stderr,
    )


# Characters of a derived name kept before the disambiguating hash. Well under
# the 255-byte basename limit once the model, stage and suffix are added.
MAX_SLUG_CHARS = 60


def _slug(text: str) -> str:
    """A filename-safe short name derived from a question or a search pattern.

    Two reductions lose enough to name a different run: text with no ASCII
    letters or digits — a Chinese phrase, a punctuation-only expression —
    reduces to nothing at all, and two long inputs can agree on their first 60
    characters. Either would write over an unrelated result file without saying
    so, so both get a hash of the original appended.

    Questions differing only in case or punctuation still share a file, which
    for prose is two spellings of one question. A regex is not prose — see
    `_pattern_slug`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    if not slug:
        return f"pattern-{digest}"
    if len(slug) > MAX_SLUG_CHARS:
        return f"{slug[:MAX_SLUG_CHARS].rstrip('-')}-{digest}"
    return slug


def _pattern_slug(
    pattern: str, case_sensitive: bool = False, regex: bool = False
) -> str:
    """A stable name for a pattern and the modes that change what it matches.

    Punctuation in a regex is syntax, not spelling: `a.b` and `a+b` match
    different text, and so does `a b` with one space or two. Case and literal
    versus regex mode matter the same way: the same text under either pair of
    modes is a different search. A literal keeps the plain slug only when it
    already *is* that slug and the search was case-insensitive; anything else
    carries a hash of the pattern and its mode. Pass `--slug` for a readable
    name.
    """
    base = re.sub(r"[^a-z0-9]+", "-", pattern.lower()).strip("-")
    # The readable shortcut still has to produce a filename: the empty pattern
    # is not a name, and a 300-character literal is its own slug but not a
    # basename any filesystem will take — which would spend the whole sampling
    # run and then fail on the write.
    if (
        base
        and pattern == base
        and len(base) <= MAX_SLUG_CHARS
        and not case_sensitive
        and not regex
    ):
        return base
    mode = f"{'cs' if case_sensitive else 'ci'}{'-regex' if regex else ''}"
    digest = hashlib.sha1(
        f"{pattern}\n{mode}".encode()
    ).hexdigest()[:8]
    if not base:
        return f"pattern-{digest}"
    return f"{base[:MAX_SLUG_CHARS].rstrip('-')}-{digest}"


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _filename_part(part: str) -> str:
    """One user-supplied component of a result filename, made separator-free.

    Not a security boundary — it is the user's own results/ directory — but a
    stray "/" in --slug or --index would scatter files outside it, away from
    where the user and the site look for results. Squash anything that is not
    a plain filename character.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("._-") or "x"


def _fmt_est(n: float | None) -> str:
    """An estimated token count, at the resolution the estimate supports.

    Three significant figures at most: these come from a sampled rate times a
    mean length, and printing 41,283,915 would claim precision the sample has
    nowhere near.
    """
    if n is None:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            v = n / scale
            text = f"{v:.0f}" if v >= 100 else f"{v:.1f}".removesuffix(".0")
            return text + suffix
    return f"{n:.0f}"
