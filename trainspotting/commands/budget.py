"""`budget`: every stage's rate times its size, on one token scale."""

import sys

from .. import budget
from ..paths import RESULTS
from .common import _fmt_est, _write_json


# Why the rate column is not one rule. The correction that is right for one
# sampling design is a double count under another, and which applies is a
# property of how a stage was *drawn* rather than of what kind of stage it is —
# a corpus the datasets-server indexes in full is paged uniformly over documents
# and takes the same length weighting a post-training mix takes. Keyed by the
# `weighting` string `budget` records, longest prefix first, so the table
# explains the rules it actually used and no others.
_WEIGHTING_RULES = [
    (
        "fit characters — rows",
        "A corpus paged uniformly over documents is weighed by fit characters:\n"
        "without that its rate is a share of documents rather than of training.",
    ),
    (
        "fit characters",
        'Post-training rows are drawn uniformly, so their rate is weighed by fit\n'
        'characters — otherwise it answers "what fraction of examples" rather than\n'
        '"what fraction of training".',
    ),
    (
        "none",
        "Corpus documents drawn by shard come from shards drawn with probability\n"
        "proportional to size, which already weights by tokens, so their document rate\n"
        "is used unchanged; weighing it by length would apply that a second time.",
    ),
    (
        "document count",
        "One corpus stage stores no document lengths to weigh by, so its unweighed\n"
        "document rate reads its matches as if every document were the same size.",
    ),
]


_WEIGHTING_FOOTNOTE = """
Fit tokens are what the model was trained to produce, at {cpt:g} characters per token.
Rate: {rules}
This weighs tokens, not learning: a post-training token and a pretraining token
are not equally formative, and nothing here corrects for that."""


def _weighting_footnote(est: dict) -> str:
    """The rate-column footnote, naming only the weightings this estimate used."""
    used = {s["weighting"] for s in est["stages"] if s.get("weighting")}
    rules = []
    for prefix, text in _WEIGHTING_RULES:
        if any(w.startswith(prefix) for w in used) and text not in rules:
            rules.append(text)
            used = {w for w in used if not w.startswith(prefix)}
    return _WEIGHTING_FOOTNOTE.format(
        cpt=budget.CHARS_PER_TOKEN, rules=" ".join(rules) if rules else "n/a"
    )


def _warn_mixed_questions(est: dict) -> bool:
    """Say so, on stdout, when a slug covers more than one wording — and report
    whether it does, so callers can withhold the total.

    A slug is not a question: `--slug` takes any string and a generated one is
    truncated to 60 characters, so stages sharing a slug can have been scored
    against different words. Summing them produces a number no single question
    ever measured. The warning goes to stdout with the table rather than to
    stderr beside it, because the table is what gets piped into a document and
    the caveat has to travel with it.
    """
    if not est.get("mixed"):
        return False
    variants = est.get("question_variants") or []
    judges = est.get("classifiers") or []
    conflict = est.get("rubric_conflict") or []
    # Three ways to be mixed, and they read very differently to someone deciding
    # whether the withheld total was worth withholding — so say which it was.
    what = []
    if len(variants) > 1:
        what.append(f"{len(variants)} different wordings of the question")
    if len(judges) > 1:
        what.append(f"{len(judges)} different classifiers ({', '.join(judges)})")
    if conflict:
        what.append(f"a rubric that changed between {', '.join(conflict)} stages")
    if not what:
        what.append("different judging instruments")
    print(
        f"WARNING: the stages under slug {est['slug']!r} were scored with"
        f" {' and '.join(what)}, so they do not add up to one measurement."
        " No total is shown.\n"
    )
    for q in variants:
        print(f"  - {q}")
    print()
    return True


def _share_phrase(t: dict) -> str:
    """The whole-pipeline share, said as precisely as it is true.

    Three cases, and only one of them is a bound:

    - every stage sized, every stage measured — the share, flat.
    - every stage sized, some unmeasured — a genuine lower bound. Those stages
      are already in the denominator, so measuring one can only add matches.
    - some stage unsized — not a bound in either direction. `totals()` drops an
      unsized stage from the denominator *and* the numerator, so sizing it later
      moves both, and if its own rate is below this aggregate the share falls.
      Saying "at least" there is arithmetic nobody can defend.
    """
    pct = _fmt_share(t["share"])
    if t["unsized"]:
        return f"{pct} of the {_fmt_est(t['size_tokens'])} that could be sized"
    return f"at least {pct}" if t["measured"] < t["stages"] else pct


def _fmt_share(share: float) -> str:
    """A share as a percentage, with enough digits to be a number.

    A question answered only over post-training is a rounding error against
    5.93T pretraining tokens, and "0.00%" would read as "none" rather than as
    the three-orders-of-magnitude gap that is the actual finding.
    """
    pct = share * 100
    return f"{pct:.2f}%" if pct >= 0.01 or pct == 0 else f"{pct:.2g}%"


BUDGET_COLS = f"{'stage':<14}{'fit tokens':>11}  {'sampled':>13}  {'rate':>9}  {'matching tokens':>17}"


def _budget_row(s: dict) -> str:
    size = _fmt_est(s.get("size_tokens")) + ("*" if s.get("size_is_floor") else " ")
    if not s.get("measured"):
        # "never asked" and "asked, but nothing came back that could be weighed"
        # are different facts about the stage, and collapsing them would read as
        # a gap in the run rather than a gap in the data.
        why = s.get("unusable") or "not measured"
        return f"{s['stage']:<14}{size:>12}  {why}"
    sampled = f"{s['matched']}/{s['n']}  {s['count_rate'] * 100:4.1f}%"
    matching = _fmt_est(s.get("matching_tokens"))
    ci = s.get("matching_tokens_ci")
    if ci:
        matching += f" ({_fmt_est(ci[0])}–{_fmt_est(ci[1])})"
    # `rate` is the estimator the stage's sampling design calls for, which is
    # not the same rule for both halves of the pipeline — see the footnote.
    return (
        f"{s['stage']:<14}{size:>12}  {sampled:>13}  {s['rate'] * 100:8.1f}%"
        f"  {matching:>17}"
    )


# Exit code for "there is nothing here to add up yet" — no ask run, or only
# unusable ones. Distinct from 1 because an uncaught exception also exits 1, and
# a caller that tolerates a missing measurement must not thereby tolerate a
# traceback. `scripts/human_life_value.sh` is that caller.
NO_MEASUREMENT = 3


def cmd_budget(args):
    """Roll an `ask` question up into a share of the training budget.

    Reads committed runs only. Every stage the question was never asked of
    prints "not measured" rather than dropping out, because a total that
    silently excludes 5.93T tokens of pretraining is the exact error this
    command exists to prevent.
    """
    est = budget.estimate(args.target, args.slug)
    measured = [s for s in est["stages"] if s.get("measured")]
    if not measured:
        # An artifact that exists and cannot be used is not a missing artifact,
        # and telling someone to re-run the command that produced it sends them
        # in a circle. Each stage already recorded why it failed; print that.
        unusable = [s for s in est["stages"] if s.get("unusable")]
        if unusable:
            print(
                f"every ask run for {args.target} under slug {args.slug!r} is unusable:",
                file=sys.stderr,
            )
            for s in unusable:
                print(f"  {s['stage']}: {s['unusable']}", file=sys.stderr)
            for s in unusable:
                for note in s.get("notes", []):
                    print(f"  {s['stage']}: {note}", file=sys.stderr)
            sys.exit(NO_MEASUREMENT)
        print(
            f"no ask run for {args.target} with slug {args.slug!r}"
            f" — run `trainspotting ask {args.target} \"...\" --slug {args.slug}`",
            file=sys.stderr,
        )
        sys.exit(NO_MEASUREMENT)
    print(f"# Training budget — {args.target}\n")
    mixed = _warn_mixed_questions(est)
    if not mixed:
        print(f"question: {est['question']}\n")

    print(BUDGET_COLS)
    print("-" * len(BUDGET_COLS))
    for s in est["stages"]:
        print(_budget_row(s))

    print()
    for family, label in (("pretrain", "pretraining"), ("post-training", "post-training"), ("all", "whole pipeline")):
        t = est["totals"][family]
        if not t["stages"]:
            continue
        line = f"{label:<16}{_fmt_est(t['size_tokens']):>10} fit tokens"
        # A measured stage that could not be sized is dropped from both the
        # denominator and the matching sum, so the share is as partial as an
        # unasked one — `unsized` has to count here too.
        partial = t["measured"] < t["stages"] or bool(t["unsized"])
        if mixed:
            line += "  →  no single total (see above)"
        elif t["measured"]:
            # The denominator is every sized stage, asked or not, so with one
            # still unasked this is a lower bound on the whole pipeline — not a
            # share of the part that was measured. Saying "at least" is the
            # difference between a number and a wrong number: the corpora are
            # 99.7% of these tokens. `_share_phrase` also knows when it is not
            # a bound at all.
            line += f"  →  {_fmt_est(t['matching_tokens'])} matching  ({_share_phrase(t)})"
        if partial:
            line += f"  [{t['stages'] - t['measured']} stage(s) not measured"
            # Naming the measured size is what stops the share above reading as
            # a share of the whole thing — but with nothing measured at all,
            # "0 of it was" is noise on top of "not measured".
            line += (
                f"; {_fmt_est(t['measured_size_tokens'])} of it was]"
                if t["measured"]
                else "]"
            )
        print(line)

    floors = est["totals"]["all"]["floor"]
    unsized = est["totals"]["all"]["unsized"]
    if floors:
        print(
            f"\n* {', '.join(floors)} sized at one reference rollout per prompt — a floor."
            " The rollouts the policy was actually fit to are not in the published mix."
        )
    if unsized:
        print(f"\nno size for: {', '.join(unsized)} — excluded from every total above")
    notes = [(s["stage"], n) for s in est["stages"] for n in s.get("notes", [])]
    if notes:
        print()
        for stage, note in notes:
            print(f"note ({stage}): {note}")
    print(_weighting_footnote(est))
    if args.json:
        path = _write_json(RESULTS / f"{args.target}.budget-{args.slug}.json", est)
        print(f"\nwrote {path}", file=sys.stderr)
