"""`steps`: count a string along Pythia's published training order."""

import sys

from .. import pretrain, registry, steps
from ..paths import RESULTS
from .common import _filename_part, _fmt_bytes, _fmt_est, _fmt_tokens, _pattern_slug, _stamp, _write_json


# Checkpoints the exposure table prints. The file carries all 154; on a terminal
# five spread across the run say the shape, and the rest is a straight line.
_SHOWN_CHECKPOINTS = (1000, 10_000, 50_000, 100_000, 143_000)


def cmd_steps(args):
    """Count a pattern in sampled training batches, in the order the model saw them.

    Only for a target whose order is published, which is Pythia. Each sampled
    step is one 4.2 MB range request into the preshuffled token stream, decoded
    with the run's tokenizer and searched sequence by sequence. The output is a
    rate along the run, the same rate over equal slices of it, and what that rate
    says the model had seen by each saved checkpoint.
    """
    target = registry.resolve(args.target)
    found = registry.training_order(target)
    if not found:
        sys.exit(
            f"{args.target} has no published training order. This layer reads the batches in "
            "the sequence the optimizer took them, which of the registered targets only Pythia "
            "publishes; for Olmo the corpus is public and the order is not."
        )
    if not args.pattern:
        sys.exit("an empty pattern matches every position of every sequence")
    s, order = found
    steps.check_layout(order)
    try:
        # `--at` is purposive: it adds a batch for inspection, not another
        # random draw. Keep the probability sample separate so a checkpoint
        # selected precisely because it looks unusual cannot tilt the corpus
        # rate, its interval, or every checkpoint exposure estimate.
        sampled_steps = steps.draw_steps(order["steps"], args.sample, args.seed)
        picks = steps.draw_steps(order["steps"], args.sample, args.seed, args.at or ())
    except ValueError as e:
        sys.exit(str(e))
    rx = steps.compile_pattern(args.pattern, args.regex, args.case_sensitive)
    revision = pretrain.resolve_revision(order["dataset"])
    tokenizer_revision = steps.resolve_tokenizer_revision(order["tokenizer"])
    decode = steps.decoder(order, tokenizer_revision)
    to_read = steps.step_bytes(order) * len(picks)
    explicit = len(picks) - len(sampled_steps)
    read_description = f"{len(sampled_steps)} sampled"
    if explicit:
        read_description += f" + {explicit} explicit"
    print(
        f"# steps {args.pattern!r} — {read_description} of {order['steps']:,} steps, "
        f"{_fmt_bytes(to_read)} to read from {order['dataset']} at {revision[:7]}",
        file=sys.stderr,
    )

    def progress(i, n, step):
        print(f"\r  step {step:>7,}  ({i}/{n})   ", end="", file=sys.stderr, flush=True)

    per_step, examples = steps.scan(
        order,
        picks,
        revision,
        rx,
        decode,
        examples_limit=args.examples,
        progress=progress,
        priority_steps=args.at or (),
    )
    print(file=sys.stderr)

    sampled_set = set(sampled_steps)
    estimate_steps = [count for count in per_step if count["step"] in sampled_set]
    summary = steps.summarize(estimate_steps)
    slices = steps.by_slice(
        estimate_steps, order["steps"], args.slices, deff=summary["design_effect"]
    )
    exposure = steps.exposure(summary, order["checkpoints"], order["sequences_per_step"])
    second = steps.second_pass_step(order)

    n, k = summary["sequences"], summary["matched"]
    print(
        f"{s['stage']}: {k:,}/{n:,} sequences hold it = {k / n * 100 if n else 0:.3f}%"
        f" (95% CI {summary['lo'] * 100:.3f}–{summary['hi'] * 100:.3f}%),"
        f" {summary['occurrences']:,} occurrences, {len(estimate_steps)} sampled steps",
        file=sys.stderr,
    )
    print(f"  by stretch of the run ({args.slices} slices of {order['steps']:,} steps):", file=sys.stderr)
    for sl in slices:
        if sl["sequences"]:
            rate = (
                f"{sl['matched']:>5,}/{sl['sequences']:>7,} = {sl['rate'] * 100:6.3f}%"
                f"  ({sl['lo'] * 100:.3f}–{sl['hi'] * 100:.3f}%)  {sl['steps']} step(s)"
            )
        else:
            rate = "no step sampled"
        print(f"    {sl['from_step']:>7,}–{sl['to_step']:<7,}  {rate}", file=sys.stderr)
    if second:
        print(
            f"  from about step {second:,} the run is re-reading the corpus"
            f" (~{_fmt_tokens(order['corpus_tokens'])} tokens against a"
            f" {_fmt_tokens(_run_tokens(order))} budget)",
            file=sys.stderr,
        )
    print("  expected sequences holding it, seen by checkpoint (if the rate holds along the run):", file=sys.stderr)
    shown = {e["step"]: e for e in exposure}
    for step in _SHOWN_CHECKPOINTS:
        e = shown.get(step)
        if e:
            print(
                f"    step {step:>7,}   ~{_fmt_est(e['expected']):>5}   ({_fmt_est(e['lo'])}–{_fmt_est(e['hi'])})",
                file=sys.stderr,
            )
    for ex in examples[:3]:
        flat = " ".join(ex["snippet"].split())
        print(f"  step {ex['step']:,} seq {ex['sequence']}: …{flat}…", file=sys.stderr)

    slug = (
        _filename_part(args.slug)
        if args.slug
        else _pattern_slug(args.pattern, args.case_sensitive, regex=args.regex)
    )
    payload = {
        "dataset": order["dataset"],
        "stage": s["stage"],
        "name": s["name"],
        "pattern": args.pattern,
        "slug": slug,
        "regex": args.regex,
        "case_sensitive": args.case_sensitive,
        **_stamp(order["dataset"], revision=revision),
        "tokenizer": order["tokenizer"],
        "tokenizer_revision": tokenizer_revision,
        "steps_total": order["steps"],
        "sequence_tokens": order["sequence_tokens"],
        "sequences_per_step": order["sequences_per_step"],
        "sample": len(sampled_steps),
        "seed": args.seed,
        "at": sorted(set(args.at or [])),
        "bytes_read": to_read,
        "second_pass_step": second,
        "caveat": steps.CAVEAT,
        **summary,
        "slices": slices,
        "exposure": exposure,
        "per_step": per_step,
        "examples": examples,
    }
    path = _write_json(RESULTS / f"{args.target}.{s['stage']}.steps-{slug}.json", payload)
    print(f"  -> {path}", file=sys.stderr)


def _run_tokens(order: dict) -> int:
    """The run's token budget as the registry counts it: new tokens per step,
    which is one fewer than a sequence holds."""
    return order["steps"] * order["sequences_per_step"] * (order["sequence_tokens"] - 1)
