"""`grep`: exact string search over every row of each post-training mix."""

import sys

from .. import grep, hf, influence, registry
from ..paths import RESULTS
from .common import _filename_part, _fmt_bytes, _select_stages, _stamp, _write_json


def _grep_plan(con, args, stages):
    """What each selected stage would cost to scan, before anything is read.

    Resolving the whole plan first is what lets the byte cap be one decision
    rather than a surprise partway through: a Think SFT mix is tens of gigabytes
    of message text where the DPO and RL mixes are one or two, and the difference
    only shows up here.
    """
    plan = []
    for s in stages:
        listing = grep.parquet_listing(s["hf_dataset"])
        schema = grep.schema(con, listing["urls"][0])
        exprs, _, unsearched = grep.text_fields(schema, args.field)
        # What the mix holds, as opposed to what this run reads. A result file
        # with only the second cannot say whether an absent response count means
        # `--field` narrowed the search or the mix has no response column, and
        # those are opposite readings of the same blank. Schema only, no reads.
        available, _, _ = grep.text_fields(schema, None)
        if not exprs:
            sys.exit(
                f"{s['stage']}: no text columns for field(s) {', '.join(args.field or grep.GROUPS)}"
                f" in {s['hf_dataset']}"
            )
        source, source_column = grep.source_expr(
            schema, [args.by] if args.by else s["source_columns"]
        )
        if args.by and not source:
            sys.exit(f"{s['stage']}: no text column {args.by!r} in {s['hf_dataset']}")
        leaves = grep.plan_leaves(schema, args.field, source_column)
        plan.append({
            "stage": s,
            "listing": listing,
            # Resolved before the scan, like every other layer that draws rows:
            # a lookup afterwards could name a revision published while a
            # multi-gigabyte read was in flight.
            "revision": hf.dataset_revision(s["hf_dataset"]),
            "schema": schema,
            "exprs": exprs,
            "source": source,
            "source_column": source_column,
            "unsearched": unsearched,
            "available": list(available),
            "rows": grep.total_rows(con, listing["urls"]),
            "bytes": grep.byte_cost(con, listing["urls"], leaves),
        })
    return plan


def cmd_grep(args):
    """Count rows of every post-training mix whose text contains a pattern.

    Exact, over all rows, which is the half of the question sampling cannot do.
    `classify` and `ask` estimate an unconditional rate from 300 prompts; a
    pattern that occurs in 0.1% of a mix is expected to miss such a sample
    entirely, and no interval around zero tells you it is there.
    """
    con = grep.connect()
    stages = _select_stages(args, registry.post_training_stages, "post-training")
    plan = _grep_plan(con, args, stages)

    total_bytes = sum(p["bytes"] for p in plan)
    print(f"# grep {args.pattern!r} — {len(plan)} stage(s), {_fmt_bytes(total_bytes)} to read\n", file=sys.stderr)
    for p in plan:
        fields = "/".join(p["exprs"])
        print(
            f"- {p['stage']['stage']:6s} {p['rows']:>9,} rows  {_fmt_bytes(p['bytes']):>9}"
            f"  {fields}  ({p['stage']['hf_dataset']})",
            file=sys.stderr,
        )
    cap = int(args.max_gb * 1e9)
    if total_bytes > cap and not args.yes:
        sys.exit(
            f"\nthat is {_fmt_bytes(total_bytes)}, over the {args.max_gb} GB cap, and nothing has been "
            f"read yet. Narrow it (--stage, --field) or allow it (--max-gb {total_bytes / 1e9:.1f}, or --yes)."
        )

    # `slugify` already yields a filename-safe slug, but an explicit `--slug`
    # is raw user input that lands in a path, and `_write_json` creates parent
    # directories: `--slug a/b` would quietly file a multi-gigabyte scan where
    # neither `_grep_traces` nor the site export looks, and `../..` would write
    # outside results/ entirely. Same treatment `find` gives its components.
    slug = _filename_part(args.slug) if args.slug else grep.slugify(args.pattern)
    written = []
    for p in plan:
        s = p["stage"]
        if p["listing"]["partial"]:
            print(
                f"\n{s['stage']}: WARNING — the server converted only part of this repo, "
                "so every count below is a lower bound",
                file=sys.stderr,
            )
        if p["unsearched"]:
            print(
                f"\n{s['stage']}: not searched: {', '.join(p['unsearched'])}"
                " — text columns this layer does not recognise as prompt, response or reference",
                file=sys.stderr,
            )
        print(f"\nscanning {s['stage']} ({_fmt_bytes(p['bytes'])}) ...", file=sys.stderr)
        result = grep.scan(
            con,
            grep.read_parquet_sql(p["listing"]["urls"]),
            p["exprs"],
            p["source"],
            args.pattern,
            regex=args.regex,
            case_sensitive=args.case_sensitive,
            examples=args.examples,
        )
        rows = p["rows"]
        k = result["matched"]
        print(f"{s['stage']}: {k:,}/{rows:,} rows = {k / rows * 100 if rows else 0:.3f}%", file=sys.stderr)
        for group, n in result["by_group"].items():
            print(f"  {group:10s} {n:,}", file=sys.stderr)
        # Unconditional, though a zero-match stage has no percentages to
        # compute. Skipping it when `k == 0` saved one read of a label column and
        # cost the plan its meaning: the priced bytes always included both source
        # reads, so a pattern absent from the mix was charged for a query that
        # never ran. Running it always makes the quoted figure the figure, and it
        # earns its keep — a stage with no matches still reports what the
        # denominators were, so "0 of 48,398 rlvr_general_mix rows" is sayable
        # rather than just "0".
        totals = (
            grep.source_totals(con, grep.read_parquet_sql(p["listing"]["urls"]), p["source"])
            if p["source"] else {}
        )
        shown = list(result["by_source"].items())[:12]
        for src, n in shown:
            of = totals.get(src)
            rate = f" = {n / of * 100:5.2f}% of it" if of else ""
            print(f"  {n:>7,} / {of or rows:>9,}{rate}  {src}", file=sys.stderr)
        if len(result["by_source"]) > len(shown):
            rest = len(result["by_source"]) - len(shown)
            print(f"  … and {rest} more source(s), all of them in the result file", file=sys.stderr)
        payload = {
            "dataset": s["hf_dataset"],
            "stage": s["stage"],
            "pattern": args.pattern,
            "slug": slug,
            "regex": args.regex,
            "case_sensitive": args.case_sensitive,
            "fields": list(p["exprs"]),
            "available_fields": p["available"],
            "source_column": p["source_column"],
            # `_stamp` carries `generated` and the *dataset* revision every
            # other result file records. The Parquet-branch revision is a
            # second, different tree — the server's conversion of that
            # dataset — so it travels under its own name rather than
            # overwriting the one the rest of the tool means by "revision".
            **_stamp(s["hf_dataset"], revision=p["revision"]),
            "parquet_revision": p["listing"]["revision"],
            "partial": p["listing"]["partial"],
            "shards": len(p["listing"]["urls"]),
            "bytes_read": p["bytes"],
            "unsearched_columns": p["unsearched"],
            "total_rows": rows,
            "rows_by_source": totals,
            **result,
        }
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.grep-{slug}.json", payload)
        written.append(payload)
        print(f"  -> {path}", file=sys.stderr)

    # The counts above are one mix each. Read together they are a claim about
    # where a string most plausibly entered the model, which is a different
    # question from how many rows hold it — and one that needs the stages
    # nobody scanned named as such rather than left out.
    target = registry.resolve(args.target)
    # Only for a model. A dataset target is a corpus, not a pipeline: nothing was
    # trained on WildChat's chat log, so ranking its one stage as where a phrase
    # entered a model produced "Most plausibly chat" — a training-origin verdict
    # about a target that has no training. The counts above are exactly as useful
    # for a corpus; it is the ranking that has nothing to rank.
    if target["is_model"]:
        trace = influence.compare(written, target["stages"])
        print("", file=sys.stderr)
        for line in influence.render(trace, args.target, note=True):
            print(line, file=sys.stderr)
