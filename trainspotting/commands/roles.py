"""`roles`: which turn roles every row of each post-training mix carries, and
what its system prompts say."""

import sys

from .. import grep, hf, registry, roles
from ..paths import RESULTS
from .common import _fmt_bytes, _select_stages, _stamp, _write_json


def _plan(con, stages):
    """What each stage would cost to scan, resolved before anything is read —
    the same up-front plan `grep` makes, for the same byte cap."""
    plan = []
    for s in stages:
        listing = grep.parquet_listing(s["hf_dataset"])
        schema = grep.schema(con, listing["urls"][0])
        columns = roles.role_columns(schema)
        source, source_column = grep.source_expr(schema, s["source_columns"])
        plan.append({
            "stage": s,
            "listing": listing,
            "revision": hf.dataset_revision(s["hf_dataset"]),
            "columns": columns,
            "source": source,
            "source_column": source_column,
            "rows": grep.total_rows(con, listing["urls"]),
            # A stage with no turns to read costs nothing: it is written from
            # its footers alone.
            "bytes": grep.byte_cost(con, listing["urls"], roles.plan_leaves(columns, source_column))
            if columns else 0,
        })
    return plan


def cmd_roles(args):
    """Count rows per turn role over every row of each post-training mix, and
    group the system prompts by how they open."""
    con = grep.connect()
    stages = _select_stages(args, registry.post_training_stages, "post-training")
    plan = _plan(con, stages)

    total_bytes = sum(p["bytes"] for p in plan)
    print(f"# roles — {len(plan)} stage(s), {_fmt_bytes(total_bytes)} to read\n", file=sys.stderr)
    for p in plan:
        cols = "/".join(p["columns"]) or "no turn roles stored"
        print(
            f"- {p['stage']['stage']:6s} {p['rows']:>9,} rows  {_fmt_bytes(p['bytes']):>9}"
            f"  {cols}  ({p['stage']['hf_dataset']})",
            file=sys.stderr,
        )
    cap = int(args.max_gb * 1e9)
    if total_bytes > cap and not args.yes:
        sys.exit(
            f"\nthat is {_fmt_bytes(total_bytes)}, over the {args.max_gb} GB cap, and nothing has been "
            f"read yet. Narrow it (--stage) or allow it (--max-gb {total_bytes / 1e9:.1f}, or --yes)."
        )

    for p in plan:
        s = p["stage"]
        if p["listing"]["partial"]:
            print(
                f"\n{s['stage']}: WARNING — the server converted only part of this repo, "
                "so every count below is a lower bound",
                file=sys.stderr,
            )
        rows = p["rows"]
        if p["columns"]:
            print(f"\nscanning {s['stage']} ({_fmt_bytes(p['bytes'])}) ...", file=sys.stderr)
            from_sql = grep.read_parquet_sql(p["listing"]["urls"])
            result = roles.scan(con, from_sql, p["columns"], p["source"])
            totals = grep.source_totals(con, from_sql, p["source"]) if p["source"] else {}
            system = result["system"]
            k, base = system["rows"], result["rows_with_turns"]
            # Over the rows that store turns. The RL mixes keep a conversation
            # for a few thousand rows and a flat string for the rest, and the
            # rest are not rows without a system turn.
            print(
                f"{s['stage']}: {k:,}/{base:,} rows with stored turns = "
                f"{k / base * 100 if base else 0:.2f}% have a system turn "
                f"({system['distinct_texts']:,} distinct texts)"
                + (f"; {rows - base:,} of {rows:,} rows store no turns" if base < rows else ""),
                file=sys.stderr,
            )
            for role, n in result["rows_by_role"].items():
                print(f"  {n:>9,}  {role}", file=sys.stderr)
            for pr in system["prompts"][:5]:
                print(f"  {pr['rows']:>9,}  {pr['preview'][:100]!r}", file=sys.stderr)
        else:
            print(
                f"\n{s['stage']}: no column stores turn roles — the prompt is one string, so "
                "whether a system turn wrapped it is up to the chat template, not the data",
                file=sys.stderr,
            )
            result, totals = {"total_rows": rows}, {}
        payload = {
            "dataset": s["hf_dataset"],
            "stage": s["stage"],
            **_stamp(s["hf_dataset"], revision=p["revision"]),
            "parquet_revision": p["listing"]["revision"],
            "partial": p["listing"]["partial"],
            "shards": len(p["listing"]["urls"]),
            "bytes_read": p["bytes"],
            # Empty when the mix stores no turn roles at all. The page reads that
            # as "not recorded", which is a different claim from zero system turns.
            "columns": p["columns"],
            "source_column": p["source_column"],
            "preview_chars": roles.PREVIEW_CHARS,
            "rows_by_source": totals,
            **result,
        }
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.roles.json", payload)
        print(f"  -> {path}", file=sys.stderr)
