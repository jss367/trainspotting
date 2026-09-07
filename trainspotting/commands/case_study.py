"""`case-study`: run a committed lookup study and write its result file."""

import json
import sys

from .. import casestudy, lookup
from ..paths import RESULTS


def cmd_case_study(args):
    """Run a committed lookup study and write its result file for the site."""
    if args.slug not in casestudy.CASE_STUDIES:
        sys.exit(f"unknown case study {args.slug!r}; known: {', '.join(casestudy.CASE_STUDIES)}")
    RESULTS.mkdir(exist_ok=True)

    def progress(query, index):
        line = f"  {lookup.INDEX_BY_ID[index]['label']} · {query}"
        print(f"\r{line[:78]:<78}", end="", file=sys.stderr, flush=True)

    out = casestudy.run(args.slug, progress=progress)
    print(file=sys.stderr)
    path = RESULTS / f"case-study.{args.slug}.json"
    path.write_text(json.dumps(out, indent=2))

    probe, spread = out["probe"], out["spread"]
    print(
        f"{probe['query']!r}: {probe['occurrences']} occurrence(s) in "
        f"{len(probe['documents'])} document(s)"
        + ("" if probe["exhaustive"] else " (sampled)"),
        file=sys.stderr,
    )
    top = spread["domains"][:3]
    print(
        f"{spread['query']!r}: {spread['occurrences']:,} occurrences; of "
        f"{spread['drawn']} drawn, "
        + ", ".join(f"{d['domain']} {d['share'] * 100:.0f}%" for d in top),
        file=sys.stderr,
    )
    print(f"-> {path}", file=sys.stderr)
