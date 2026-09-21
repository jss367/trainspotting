"""`pairs`: what separates the two sides of a preference pair, besides the answer.

Reads the committed `context` run for each DPO stage and writes
<target>.<stage>.pairs.json: how often the chosen side is the longer one, by how
much, how much of the gap is thinking rather than answering, which generator
wrote each side, and the same length rate per source. No network, no API key,
no model — the lengths are already in the records.
"""

import json
import sys

from .. import pairs, paths, registry
from ..paths import RESULTS
from .common import _select_stages, _write_json


def _fmt_rate(r: dict) -> str:
    if not r.get("n"):
        return "no pairs"
    ci = "" if r.get("lo") is None else f" (95% CI {r['lo']:.1%}–{r['hi']:.1%})"
    return f"{r['k']}/{r['n']} = {r['rate']:.1%}{ci}"


def cmd_pairs(args):
    wrote = 0
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        if registry.stage_kind(s) != "dpo":
            continue
        path = paths.find(f"{args.target}.{s['stage']}.context.json")
        if not path:
            print(
                f"{s['stage']}: no context run — run `context {args.target} --stage {s['stage']}` first",
                file=sys.stderr,
            )
            continue
        out = pairs.stage_pairs(json.loads(path.read_text()))
        written = _write_json(RESULTS / f"{args.target}.{s['stage']}.pairs.json", out)
        rule = out["length_rule"]
        delta = out["delta"]
        print(
            f"{s['stage']}: the longer side is the chosen one in {_fmt_rate(rule)}"
            f" of the {out['n']} sampled pairs"
            + (f", {out['ties']} tied" if out["ties"] else "")
            + f"\n  chosen − rejected: mean {delta['mean']:+,.0f} chars, median {delta['median']:+,.0f}",
            file=sys.stderr,
        )
        if out.get("split"):
            print(
                f"  of which reasoning {out['split']['reasoning']['mean']:+,.0f}"
                f" and answer {out['split']['answer']['mean']:+,.0f}",
                file=sys.stderr,
            )
        rules = out["models"]["rule"]
        if rules:
            # Said as "fits", never "predicts": the rule is read off the same
            # rows it is scored on, so this is the ceiling for a generator-name
            # rule on this sample and not an estimate of one on any other.
            print(
                f"  generator names alone fit the choice in {_fmt_rate(rules)}"
                f" over {rules['matchups']} matchup(s)",
                file=sys.stderr,
            )
        print(f"  -> {written}", file=sys.stderr)
        wrote += 1
    if not wrote:
        sys.exit(f"{args.target} has no preference stage with a committed context run")
