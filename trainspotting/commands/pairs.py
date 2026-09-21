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
    """Two different non-results, and only one of them is a failure.

    A target with no preference stage at all is not a failed run: the question
    this command asks does not apply to it, the same way `rewards` has nothing
    to say about an SFT-only pipeline. It exits 0 so a caller like
    `scripts/refresh_samples.sh` does not have to blanket the status with
    `|| true` — which is how a *real* failure, a preference stage whose context
    run is missing or holds no DPO records, used to be swallowed in the middle of
    a refresh that had just written that very run.
    """
    wrote, stages, failed = 0, 0, False
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        if registry.stage_kind(s) != "dpo":
            continue
        stages += 1
        path = paths.find(f"{args.target}.{s['stage']}.context.json")
        if not path:
            print(
                f"{s['stage']}: no context run — run `context {args.target} --stage {s['stage']}` first",
                file=sys.stderr,
            )
            failed = True
            continue
        try:
            out = pairs.stage_pairs(json.loads(path.read_text()))
        except ValueError as exc:
            # The context run exists but carries no DPO records — a stage the
            # registry calls preference whose sample says otherwise. Reported as
            # a line rather than a traceback, and still an exit code, because the
            # run that wrote it thought it succeeded.
            print(f"{s['stage']}: {exc}", file=sys.stderr)
            failed = True
            continue
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
    if not stages:
        print(
            f"{args.target} has no preference stage among the selected ones — nothing to measure",
            file=sys.stderr,
        )
        return
    if failed or not wrote:
        sys.exit(
            f"{args.target} has a preference stage this could not measure —"
            " see the lines above"
        )
