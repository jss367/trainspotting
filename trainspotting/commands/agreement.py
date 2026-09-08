"""`agreement`: check the classifier against a second run of itself.

`classify --replicate` labels the same draw again and writes it beside the main
run. This joins the two on row index (or, for runs that predate the row index,
on the prompt prefix the site joins on), and writes
<target>.<stage>.agreement.json with the agreement rate, its interval, Cohen's
kappa, the confusion, and each label's share under both runs. The site prints
it next to the shares it qualifies.
"""

import json
import sys

from .. import agreement, paths, registry
from ..paths import RESULTS
from .common import _select_stages, _write_json


def _load(path):
    return json.loads(path.read_text())


def _fmt(summary: dict) -> str:
    if not summary["n"]:
        return "no pairs"
    lo, hi = summary["accuracy_ci"]
    kappa = summary["kappa"]
    k = "κ undefined" if kappa is None else f"κ {kappa:.2f}"
    return f"{summary['agree']}/{summary['n']} same label = {summary['accuracy']:.1%} (95% CI {lo:.0%}–{hi:.0%}), {k}"


def cmd_agreement(args):
    wrote = 0
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        labels_path = paths.find(f"{args.target}.{s['stage']}.labels.json")
        if not labels_path:
            print(f"{s['stage']}: no labels run", file=sys.stderr)
            continue
        rep_path = paths.find(f"{args.target}.{s['stage']}.labels-replicate.json")
        if not rep_path:
            print(
                f"{s['stage']}: nothing to check against — run `classify {args.target} --replicate` first",
                file=sys.stderr,
            )
            continue
        labels, rep = _load(labels_path), _load(rep_path)
        compared = agreement.compare(labels["records"], rep["records"])
        compared["classifier"] = rep.get("classifier")
        compared["generated"] = rep.get("generated")
        # Only rows both runs drew compare, so a replicate at another --sample
        # or --seed is a smaller check than it looks; say so rather than let
        # the pair count stand in for the sample size.
        compared["same_draw"] = (rep.get("sample"), rep.get("seed")) == (
            labels.get("sample"),
            labels.get("seed"),
        )
        out = {
            "dataset": labels["dataset"],
            "labels_run": {
                k: labels.get(k) for k in ("sample", "seed", "classifier", "system_sha", "revision", "generated")
            },
            "replicate": compared,
        }
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.agreement.json", out)
        note = "" if compared["same_draw"] else "\n  note: the replicate was drawn with a different --sample/--seed; only shared rows compare"
        print(
            f"{s['stage']}: vs a second run of {rep.get('classifier')}: {_fmt(compared)}{note}\n  -> {path}",
            file=sys.stderr,
        )
        wrote += 1
    if not wrote:
        sys.exit(1)
