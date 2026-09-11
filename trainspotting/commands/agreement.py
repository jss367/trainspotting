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


PROVENANCE = ("dataset", "revision", "system_sha", "classifier", "sample", "seed")


def _comparison_issues(first, second):
    """Matching row numbers establish repeatability only for a known instrument/draw."""
    issues = []
    for key in PROVENANCE:
        a, b = first.get(key), second.get(key)
        if a is None or a == "" or b is None or b == "":
            issues.append(f"unknown {key}")
        elif a != b:
            issues.append(f"different {key}")
    if first.get("revision_moved_to") or second.get("revision_moved_to"):
        issues.append("dataset revision moved during a run")
    return issues


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
        # Raw intersecting-row statistics remain available for inspection, but
        # unknown or changed provenance cannot establish repeatability.
        issues = _comparison_issues(labels, rep)
        compared.update({k: rep.get(k) for k in (*PROVENANCE, "generated", "revision_moved_to")})
        compared["same_draw"] = not issues
        compared["comparison_issues"] = issues
        out = {
            "dataset": labels.get("dataset"),
            "labels_run": {
                k: labels.get(k) for k in (*PROVENANCE, "generated", "revision_moved_to")
            },
            "replicate": compared,
        }
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.agreement.json", out)
        note = "" if not issues else (
            "\n  note: not a repeatability check: " + "; ".join(issues)
            + "; statistics join shared row identifiers only"
        )
        print(
            f"{s['stage']}: vs a second run of {rep.get('classifier')}: {_fmt(compared)}{note}\n  -> {path}",
            file=sys.stderr,
        )
        wrote += 1
    if not wrote:
        sys.exit(1)
