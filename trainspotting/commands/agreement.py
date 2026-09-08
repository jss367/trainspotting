"""`gold` and `agreement`: check the classifier against a person and against itself.

`gold` draws a blind, stratified set of already-labeled prompts and writes it
under results/gold/ with an empty `human_label` on every item, for a person to
fill in under the same rubric the classifier used. `agreement` reads that file
back, joins it to the labels run, and writes <target>.<stage>.agreement.json
with the accuracy, its interval, Cohen's kappa and the confusion — plus, when a
`classify --replicate` run exists for the stage, the same arithmetic between the
two classifier runs. The site prints both next to the shares they qualify.
"""

import json
import sys

from .. import agreement, classify, paths, registry
from ..paths import RESULTS
from .common import _select_stages, _write_json

GOLD = RESULTS / "gold"


def _gold_path(target: str, stage: str):
    return GOLD / f"{target}.{stage}.gold.json"


def _load(path):
    return json.loads(path.read_text())


def cmd_gold(args):
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        labels_path = paths.find(f"{args.target}.{s['stage']}.labels.json")
        if not labels_path:
            print(f"{s['stage']}: no labels run to draw from — run `classify` first", file=sys.stderr)
            continue
        out = _gold_path(args.target, s["stage"])
        if out.exists() and any(i.get("human_label") for i in _load(out)["items"]):
            # A gold file with labels in it is hours of somebody's work. Refuse
            # rather than overwrite; deleting it is a deliberate act.
            sys.exit(f"{out} already holds hand labels; delete it to redraw")
        data = _load(labels_path)
        items = agreement.draw_gold(data["records"], args.per_label, args.seed)
        kind = registry.stage_kind(s)
        payload = {
            "dataset": data["dataset"],
            # Which labels run this was drawn from. `agreement` joins on row
            # index, which only means anything within one draw of one revision.
            "labels_run": {
                k: data.get(k) for k in ("sample", "seed", "classifier", "system_sha", "revision", "generated")
            },
            "per_label": args.per_label,
            "seed": args.seed,
            "labels": agreement.LABELS,
            # The exact rubric the classifier read. A person labeling under a
            # different one would be measuring the rubric, not the classifier.
            "rubric": classify.system_for(kind) or classify.SYSTEM,
            "items": items,
        }
        GOLD.mkdir(parents=True, exist_ok=True)
        path = _write_json(out, payload)
        print(
            f"{s['stage']}: {len(items)} prompts drawn (up to {args.per_label} per label) -> {path}\n"
            f"  fill in `human_label` on each item with one of: {', '.join(agreement.LABELS)}\n"
            f"  then run: trainspotting agreement {args.target}",
            file=sys.stderr,
        )


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
        labels = _load(labels_path)
        out = {
            "dataset": labels["dataset"],
            "labels_run": {
                k: labels.get(k) for k in ("sample", "seed", "classifier", "system_sha", "revision", "generated")
            },
        }
        lines = []
        gold_path = _gold_path(args.target, s["stage"])
        if gold_path.exists():
            gold = _load(gold_path)
            scored = agreement.score(gold["items"], labels["records"])
            scored["gold_file"] = str(gold_path.relative_to(RESULTS.parent))
            out["gold"] = scored
            if scored["n"]:
                lines.append(f"  vs {scored['n']} hand labels: {_fmt(scored)}")
            if scored["unlabeled_gold"]:
                lines.append(f"  {scored['unlabeled_gold']} gold items still unlabeled")
            if scored["missing_rows"]:
                lines.append(
                    f"  {scored['missing_rows']} gold rows are not in the labels run — redraw with `gold`"
                )
            if scored["invalid_labels"]:
                lines.append(f"  labels outside the taxonomy in the gold file: {scored['invalid_labels']}")
        rep_path = paths.find(f"{args.target}.{s['stage']}.labels-replicate.json")
        if rep_path:
            rep = _load(rep_path)
            compared = agreement.compare(labels["records"], rep["records"])
            compared["classifier"] = rep.get("classifier")
            compared["generated"] = rep.get("generated")
            compared["same_draw"] = (rep.get("sample"), rep.get("seed")) == (
                labels.get("sample"),
                labels.get("seed"),
            )
            out["replicate"] = compared
            lines.append(f"  vs a second run of {rep.get('classifier')}: {_fmt(compared)}")
            if not compared["same_draw"]:
                lines.append("  note: the replicate was drawn with a different --sample/--seed; only shared rows compare")
        if "gold" not in out and "replicate" not in out:
            print(
                f"{s['stage']}: nothing to check against — run `gold` and label it, or `classify --replicate`",
                file=sys.stderr,
            )
            continue
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.agreement.json", out)
        print(f"{s['stage']}:\n" + "\n".join(lines) + f"\n  -> {path}", file=sys.stderr)
        wrote += 1
    if not wrote:
        sys.exit(1)
