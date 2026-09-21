"""Measure changes between explicitly identified checkpoints."""

import json

from .. import changes, paths, registry


def cmd_changes(args):
    target = registry.resolve(args.target)
    if not target["is_model"]:
        raise SystemExit("Training change needs a model, not a standalone dataset")
    try:
        plan = json.loads(args.plan.read_text()) if args.plan else changes.default_plan(args.target)
        changes.validate_plan(plan, {s["stage"] for s in target["stages"]})
        probes = changes.read_probes(args.probes or changes.PROBES)
        result = changes.measure(args.target, plan, probes, args.device)
    except ImportError as error:
        raise SystemExit("Install checkpoint measurement dependencies: pip install -e '.[changes]'") from error
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    output = paths.RESULTS / f"{args.target}.changes.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"Wrote {output}")
