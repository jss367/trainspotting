"""Apply the verifier rule to classify runs committed before it existed.

`trainspotting classify` now takes an RLVR row's label from its verifier where
the verifier settles it (`classify.VERIFIER_LABELS`). Earlier runs asked the
model about every prompt, so a row from the IFEval constraint mix came back
labeled by its topic: a jailbreak the constraint checker is happy to see
answered counted as harmlessness content, the opposite of what training does
with it.

Re-running classify would need the API and would move every other label with
it, so this rewrites only the labels the verifier settles. It reads each row's
verifier out of the context file drawn from the same rows, joining on the
400-character prompt prefix the way the site does, so it needs no network and no
key. Only RLVR context records carry a reward, so the other stages fall out on
their own.

    python3 scripts/relabel_by_verifier.py
    python3 scripts/export_site_data.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import classify, context  # noqa: E402


def draw(d: dict) -> tuple:
    """Which rows a run was drawn from. Two files join only if these agree."""
    return d.get("dataset"), d.get("sample"), d.get("seed")


def context_for(base: str, labels: dict) -> tuple[Path | None, list[tuple]]:
    """The context file drawn from the same rows as this label run.

    Context records are a gitignored cache under results/ and a committed copy
    under docs/data/, so a fresh clone relabels from the copy it actually has.
    But `trainspotting context` always writes the same filename, so a local run
    at a different `--sample` or `--seed` sits there under the name of the run
    that matches: joining against it would miss on every prefix and leave the
    labels quietly as they were. Pick by what the file was drawn from, not by
    where it is, and return the rejects so the caller can say what it skipped.
    """
    rejected = []
    for p in (
        ROOT / "results" / f"{base}.context.json",
        ROOT / "docs" / "data" / f"{base}.context.json",
    ):
        if not p.exists():
            continue
        drawn = draw(json.loads(p.read_text()))
        if drawn == draw(labels):
            return p, rejected
        rejected.append((p, drawn))
    return None, rejected


for path in sorted((ROOT / "results").glob("*.labels.json")):
    base = path.name[: -len(".labels.json")]
    data = json.loads(path.read_text())
    ctx_path, rejected = context_for(base, data)
    if ctx_path is None:
        why = (
            "; ".join(
                f"{p.parent.name}/{p.name} was drawn from {d}, this run from {draw(data)}"
                for p, d in rejected
            )
            or "no context file — run `trainspotting context` first"
        )
        print(f"{path.name}: skipped — {why}")
        continue
    kinds = {
        r["key"]: (r.get("reward") or {}).get("kind")
        for r in json.loads(ctx_path.read_text())["records"]
    }
    settled = changed = unjoined = 0
    for rec in data["records"]:
        key = rec["prompt"][: context.KEY_CHARS]
        if key not in kinds:
            unjoined += 1
            continue
        label = classify.VERIFIER_LABELS.get(kinds[key])
        if not label:
            continue
        settled += 1
        changed += rec["label"] != label
        rec["label"], rec["by"] = label, "verifier"
    if settled:
        path.write_text(json.dumps(data, indent=2))
    tail = f", {unjoined} rows had no context record" if unjoined else ""
    src = f" (from {ctx_path.parent.name}/)" if rejected else ""
    print(
        f"{path.name}: {settled} labels from the verifier, {changed} changed{tail}{src}"
    )
