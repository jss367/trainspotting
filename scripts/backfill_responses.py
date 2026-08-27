"""Backfill full prompts and chosen/rejected/target responses into results/ files.

Result files written before responses were stored have 1500-char prompts and no
completions. Sampling is deterministic (same dataset size + seed), so this
re-fetches the same rows and merges the missing fields without re-classifying.

    python3 scripts/backfill_responses.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import extract, hf, registry  # noqa: E402

for path in sorted((ROOT / "results").glob("*.json")):
    m = re.match(r"^(.+?)\.(sft|dpo|rlvr)\.", path.name)
    if not m:
        continue
    model_name, stage_name = m.groups()
    stage = next(
        s for s in registry.post_training_stages(registry.get_model(model_name))
        if s["stage"] == stage_name
    )
    data = json.loads(path.read_text())
    rows = hf.sample_rows(stage["hf_dataset"], data["sample"], seed=data["seed"])
    prompts = [extract.extract_prompt(r, stage["prompt_path"]) for r in rows]
    keep = [(rows[i], p) for i, p in enumerate(prompts) if p]
    if len(keep) < len(data["records"]):
        print(f"SKIP {path.name}: resample gave {len(keep)} prompts for {len(data['records'])} records")
        continue
    mismatches = 0
    for rec, (row, full) in zip(data["records"], keep):
        if rec["prompt"][:200] != full[:200]:
            mismatches += 1
            continue
        rec["prompt"] = extract.clip(full)
        for k, v in extract.extract_responses(row, stage["prompt_path"]).items():
            rec[k] = extract.clip(v)
    path.write_text(json.dumps(data, indent=2))
    status = f"{mismatches} mismatched rows left untouched" if mismatches else "all rows matched"
    print(f"backfilled {path.name}: {status}")
