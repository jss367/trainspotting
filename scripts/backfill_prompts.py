"""Upgrade the 1500-character prompts in old results/ files to full prompts.

Sampling is deterministic (same dataset size + seed), so this re-fetches the
same rows and rewrites each record's prompt without re-classifying. Responses
live in the context files that `trainspotting context` writes, not here.

    python3 scripts/backfill_prompts.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import extract, hf, registry  # noqa: E402

# The stage token in a result filename is a kind, so read the alternation off
# the registry rather than hardcoding the three model stages — a dataset's files
# are named for its own kind (wildchat-1m.chat.labels.json) and a literal list
# would skip them without saying so.
STAGE_RE = re.compile(rf"^(.+?)\.({'|'.join(registry.KINDS)})\.")

for path in sorted((ROOT / "results").glob("*.json")):
    m = STAGE_RE.match(path.name)
    if not m:
        continue
    target_name, stage_name = m.groups()
    stage = next(
        s for s in registry.post_training_stages(registry.resolve(target_name))
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
    for rec, (_, full) in zip(data["records"], keep):
        if rec["prompt"][:200] != full[:200]:
            mismatches += 1
            continue
        rec["prompt"] = extract.clip(full)
    path.write_text(json.dumps(data, indent=2))
    status = f"{mismatches} mismatched rows left untouched" if mismatches else "all rows matched"
    print(f"backfilled {path.name}: {status}")
