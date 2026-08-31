"""Upgrade the 1500-character prompts in old results/ files to full prompts.

Re-fetches the stage's draw and rewrites each record's prompt without
re-classifying. Responses live in the context files that `trainspotting context`
writes, not here.

Records are matched to re-fetched prompts by their stored prefix, not by
position. Position was never a safe join: rows carrying no user prompt drop out
of the draw, and the sampler has since learned to deduplicate overlapping pages
and top the sample back up, which changes both membership and order for the same
(sample, seed). Matching on content means a record is upgraded when its row is
still in the draw and left alone when it is not, whichever sampler wrote it.

    python3 scripts/backfill_prompts.py
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
    # Keyed on the prefix the records store, so a record finds its own row
    # wherever the draw put it. A prefix collision would be two rows that agree
    # for 200 characters and disagree after; drop both rather than guess.
    by_prefix = {}
    for full in prompts:
        if full:
            by_prefix.setdefault(full[:200], []).append(full)
    unique = {k: v[0] for k, v in by_prefix.items() if len(v) == 1}

    upgraded = ambiguous = 0
    missing = []
    for rec in data["records"]:
        prefix = rec["prompt"][:200]
        if prefix in unique:
            rec["prompt"] = extract.clip(unique[prefix])
            upgraded += 1
        elif prefix in by_prefix:
            ambiguous += 1
        else:
            missing.append(prefix)
    if upgraded:
        path.write_text(json.dumps(data, indent=2))
    parts = [f"{upgraded} upgraded"]
    if missing:
        parts.append(f"{len(missing)} not in this draw")
    if ambiguous:
        parts.append(f"{ambiguous} ambiguous prefix")
    print(f"backfilled {path.name}: {', '.join(parts)} of {len(data['records'])}")
