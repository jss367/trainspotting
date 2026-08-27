"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/ask/context runs:
    python3 scripts/export_site_data.py

Context records are copied minified (they are bulk text the site fetches on
click); everything else is copied verbatim so its diffs stay readable.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import registry  # noqa: E402

out = ROOT / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)

(out / "registry.json").write_text(json.dumps(registry.MODELS, indent=2))

copied, total = [], 0
for f in sorted((ROOT / "results").glob("*.json")):
    if f.name.endswith(".context.json"):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    total += (out / f.name).stat().st_size
    copied.append(f.name)
(out / "manifest.json").write_text(json.dumps(copied, indent=2))
print(f"wrote registry.json + manifest.json and {len(copied)} result files ({total / 1e6:.1f} MB)")
