"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/sources runs:
    python3 scripts/export_site_data.py
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

copied = []
for f in sorted((ROOT / "results").glob("*.json")):
    shutil.copy(f, out / f.name)
    copied.append(f.name)
(out / "manifest.json").write_text(json.dumps(copied, indent=2))
print(f"wrote registry.json + manifest.json and copied {len(copied)} result files")
