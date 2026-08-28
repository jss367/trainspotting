"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/ask/context runs:
    python3 scripts/export_site_data.py

Context and pretraining-document records are copied minified (they are bulk text
the site fetches on click); everything else is copied verbatim so its diffs stay
readable.
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
    if f.name.endswith((".context.json", ".docs.json")):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    copied.append(f.name)

# The manifest lists what docs/data actually holds, not just what this run
# copied. Context records are gitignored under results/ but committed under
# docs/data/, so a checkout without them must not drop them from the manifest —
# that silently breaks every drill-down on the site.
SKIP = {"registry.json", "manifest.json"}
served = sorted(f.name for f in out.glob("*.json") if f.name not in SKIP)
(out / "manifest.json").write_text(json.dumps(served, indent=2))
total = sum((out / name).stat().st_size for name in served)
print(
    f"wrote registry.json + manifest.json; copied {len(copied)} result files,"
    f" serving {len(served)} ({total / 1e6:.1f} MB)"
)
