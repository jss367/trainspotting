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

from trainspotting import languages, registry  # noqa: E402

out = ROOT / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)

(out / "registry.json").write_text(json.dumps(registry.MODELS, indent=2))
# The site labels bars with these; keeping the map here means the CLI and the
# page can never disagree about what "gu" is called.
(out / "language-names.json").write_text(json.dumps(languages.NAMES, indent=2))

copied = []
for f in sorted((ROOT / "results").glob("*.json")):
    if f.name.endswith(".context.json"):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    copied.append(f.name)

# The manifest lists what is published, not what was just copied. results/ holds
# the context files only on the machine that generated them (they are gitignored
# bulk text), so building the manifest from results/ on any other checkout would
# silently drop every context file and kill the drill-down.
INDEX = {"registry.json", "manifest.json", "language-names.json"}
published = sorted(f.name for f in out.glob("*.json") if f.name not in INDEX)
total = sum((out / f).stat().st_size for f in published)
(out / "manifest.json").write_text(json.dumps(published, indent=2))
print(
    f"wrote registry.json + language-names.json + manifest.json; "
    f"copied {len(copied)} from results/, {len(published)} published ({total / 1e6:.1f} MB)"
)
