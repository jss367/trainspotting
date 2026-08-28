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
# The site labels language bars with these; keeping the map here means the CLI
# and the page can never disagree about what "gu" is called.
(out / "language-names.json").write_text(json.dumps(languages.NAMES, indent=2))

copied, total = [], 0
for f in sorted((ROOT / "results").glob("*.json")):
    if f.name.endswith(".context.json"):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    total += (out / f.name).stat().st_size
    copied.append(f.name)

# The manifest is what this run copied, plus the context files already sitting
# in docs/data. results/*.context.json is gitignored (a regenerable cache of
# upstream rows), so a fresh checkout copies none of them, and listing only the
# copied files would drop every committed context file — which the site reads as
# "no drill-downs". Everything else still comes from results/ alone, so deleting
# a labels or ask run there drops it from the site as before.
kept = sorted(
    f.name for f in out.glob("*.context.json") if f.name not in set(copied)
)
(out / "manifest.json").write_text(json.dumps(sorted(copied + kept), indent=2))
print(
    f"wrote registry.json + language-names.json + manifest.json and {len(copied)} result files ({total / 1e6:.1f} MB)"
    + (f"; manifest also lists {len(kept)} committed context file(s)" if kept else "")
)
