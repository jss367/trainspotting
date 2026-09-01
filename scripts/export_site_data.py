"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/ask/context/pretrain runs:
    python3 scripts/export_site_data.py

Context and pretraining-document records are copied minified (they are bulk text
the site fetches on click); everything else is copied verbatim so its diffs stay
readable. Each document sample also gets a `.corpus.json` summary — the same file
without its records — so the site can draw the card without pulling megabytes.
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import budget, languages, paths, registry, rewards  # noqa: E402

# Bulk text the site fetches on demand: full training examples behind a prompt,
# and sampled pretraining documents. Both are regenerable caches of upstream
# data, so both are gitignored under results/ and committed under docs/data/.
BULK = (".context.json", ".docs.json")

# Derived here rather than copied from results/, so a checkout without the
# gitignored sources produces none of these — but the committed copies are still
# what the site reads.
#
# A budget is pure arithmetic over the committed ask runs, so rebuilding it is
# how it stays true: copying a committed one would let it survive a re-asked
# question and keep reporting the old total with nothing on the page to say so.
DERIVED = (".corpus.json",)

# Everything the manifest must keep listing even when this run did not produce
# it. Getting this set wrong is silent: the files stay on disk, drop out of the
# manifest, and the site stops asking for them.
COMMITTED = BULK + DERIVED

out = ROOT / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)

# Every dataset the registry samples must appear in the README's stage table.
# This is the second hand-maintained fact to drift from the registry in review —
# the table said the 7B models sample the -1125 mixes long after they moved to
# -1025 — so the command that rebuilds the site checks it rather than trusting
# the next person to remember.
readme = (ROOT / "README.md").read_text()
missing = sorted(
    {
        s["sample_dataset"]
        for m in registry.MODELS.values()
        for s in registry.pretrain_stages(m)
    }
    - {d for d in readme.split("`") if d.startswith("allenai/")}
)
if missing:
    sys.exit(
        "README's corpus table is out of date with the registry; missing: "
        + ", ".join(missing)
    )


# Models and standalone datasets in one map, each already resolved to the
# {is_model, hf_model, stages} shape the page reads — so a dataset renders
# through the same code as a model, and the cross-model compare has the flag it
# needs to leave datasets out of an axis they don't belong on.
(out / "registry.json").write_text(
    json.dumps({name: registry.resolve(name) for name in registry.targets()}, indent=2)
)
# The site labels language bars with these; keeping the map here means the CLI
# and the page can never disagree about what "gu" is called.
(out / "language-names.json").write_text(json.dumps(languages.NAMES, indent=2))
# The mix→verifier table, so the site can explain each RL row's reward from its
# kind (instead of trusting text baked into old context exports) and roll a
# stage's dataset_source counts up into exact reward-type shares.
(out / "reward-kinds.json").write_text(json.dumps(rewards.site_export(), indent=2))

copied, total = [], 0
for f in sorted((ROOT / "results").glob("*.json")):
    # A budget is derived below from the committed ask runs, never copied. A
    # local `budget --json` leaves one in results/ — gitignored, so it is not
    # part of what this repo publishes — and copying it here put its name in
    # `copied` a second time when the derive loop wrote the real one, so the
    # manifest listed it twice.
    if ".budget-" in f.name:
        continue
    if f.name.endswith(BULK):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    total += (out / f.name).stat().st_size
    copied.append(f.name)

    # A document sample is a few megabytes, nearly all of it the documents
    # themselves, but the card above them needs only the counts. Split the
    # summary out so opening the model tab costs kilobytes and the documents
    # load when someone actually asks to read them.
    if f.name.endswith(".docs.json"):
        d = json.loads(f.read_text())
        summary = {k: v for k, v in d.items() if k != "records"}
        summary["records"] = len(d["records"])
        name = f.name.replace(".docs.json", ".corpus.json")
        (out / name).write_text(json.dumps(summary, separators=(",", ":")))
        total += (out / name).stat().st_size
        copied.append(name)

# Ordinary result files that results/ no longer has. The copy loop only ever
# writes, so deleting a run — or moving it to another slug — used to leave its
# old copy sitting here: dropped from the manifest, so the site stopped drawing
# its card, but still the first thing `paths.find` returns to anything reading a
# run back. The budgets derived below would then be rolled up over a
# measurement whose card is no longer on the page.
#
# Bulk and derived files are exempt: those are gitignored under results/, so a
# fresh clone has none of them there and this is their only copy.
stale = [
    f
    for f in sorted(out.glob("*.json"))
    if not f.name.endswith(COMMITTED)
    and f.name not in set(copied)
    and f.name not in {"registry.json", "language-names.json", "reward-kinds.json", "manifest.json"}
    and not f.name.startswith(tuple(f"{n}.budget-" for n in registry.targets()))
]
for f in stale:
    f.unlink()
if stale:
    print(f"removed {len(stale)} stale site file(s) results/ no longer has: "
          + ", ".join(f.name for f in stale))

# Every question asked of a target gets its budget rolled up here, from whatever
# stages were committed. A question asked of only the post-training half still
# gets a card — with the corpora shown as unmeasured, which is the finding.
# Budgets are rebuilt from scratch every run, so drop the previous set first:
# they are derived, not copied (the loop above skips them), so nothing here is
# lost. A slug that no longer has any ask run loses its budget file too, and the
# loop below only writes the ones it can still build.
for f in out.glob("*.budget-*.json"):
    f.unlink()
for name in registry.targets():
    for slug in paths.runs(name, "ask"):
        est = budget.estimate(name, slug)
        if not any(st.get("measured") for st in est["stages"]):
            continue
        out_name = f"{name}.budget-{slug}.json"
        (out / out_name).write_text(json.dumps(est, separators=(",", ":")))
        total += (out / out_name).stat().st_size
        copied.append(out_name)

# The manifest is what this run copied, plus the bulk files already sitting in
# docs/data. Those are gitignored under results/, so a fresh checkout copies none
# of them, and listing only the copied files would drop every committed context
# and pretraining sample — which the site reads as "no drill-downs" and "no
# pretraining sample". Everything else still comes from results/ alone, so
# deleting a labels or ask run there drops it from the site as before.
kept = sorted(
    f.name
    for f in out.glob("*.json")
    if f.name.endswith(COMMITTED) and f.name not in set(copied)
)
# The same drift, one class over: the README quotes measured figures from the
# committed samples, and re-sampling moves them. Checking dataset ids alone let
# "12 of ~390 draws" survive two re-samples that made it "7 of ~318".
#
# This runs after the copy loop deliberately. Reading docs/data/ beforehand
# validates the *previous* export, so a re-sample that moved the figure would
# pass against the old copy and then ship the sample that contradicts it — and
# updating the README first would fail against output being replaced in the same
# command. Checking afterwards checks what was actually written.
flat_readme = readme.replace("\n", " ")
for docs in out.glob("*.docs.json"):
    d = json.loads(docs.read_text())
    # Keyed on model as well as stage. Every model samples its own corpora, so a
    # stage-only match would test a 32B sample against the 7B's figure and fail
    # the export the first time anyone samples a second model.
    model = docs.name.rsplit(f".{d['stage']}.docs.json", 1)[0]
    claimed = re.search(
        rf"`{re.escape(model)}`\s+{re.escape(d['stage'])}\s*\|\s*(\d+)\s*\|",
        flat_readme,
    )
    if claimed and int(claimed.group(1)) != d["short_draws"]:
        sys.exit(
            f"README says {claimed.group(1)} short draws for {model} {d['stage']}, "
            f"but the exported sample records {d['short_draws']}"
        )

(out / "manifest.json").write_text(json.dumps(sorted(copied + kept), indent=2))
print(
    f"wrote registry.json + language-names.json + manifest.json and {len(copied)} result files ({total / 1e6:.1f} MB)"
    + (f"; manifest also lists {len(kept)} committed file(s) this run did not build" if kept else "")
)
