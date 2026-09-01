"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/ask/context/pretrain runs:
    python3 scripts/export_site_data.py

Context and pretraining-document records are copied minified (they are bulk text
the site fetches on click); everything else is copied verbatim so its diffs stay
readable. The two bulk kinds also get a small summary derived from them — a
`.corpus.json` per document sample and a `.profile.json` per context run — so the
site can draw a card without pulling megabytes.
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import derive, languages, registry, rewards  # noqa: E402

# Bulk text the site fetches on demand: full training examples behind a prompt,
# and sampled pretraining documents. Both are regenerable caches of upstream
# data, so both are gitignored under results/ and committed under docs/data/.
# Getting this set wrong is silent: the files stay on disk, drop out of the
# manifest, and the site stops asking for them.
BULK = (".context.json", ".docs.json")

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
    if f.name.endswith(BULK):
        (out / f.name).write_text(json.dumps(json.loads(f.read_text()), separators=(",", ":")))
    else:
        shutil.copy(f, out / f.name)
    total += (out / f.name).stat().st_size
    copied.append(f.name)

# Summaries of the bulk files, derived from docs/data rather than results/.
# Both bulk kinds are gitignored under results/ and committed here, so a fresh
# checkout has the samples but not their sources — deriving from what was just
# written means the summaries always describe the sample the site actually
# serves, instead of going stale the moment a re-sample lands from another
# machine.
derived = []


def write_derived(name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(payload, separators=(",", ":")))
    derived.append(name)


# A document sample is a few megabytes, nearly all of it the documents
# themselves, but the card above them needs only the counts and the lengths.
# Split the summary out so opening the model tab costs kilobytes and the
# documents load when someone actually asks to read them.
for docs_file in sorted(out.glob("*.docs.json")):
    d = json.loads(docs_file.read_text())
    summary = {k: v for k, v in d.items() if k != "records"}
    summary["records"] = len(d["records"])
    # How long a pretraining document is, on the same bins as a training example.
    summary["lengths"] = derive.corpus_lengths(d)
    write_derived(docs_file.name.replace(".docs.json", ".corpus.json"), summary)

# What a stage's examples are made of: how long they are, how much of that the
# model is fit to, and which metadata column each sampled row carries — the
# join that lets the site cross a taxonomy label against where in the mix the
# prompt came from without downloading the whole context file.
for ctx_file in sorted(out.glob("*.context.json")):
    d = json.loads(ctx_file.read_text())
    target = ctx_file.name.rsplit(f".{d['stage']}.context.json", 1)[0]
    src_file = out / f"{target}.sources.json"
    # The exact row count comes from the sources layer, which counts rather than
    # samples. Without it the shape of an example is still measurable and the
    # token total is not, so the profile simply carries no estimate.
    src = json.loads(src_file.read_text()).get(d["stage"], {}) if src_file.exists() else {}
    # A token total multiplies a mean measured over one tree by a row count
    # counted over another. These dataset ids move — Ai2 has republished these
    # mixes — and both layers stamp the revision they read for exactly that
    # reason, so the two stamps have to agree before the multiplication means
    # anything. A mismatch is a mixed-tree number and gets no estimate at all;
    # where either side predates revision recording the pairing is merely
    # unproven, and the profile says so rather than dropping a figure that is
    # probably fine.
    revisions = {"context": d.get("revision"), "sources": src.get("revision")}
    if revisions["context"] and revisions["sources"]:
        agree = revisions["context"] == revisions["sources"]
    else:
        agree = None
    profile = derive.stage_profile(d, src.get("total") if agree is not False else None)
    profile["revisions"] = {**revisions, "agree": agree}
    if agree is False:
        print(
            f"  ! {ctx_file.name}: sampled at {revisions['context'][:7]} but counted at "
            f"{revisions['sources'][:7]} — no token estimate, re-run one of them"
        )
    write_derived(ctx_file.name.replace(".context.json", ".profile.json"), profile)

total += sum((out / name).stat().st_size for name in derived)

# The manifest is what this run copied, plus the bulk files already sitting in
# docs/data. Those are gitignored under results/, so a fresh checkout copies none
# of them, and listing only the copied files would drop every committed context
# and pretraining sample — which the site reads as "no drill-downs" and "no
# pretraining sample". Everything else still comes from results/ alone, so
# deleting a labels or ask run there drops it from the site as before.
kept = sorted(
    f.name for f in out.glob("*.json") if f.name.endswith(BULK) and f.name not in set(copied)
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

(out / "manifest.json").write_text(json.dumps(sorted(set(copied + kept + derived)), indent=2))
print(
    f"wrote registry.json + language-names.json + manifest.json, {len(copied)} result files "
    f"and {len(derived)} derived summaries ({total / 1e6:.1f} MB)"
    + (f"; manifest also lists {len(kept)} committed file(s) this run did not build" if kept else "")
)
