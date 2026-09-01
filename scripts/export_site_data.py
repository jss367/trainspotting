"""Regenerate docs/data/ from the registry and committed results.

Run after adding a model or committing new classify/ask/context/pretrain runs:
    python3 scripts/export_site_data.py

Context and pretraining-document records are copied minified (they are bulk text
the site fetches on click); everything else is copied verbatim so its diffs stay
readable. The two bulk kinds also get a small summary derived from them — a
`.corpus.json` per document sample and a `.profile.json` per context run — so the
site can draw a card without pulling megabytes. The same bulk files are indexed
by three-character run into `search-index.json`, which is how the page's search
box avoids downloading them all.
"""

import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainspotting import (  # noqa: E402
    budget,
    casestudy,
    derive,
    languages,
    paths,
    registry,
    rewards,
    searchindex,
)

# Bulk text the site fetches on demand: full training examples behind a prompt,
# and sampled pretraining documents. Both are regenerable caches of upstream
# data, so both are gitignored under results/ and committed under docs/data/.
# Getting this set wrong is silent: the files stay on disk, drop out of the
# manifest, and the site stops asking for them.
BULK = (".context.json", ".docs.json")

# Result files with no reader on the page. The site draws a card per labels/ask/
# languages run and ignores anything it does not recognise, so exporting these
# would only add unread weight to what the page ships. Drop a prefix from here
# when the page learns to render that kind of run.
#
# `.search-` is here for the same reason as `.grep-` and is easy to miss: the
# page has a search box, but it searches the committed samples directly and
# never reads a `.search-` result file. Exporting one ships bytes nobody
# fetches.
UNRENDERED = (".grep-", ".search-")

out = ROOT / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)

# Every dataset the registry samples must appear in the README's stage table.
# This is the second hand-maintained fact to drift from the registry in review —
# the table said the 7B models sample the -1125 mixes long after they moved to
# -1025 — so the command that rebuilds the site checks it rather than trusting
# the next person to remember.
readme = (ROOT / "README.md").read_text()
# Any `owner/name` in a backtick span counts. An earlier version required the
# owner to be `allenai/`, which would have reported EleutherAI's Pile corpus
# missing however many times the README named it.
CORPUS_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
missing = sorted(
    {
        s["sample_dataset"]
        for m in registry.MODELS.values()
        for s in registry.pretrain_stages(m)
    }
    - {d for d in readme.split("`") if CORPUS_ID.match(d)}
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
    #
    # UNRENDERED is the same idea for runs the page has no card for at all.
    if ".budget-" in f.name or any(marker in f.name for marker in UNRENDERED):
        continue
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

# Significant digits kept for a derived float. Everything under this heading is
# computed — a mean, a standard error, a design effect, an estimated token
# total — and the last few digits of a float are an artifact of the order the
# arithmetic happened in and the libm that did it, not of the data.
#
# These files are committed, so those digits are a diff. CI caught it the first
# time it ran: the same export on the same tree produced
# `deff: 3.4905799081030855` here and `...846` on the runner, a one-unit
# difference in the seventeenth digit, and the check that the committed export
# matches the tree failed on it. Rounding at the boundary makes a derived file a
# function of the sample rather than of the machine.
#
# Twelve is far past anything meaningful here — the widest quantity is a token
# estimate around 1e9, whose interval is ±60% — and four digits clear of where
# the noise lives.
FLOAT_DIGITS = 12


def stable(value):
    """`value` with every float rounded to `FLOAT_DIGITS` significant digits."""
    if isinstance(value, float):
        if not math.isfinite(value) or value == 0:
            return value
        return round(value, FLOAT_DIGITS - 1 - math.floor(math.log10(abs(value))))
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def write_derived(name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(stable(payload), separators=(",", ":")))
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
    # A run that outlasted a republish records where the dataset went, and its
    # rows may come from either tree — so a matching starting revision does not
    # make it comparable to anything. Both commands emit the field for exactly
    # this reason; a run that carries it can only be paired by re-running it.
    revisions = {
        "context": d.get("revision"),
        "sources": src.get("revision"),
        "moved": [
            layer
            for layer, run in (("context", d), ("sources", src))
            if run.get("revision_moved_to")
        ],
    }
    if revisions["moved"]:
        agree = False
    elif revisions["context"] and revisions["sources"]:
        agree = revisions["context"] == revisions["sources"]
    else:
        agree = None
    profile = derive.stage_profile(d, src.get("total") if agree is not False else None)
    profile["revisions"] = {**revisions, "agree": agree}
    if agree is False:
        why = (
            f"the {' and '.join(revisions['moved'])} run spans two dataset trees"
            if revisions["moved"]
            else f"sampled at {revisions['context'][:7]} but counted at {revisions['sources'][:7]}"
        )
        print(f"  ! {ctx_file.name}: {why} — no token estimate, re-run it")
    write_derived(ctx_file.name.replace(".context.json", ".profile.json"), profile)

total += sum((out / name).stat().st_size for name in derived)

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
    # `derived` rather than a suffix list: this branch derives a `.profile.json`
    # per context run as well as a `.corpus.json` per document sample, and a
    # sweep that knew only the second deleted every profile it had just written.
    if not (f.name.endswith(BULK) or f.name in set(derived))
    and f.name not in set(copied)
    # The files this script writes itself rather than copying from results/.
    # `search-index.json` is built below from the bulk files, so results/ never
    # holds one — a sweep that only knew the hand-written names deleted it every
    # run and reported it as stale, on the way to rebuilding it.
    and f.name not in {"registry.json", "language-names.json", "reward-kinds.json",
                       "manifest.json", "search-index.json"}
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
        # Derived the same way the profiles are, and rounded the same way: a
        # budget is stage sizes times measured rates, so its floats carry the
        # same machine-dependent tail.
        (out / out_name).write_text(json.dumps(stable(est), separators=(",", ":")))
        total += (out / out_name).stat().st_size
        copied.append(out_name)

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

# The third class of the same drift, and the one most likely to happen: a lookup
# study runs against a live index with no revision to pin, so every re-run moves
# its numbers and the README paragraph quoting them goes stale silently.
#
# The date used to stand in for the figures, and it cannot: two runs on the same
# day return different documents while the date holds still, and a date can be
# bumped without reading a single number under it. So the README carries a
# fingerprint of the figures themselves — see `casestudy.quoted_figures` — and
# any of them moving fails this until someone re-reads the paragraph and pastes
# in the new one. That is the whole point of it: the fingerprint records that a
# person checked the prose against this data, not that a file was regenerated.
#
# Only the studies this run copied. A study is retired by deleting its results
# file and its README section, and the export does not delete what it no longer
# writes — so globbing docs/data/ would hold the retired copy's figures against
# a README that correctly no longer mentions them, and fail every export until
# someone thought to delete a file the manifest already ignores.
for name in sorted(n for n in copied if n.startswith("case-study.")):
    study = out / name
    d = json.loads(study.read_text())
    fp = casestudy.fingerprint(d)
    marker = f"<!-- figures: {study.stem} {fp} -->"
    if marker not in readme:
        sys.exit(
            f"{study.name}'s quoted figures have moved (run on {d['run_on']}).\n"
            f"Re-read the paragraph in the README against the new numbers:\n"
            + "\n".join(f"    {k}: {v}" for k, v in casestudy.quoted_figures(d).items())
            + f"\nThen put this line under it:\n    {marker}"
        )
    # In this study's own section, not anywhere in the file. Two studies run on
    # one day would otherwise satisfy each other's check, and the second could
    # carry no date at all — or a stale one — while the export passed. The
    # section is the text after the marker line the fingerprint check just
    # confirmed, up to the next one, which is what makes the marker do double
    # duty as a section boundary.
    section = readme.split(marker)[0].rsplit("<!-- figures: ", 1)[-1]
    if f"As of {d['run_on']}" not in section.replace("\n", " "):
        sys.exit(
            f"{study.name} was run on {d['run_on']}, which its own README section "
            f"does not claim ('As of {d['run_on']}'). A date elsewhere in the file "
            "belongs to another study."
        )

# Everything the search box can read, indexed by trigram. Built from docs/data
# rather than results/ so it covers the committed bulk files a fresh checkout
# never rebuilt (the same reason `kept` exists), and built last so it indexes
# exactly what this run wrote.
searchable = sorted(out.glob("*.context.json")) + sorted(out.glob("*.docs.json"))
index = searchindex.build_from_paths(searchable)
# ensure_ascii would triple the size of every non-Latin trigram, and the page
# parses the file as UTF-8 either way.
(out / "search-index.json").write_text(
    json.dumps(index, separators=(",", ":"), ensure_ascii=False)
)
copied.append("search-index.json")
index_mb = (out / "search-index.json").stat().st_size / 1e6
total += (out / "search-index.json").stat().st_size

(out / "manifest.json").write_text(json.dumps(sorted(set(copied + kept + derived)), indent=2))
print(
    f"indexed {len(index['grams']):,} trigrams across {len(searchable)} sampled files "
    f"for search ({index_mb:.1f} MB)"
)
print(
    f"wrote registry.json + language-names.json + manifest.json, {len(copied)} result files "
    f"and {len(derived)} derived summaries ({total / 1e6:.1f} MB)"
    + (f"; manifest also lists {len(kept)} committed file(s) this run did not build" if kept else "")
)
