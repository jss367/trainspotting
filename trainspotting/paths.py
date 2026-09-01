"""Where results live, and how to read one back.

Two directories hold the same kinds of file for different reasons. `results/`
is what a run writes; `docs/data/` is the committed copy the site serves, and
for the bulk artifacts (context records, sampled corpus documents) it is the
*only* copy in a fresh clone, because those are gitignored regenerable caches
of upstream rows.

Anything that reads a previous run back — rather than computing one — has to
look in both, or it tells someone who just cloned the repo that the samples
shipped with it do not exist.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# The committed half of the bulk artifacts: gitignored under results/, shipped
# here for the site, and so the only copy present in a fresh clone.
SITE_DATA = ROOT / "docs" / "data"


def find(name: str) -> Path | None:
    """A committed or freshly written result file by name, or None.

    `results/` wins: a run just finished is the newer answer, and the export to
    `docs/data/` is a separate step someone may not have taken yet.
    """
    for path in (RESULTS / name, SITE_DATA / name):
        if path.exists():
            return path
    return None


def runs(target: str, kind: str) -> dict[str, list[str]]:
    """`{slug: [stage, ...]}` for every committed `<target>.<stage>.<kind>-<slug>.json`.

    Both directories are searched and the stages merged, so a report reads the
    same set of runs the site does. Stages come back in no particular order;
    callers that print them walk the registry instead, which is the order the
    pipeline runs in.
    """
    out: dict[str, set[str]] = {}
    for directory in (RESULTS, SITE_DATA):
        for path in directory.glob(f"{target}.*.{kind}-*.json"):
            rest = path.name[len(target) + 1 : -len(".json")]
            stage, _, slug = rest.partition(f".{kind}-")
            if slug:
                out.setdefault(slug, set()).add(stage)
    return {slug: sorted(stages) for slug, stages in sorted(out.items())}
