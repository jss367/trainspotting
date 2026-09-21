"""Capture published OLMo stage evaluations with pinned source provenance.

These use the release's evaluation protocol, separately from our fixed-prompt
measurements. Run explicitly; the offline site export never fetches the network.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trainspotting import paths  # noqa: E402

SOURCES = {
    "olmo-3-7b-instruct": ("allenai/Olmo-3-7B-Instruct-SFT", 2,
                          ["Olmo 3 Instruct 7B SFT", "Olmo 3 Instruct 7B DPO", "Olmo3 Instruct 7B"]),
    "olmo-3-7b-think": ("allenai/Olmo-3-7B-Think-SFT", 2,
                       ["Olmo 3 Think 7B SFT", "Olmo 3 Think 7B DPO", "Olmo 3 Think 7B"]),
    "olmo-3-32b-think": ("allenai/Olmo-3-32B-Think-SFT", 1,
                        ["Olmo 3 Think 32B SFT", "Olmo 3 Think 32B DPO", "Olmo 3 Think 32B"]),
    "olmo-3.1-32b-instruct": ("allenai/Olmo-3.1-32B-Instruct-SFT", 1,
                            ["Olmo 3.1 32B Instruct SFT", "Olmo 3.1 32B Instruct DPO", "Olmo 3.1 32B Instruct"]),
}
CATEGORIES = {"Math": "Mathematics", "IF": "Instruction following", "QA": "Question answering",
              "Chat": "Conversation", "Coding": "Code", "Knowledge & QA": "Knowledge and question answering"}


def cells(line):
    return [c.strip().replace("**", "") for c in line.split("|")[1:-1]]


def parse_table(text, offset=2, expected=None):
    expected = expected or SOURCES['olmo-3-7b-instruct'][2]
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if line.startswith("|") and cells(line)[offset:offset + 3] == expected), None)
    if header is None:
        raise ValueError("Source evaluation columns changed; review before importing")
    rows = []
    category = None
    for line in lines[header + 2:]:
        if not line.startswith("|"):
            break
        row = cells(line)
        if not any(row[offset:offset + 3]):
            category = row[0]
            continue
        scores = [float(v) for v in row[offset:offset + 3]]
        if len(scores) != 3 or any(not 0 <= v <= 100 for v in scores):
            raise ValueError("Expected three scores on a 0–100 scale")
        if offset == 2:
            category = row[0] or category
            name = row[1] or category
        else:
            name = row[0]
            if line.split("|")[1].strip().startswith("**"):
                category = name
        rows.append({"name": name, "category": CATEGORIES.get(category, category),
                     "scores": dict(zip(("sft", "dpo", "rlvr"), scores))})
    if not rows:
        raise ValueError("No evaluations found")
    return rows


def main():
    for target, (repo, offset, expected) in SOURCES.items():
        info = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=60)
        info.raise_for_status()
        revision = info.json()["sha"]
        response = requests.get(f"https://huggingface.co/{repo}/raw/{revision}/README.md", timeout=60)
        response.raise_for_status()
        result = {"target": target, "kind": "published_evaluations",
                  "source": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
                  "source_revision": revision, "source_sha256": hashlib.sha256(response.content).hexdigest(),
                  "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "note": "Publisher-reported scores from one stage-comparison table. Evaluation settings are those of the release; these are not our fixed-prompt probes. No base-model scores are supplied in this table.",
                  "benchmarks": parse_table(response.text, offset, expected)}
        paths.RESULTS.mkdir(exist_ok=True)
        output = paths.RESULTS / f"{target}.evaluations.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {len(result['benchmarks'])} benchmark rows to {output}")


if __name__ == "__main__":
    main()
