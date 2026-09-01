"""Re-capture the saved dataset rows that tests/test_extract.py checks against.

One row per (target, stage) in the registry — every model stage and every
standalone dataset — pulled from the datasets-server at a fixed offset so a
re-run of this script is a no-op unless the dataset changed.

The fixtures are what makes an upstream schema change a test failure instead of
a silently empty sample: the golden `prompt_*` fields below record what
`extract_prompt` pulled out of each row at capture time, so a registry
`prompt_path` that stops addressing anything shows up as a failing assertion.
Refresh them deliberately (`python scripts/capture_row_fixtures.py`) and read
the diff — a golden that changes on its own is the finding, not the noise.

Long strings are truncated and long scalar lists clipped (the RL rows carry
whole token-id arrays), so the goldens are computed after shrinking and stay
self-consistent. Keys, nesting, and value types are left exactly as served.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainspotting import extract, hf, registry  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "rows"
ROW_OFFSET = 0
MAX_STR = 4000
MAX_LIST = 8


def shrink(value):
    if isinstance(value, str):
        return value[:MAX_STR]
    if isinstance(value, dict):
        return {k: shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        # Scalar lists are payload (token ids, logprobs); lists of dicts are
        # schema (chat messages) and have to survive intact.
        if any(isinstance(v, (dict, list)) for v in value):
            return [shrink(v) for v in value]
        return value[:MAX_LIST]
    return value


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for target_name in registry.targets():
        target = registry.resolve(target_name)
        for stage in registry.post_training_stages(target):
            dataset = stage["hf_dataset"]
            j = hf._get(
                "rows",
                dataset=dataset,
                config="default",
                split="train",
                offset=ROW_OFFSET,
                length=1,
            )
            row = shrink(j["rows"][0]["row"])
            prompt = extract.extract_prompt(row, stage["prompt_path"])
            if not prompt:
                sys.exit(
                    f"{dataset}: prompt_path {stage['prompt_path']!r} extracted nothing"
                    f" from row {ROW_OFFSET} (columns: {', '.join(sorted(row))})"
                )
            path = FIXTURES / f"{target_name}.{stage['stage']}.json"
            path.write_text(
                json.dumps(
                    {
                        "target": target_name,
                        "stage": stage["stage"],
                        "dataset": dataset,
                        "prompt_path": stage["prompt_path"],
                        "row_offset": ROW_OFFSET,
                        "columns": sorted(row),
                        "prompt_chars": len(prompt),
                        "prompt_head": prompt[:120],
                        "prompt_tail": prompt[-60:],
                        "row": row,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"{path.name}: {len(prompt)} prompt chars, {len(row)} columns")


if __name__ == "__main__":
    main()
