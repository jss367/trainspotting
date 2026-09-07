"""`context`: store the full training example behind each sampled prompt."""

import sys

from .. import context, extract, hf, registry
from ..paths import RESULTS
from .common import _select_stages, _stamp, _write_json


def cmd_context(args):
    """Re-fetch the sampled rows and store the full training example behind each prompt.

    Sampling is deterministic in (sample, seed), so the same defaults as a
    classify/ask run pull exactly the rows those runs labeled. No model is
    called here — this is the deterministic half of the drill-down.
    """
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        revision = hf.dataset_revision(s["hf_dataset"])
        print(f"re-fetching {args.sample} sampled rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows_with_index(s["hf_dataset"], args.sample, seed=args.seed)
        # Thirty-odd paged requests, so the same republish window the labeling
        # path checks for applies here — smaller, but not absent, and these
        # records are what the site shows when someone clicks through to a
        # training example.
        moved = hf.dataset_revision(s["hf_dataset"])
        records = []
        for row_index, row in rows:
            prompt = extract.extract_prompt(row, s["prompt_path"])
            if prompt:
                records.append(
                    context.build(
                        row, registry.stage_kind(s), prompt, row_index, s.get("source_columns") or ()
                    )
                )
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.context.json",
            {
                "dataset": s["hf_dataset"],
                **_stamp(s["hf_dataset"], revision=revision),
                **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
                "stage": s["stage"],
                "sample": args.sample,
                "seed": args.seed,
                "records": records,
            },
        )
        print(f"{s['stage']}: {len(records)} records -> {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
