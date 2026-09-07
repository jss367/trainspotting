"""`sources`: exact mix composition from the datasets-server's column statistics."""

import sys

from .. import hf, registry
from ..paths import RESULTS
from .common import _select_stages, _stamp, _write_json


def cmd_sources(args):
    """The source-label breakdown of each post-training mix.

    Selected through `_select_stages` like every other command that reads a
    post-training stage, and for the reason that helper exists: a base-only
    target such as Pythia has no such stages, and iterating an empty list here
    exited 0 having printed nothing — and with --json wrote an audit file
    containing `{}`, which the site would serve as a measured empty breakdown.
    Failing is the honest answer, and `report` is where a base model's "there is
    no post-training" is stated deliberately.
    """
    out = {}
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        revision = hf.dataset_revision(s["hf_dataset"])
        freqs, counted, partial = hf.column_frequencies(
            s["hf_dataset"], s["source_columns"]
        )
        total = hf.num_rows(s["hf_dataset"])
        # Shares are over the rows the stats API actually scanned, which on a
        # big dataset is not all of them.
        counted = counted or total
        # /statistics and /info are two requests, and this layer is the one that
        # calls its numbers exact — so a republish between them would leave
        # frequencies and a row count describing different trees.
        moved = hf.dataset_revision(s["hf_dataset"])
        # One hub request per repo-shaped label, so only pay for it when the
        # result is being written out — the printed table has nowhere to put a URL.
        links = {}
        if args.json:
            for freq in freqs.values():
                for value in freq:
                    if value not in links:
                        url = hf.dataset_url(value)
                        if url:
                            links[value] = url
        out[s["stage"]] = {
            "dataset": s["hf_dataset"],
            **_stamp(s["hf_dataset"], revision=revision),
            **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
            "total": total,
            "counted": counted,
            "partial": partial,
            "columns": freqs,
            "links": links,
        }
        print(f"\n## {s['stage']} — {s['hf_dataset']} ({total:,} examples)")
        if partial:
            print(
                f"   shares are over the {counted:,} rows"
                f" ({counted / total * 100:.0f}%) HuggingFace's stats API scanned"
            )
        for col, freq in freqs.items():
            print(f"\n{col}:")
            for value, count in freq.items():
                print(f"  {count / counted * 100:5.1f}%  {value} ({count:,})")
    if args.json:
        path = _write_json(RESULTS / f"{args.target}.sources.json", out)
        print(f"\nwrote {path}", file=sys.stderr)
