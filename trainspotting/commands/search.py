"""`search`: find a string anywhere in the sampled examples, by side."""

import re
import sys

from .. import context, extract, hf, registry, search
from ..paths import RESULTS
from ..stats import wilson as _wilson
from .common import _pattern_slug, _print_match_rate, _select_stages, _stamp, _write_json


def cmd_search(args):
    """Search the whole training example — prompt and response — for a pattern.

    The other sampling layers read the prompt, which is half of the example and
    the half a behaviour is least likely to be in: a model claiming to be ChatGPT
    does it in an SFT response or a DPO chosen completion. So this searches every
    text field of the row and tags each hit with the side it landed on, and for
    DPO it reports which completion holds it — the same string chosen and
    rejected teach opposite things.

    No model is called. The default sample and seed are the ones the other layers
    use, so a hit is a row those runs already labeled and its full example is in
    the committed context file; a larger --sample is a wider net that no longer
    lines up with them.
    """
    try:
        pattern = re.compile(args.pattern, 0 if args.case_sensitive else re.IGNORECASE)
    except re.error as e:
        sys.exit(f"bad pattern {args.pattern!r}: {e}")
    slug = args.slug or _pattern_slug(args.pattern, args.case_sensitive)
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        # The shape of this stage's examples, which is what decides the sides a
        # hit can land on: a model stage's pipeline position, a dataset's
        # declared kind.
        kind = registry.stage_kind(s)
        # Before the draw, like every other sampling path: a lookup afterwards
        # could name a tree published while the paging ran.
        revision = hf.dataset_revision(s["hf_dataset"])
        print(f"searching {args.sample} sampled rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows_with_truncation(s["hf_dataset"], args.sample, seed=args.seed)
        moved = hf.dataset_revision(s["hf_dataset"])
        records, shortened, censored = [], 0, 0
        for row_index, row, truncated_cells in rows:
            # Only the cells this stage searches. The server shortens the
            # longest cell it finds, which on an RL row is a token-id array
            # nothing here reads.
            cut = search.truncated_columns(kind, truncated_cells)
            if cut:
                shortened += 1
            hits = search.search_row(row, kind, pattern, truncated=cut)
            if cut and not hits:
                # The server cut part of this row's text away, and what is left
                # does not match. That is not a non-match, it is a row this run
                # could not read.
                censored += 1
            if hits:
                # Rows carrying no prompt are searched like any other — a hit in
                # a response is a hit — so the join key can be empty.
                prompt = extract.extract_prompt(row, s["prompt_path"]) or ""
                records.append(
                    {
                        "key": prompt[: context.KEY_CHARS],
                        "row": row_index,
                        "sides": sorted({h["side"] for h in hits}),
                        # Which of this row's searched columns the server cut.
                        # A pair matching on one completion with the other cut
                        # is not an exclusive hit, whatever the visible text says.
                        **({"truncated": cut} if cut else {}),
                        "hits": hits,
                    }
                )
        sides = search.side_counts(records, kind)
        # Per side, matching rows whose text for that side was cut short: a
        # zero next to one of these is "not seen", not "not there".
        sides_unknown = search.unknown_sides(records, kind)
        k, n = len(records), len(rows)
        # A censored row is unknown, not a confirmed non-match, so it cannot
        # count as evidence against the string. The lower endpoint uses the
        # confirmed hits; the upper endpoint assumes every censored row could
        # have been one. With nothing censored the two agree and this is the
        # ordinary Wilson interval.
        lo, _ = _wilson(k, n)
        _, hi = _wilson(k + censored, n)
        payload = {
            "pattern": args.pattern,
            "case_sensitive": args.case_sensitive,
            "dataset": s["hf_dataset"],
            **_stamp(s["hf_dataset"], revision=revision),
            **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
            "stage": s["stage"],
            "sample": args.sample,
            "seed": args.seed,
            "scanned": n,
            "matched": k,
            "ci": [lo, hi],
            "sides": sides,
            "sides_unknown": sides_unknown,
            # Rows whose *searched* text the datasets-server shortened to fit
            # its response limit — a row cut only in a column this stage never
            # reads is unaffected and is not counted here. A
            # string past the cut is unfindable, so `matched` is a lower bound
            # rather than a count of the sample; `censored` is the part of that
            # the interval had to widen for — shortened rows with no visible
            # hit, which could have been hits in the text this run never saw.
            "truncated_rows": shortened,
            "censored": censored,
            "records": records,
        }
        if kind == "dpo":
            payload["pair_split"] = search.pair_split(records)
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.search-{slug}.json", payload)
        _print_match_rate(s["stage"], k, n, lo, hi, path)
        breakdown = ", ".join(
            f"{side} {count}" + (f" (+{sides_unknown[side]} unread)" if sides_unknown[side] else "")
            for side, count in sides.items()
        )
        if kind == "dpo":
            split = payload["pair_split"]
            breakdown += (
                f" (chosen only {split['chosen_only']},"
                f" rejected only {split['rejected_only']},"
                f" both {split['both']}"
                + (f", side unknown {split['unknown']}" if split["unknown"] else "")
                + ")"
            )
        print(f"  rows by side: {breakdown}", file=sys.stderr)
        if shortened:
            print(
                f"  note: {shortened} row(s) had searched text shortened by the datasets-server"
                f" — a hit past the cut is invisible here"
                + (
                    f"; {censored} of them show no hit at all, so the interval's"
                    " upper end assumes each could be one"
                    if censored
                    else ""
                ),
                file=sys.stderr,
            )
        for rec in records[: max(0, args.show)]:
            for hit in rec["hits"]:
                text = " ".join(hit["snippet"].split())
                print(f"  row {rec['row']} [{hit['side']}/{hit['role']}] {text}", file=sys.stderr)
