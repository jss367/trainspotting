"""`stance`: which way each stored example pushes on a question."""

import sys

from .. import budget, classify, extract, registry, stance
from ..paths import RESULTS
from ..stats import wilson as _wilson
from .common import _select_stages, _slug, _stamp, _unlabeled_note, _write_json


def _stale_context(data: dict, dataset: str) -> str | None:
    """Why these stored examples cannot stand for `dataset`, or None.

    The same two questions `budget._size_post_training` asks before sizing a
    stage from them: are they from this dataset, and were they drawn from one
    tree. Judging is the expensive caller, so it asks first.
    """
    if data.get("dataset") and data["dataset"] != dataset:
        return f"the stored examples are from {data['dataset']} but this stage names {dataset}"
    if data.get("revision_moved_to"):
        return "the stored examples straddled a republish while they were drawn"
    return None


def cmd_stance(args):
    """Judge which way each stored training example pushes on a question.

    Reads the committed context records rather than re-sampling: `context`
    already holds the whole example behind every sampled prompt, so this lands
    on exactly the rows an `ask` or `classify` run labeled and costs nothing but
    the API calls. A stage with no context run yet says so instead of quietly
    contributing nothing to the total.
    """
    slug = args.slug or _slug(args.question)
    print(f"question: {args.question}\n", file=sys.stderr)
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        kind = registry.stage_kind(s)
        if kind not in ("sft", "dpo", "rlvr"):
            # A chat log has no direction to read. Judging one would report a
            # training signal the data does not carry — the same reason
            # `context` marks no turn in it as a target.
            print(
                f"{s['stage']}: skipped — {kind} examples were not trained on,"
                " so there is no direction to judge",
                file=sys.stderr,
            )
            continue
        data = budget.load(f"{args.target}.{s['stage']}.context.json")
        if not data:
            print(
                f"{s['stage']}: no stored examples"
                f" (`trainspotting context {args.target} --stage {s['stage']}`)",
                file=sys.stderr,
            )
            continue
        # Checked before a single API call, not after. The exporter keeps bulk
        # context files that `results/` no longer has — they are gitignored
        # there, so docs/data is their only copy — which is right for reading an
        # old run back and wrong for judging a new one: a stage repointed at
        # another dataset would leave those examples sitting here, and this
        # would score them and file the result under the current stage.
        stale = _stale_context(data, s["hf_dataset"])
        if stale:
            print(
                f"{s['stage']}: skipped — {stale}; re-run"
                f" `trainspotting context {args.target} --stage {s['stage']}`",
                file=sys.stderr,
            )
            continue
        records = data["records"]
        print(
            f"judging {len(records)} whole {kind} examples with {args.classifier} ...",
            file=sys.stderr,
        )
        labels, reasons = classify.classify_prompts(
            [stance.render(r) for r in records],
            model=args.classifier,
            question=args.question,
            system=stance.SYSTEM,
            valid=stance.STANCES,
            max_chars=stance.MAX_EXAMPLE,
            batch_size=stance.BATCH,
        )
        out = [
            {
                "row": r.get("row"),
                "prompt": extract.clip((r.get("prompt_full") or {}).get("text", "")),
                "stance": lab,
            }
            for r, lab in zip(records, labels)
            if lab
        ]
        counts = {k: sum(1 for r in out if r["stance"] == k) for k in stance.STANCES}
        n = len(out)
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.stance-{slug}.json",
            {
                "question": args.question,
                "slug": slug,
                "dataset": data["dataset"],
                # The example is the one the context run stored, so the revision
                # is that run's — not whatever `main` points at now.
                **_stamp(data["dataset"], revision=data.get("revision")),
                "stage": s["stage"],
                "kind": kind,
                "sample": data.get("sample"),
                "seed": data.get("seed"),
                "classifier": args.classifier,
                "system_sha": classify.system_id(
                    classify.build_system(args.question, stance.SYSTEM)
                ),
                "judged_chars": stance.MAX_EXAMPLE,
                "unlabeled": sum(1 for label in labels if label is None),
                "unlabeled_reasons": reasons,
                "counts": counts,
                "net": stance.net(counts),
                "records": out,
            },
        )
        toward, away = counts["toward"], counts["away"]
        lo, hi = _wilson(toward, n)
        print(
            f"{s['stage']}: toward {toward}/{n} = {toward / n * 100 if n else 0:.1f}%"
            f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%), away {away}, net {toward - away}"
            f" -> {path}{_unlabeled_note(labels, reasons)}",
            file=sys.stderr,
        )
