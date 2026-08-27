import argparse
import json
import math
import re
import sys
from pathlib import Path

from . import classify, context, extract, hf, registry

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _fmt_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    return f"{n:,}"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cmd_facts(args):
    model = registry.get_model(args.model)
    print(f"# {args.model} ({model['hf_model']})\n")
    for s in model["stages"]:
        line = f"- {s['stage']:12s} {s['name']}"
        if s.get("tokens"):
            line += f" — {_fmt_tokens(s['tokens'])} tokens"
        if s.get("hf_dataset"):
            n = hf.num_rows(s["hf_dataset"])
            line += f" — {n:,} examples ({s['hf_dataset']})"
        print(line)
        if s.get("note"):
            print(f"    {s['note']}")


def cmd_sources(args):
    model = registry.get_model(args.model)
    out = {}
    for s in registry.post_training_stages(model):
        freqs = hf.column_frequencies(s["hf_dataset"], s["source_columns"])
        total = hf.num_rows(s["hf_dataset"])
        out[s["stage"]] = {"dataset": s["hf_dataset"], "total": total, "columns": freqs}
        print(f"\n## {s['stage']} — {s['hf_dataset']} ({total:,} examples)")
        for col, freq in freqs.items():
            print(f"\n{col}:")
            for value, count in freq.items():
                print(f"  {count / total * 100:5.1f}%  {value} ({count:,})")
    if args.json:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{args.model}.sources.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}", file=sys.stderr)


def cmd_classify(args):
    model = registry.get_model(args.model)
    stages = registry.post_training_stages(model)
    if args.stage:
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no post-training stage {args.stage!r} for {args.model}")
    RESULTS.mkdir(exist_ok=True)
    for s in stages:
        print(f"sampling {args.sample} rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows(s["hf_dataset"], args.sample, seed=args.seed)
        prompts = [extract.extract_prompt(r, s["prompt_path"]) for r in rows]
        keep = [(rows[i], p) for i, p in enumerate(prompts) if p]
        print(f"classifying {len(keep)} prompts with {args.classifier} ...", file=sys.stderr)
        labels = classify.classify_prompts([p for _, p in keep], model=args.classifier)
        records = [
            {"prompt": extract.clip(p), "label": label}
            for (_, p), label in zip(keep, labels)
        ]
        path = RESULTS / f"{args.model}.{s['stage']}.labels.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": s["hf_dataset"],
                    "sample": args.sample,
                    "seed": args.seed,
                    "classifier": args.classifier,
                    "records": records,
                },
                indent=2,
            )
        )
        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"{s['stage']}: {counts}  -> {path}", file=sys.stderr)


def cmd_ask(args):
    """Score sampled prompts from every post-training stage against a free-form question."""
    model = registry.get_model(args.model)
    slug = args.slug or re.sub(r"[^a-z0-9]+", "-", args.question.lower()).strip("-")[:60]
    RESULTS.mkdir(exist_ok=True)
    print(f"question: {args.question}\n", file=sys.stderr)
    for s in registry.post_training_stages(model):
        print(f"sampling {args.sample} rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows(s["hf_dataset"], args.sample, seed=args.seed)
        prompts = [extract.extract_prompt(r, s["prompt_path"]) for r in rows]
        keep = [(rows[i], p) for i, p in enumerate(prompts) if p]
        labels = classify.classify_prompts(
            [p for _, p in keep], model=args.classifier, question=args.question
        )
        records = [
            {"prompt": extract.clip(p), "match": lab == "yes"}
            for (_, p), lab in zip(keep, labels) if lab
        ]
        k, n = sum(r["match"] for r in records), len(records)
        lo, hi = _wilson(k, n)
        path = RESULTS / f"{args.model}.{s['stage']}.ask-{slug}.json"
        path.write_text(
            json.dumps(
                {
                    "question": args.question,
                    "dataset": s["hf_dataset"],
                    "sample": args.sample,
                    "seed": args.seed,
                    "classifier": args.classifier,
                    "records": records,
                },
                indent=2,
            )
        )
        print(
            f"{s['stage']}: {k}/{n} match = {k / n * 100 if n else 0:.1f}%"
            f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%) -> {path}",
            file=sys.stderr,
        )


def cmd_context(args):
    """Re-fetch the sampled rows and store the full training example behind each prompt.

    Sampling is deterministic in (sample, seed), so the same defaults as a
    classify/ask run pull exactly the rows those runs labeled. No model is
    called here — this is the deterministic half of the drill-down.
    """
    model = registry.get_model(args.model)
    stages = registry.post_training_stages(model)
    if args.stage:
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no post-training stage {args.stage!r} for {args.model}")
    RESULTS.mkdir(exist_ok=True)
    for s in stages:
        print(f"re-fetching {args.sample} sampled rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows_with_index(s["hf_dataset"], args.sample, seed=args.seed)
        records = []
        for row_index, row in rows:
            prompt = extract.extract_prompt(row, s["prompt_path"])
            if prompt:
                records.append(context.build(row, s["stage"], prompt, row_index))
        path = RESULTS / f"{args.model}.{s['stage']}.context.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": s["hf_dataset"],
                    "stage": s["stage"],
                    "sample": args.sample,
                    "seed": args.seed,
                    "records": records,
                },
                indent=2,
            )
        )
        print(f"{s['stage']}: {len(records)} records -> {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def cmd_report(args):
    model = registry.get_model(args.model)
    print(f"# Training-data audit: {args.model}\n")
    print("## Stage sizes\n")
    for s in model["stages"]:
        if s.get("tokens"):
            print(f"- {s['stage']}: {s['name']}, {_fmt_tokens(s['tokens'])} tokens")
        else:
            print(f"- {s['stage']}: {s['name']} ({s['hf_dataset']})")
    print("\n## HHH classification (sampled, LLM-labeled)\n")
    for s in registry.post_training_stages(model):
        path = RESULTS / f"{args.model}.{s['stage']}.labels.json"
        if not path.exists():
            print(f"- {s['stage']}: no classification run yet (`trainspotting classify {args.model} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = [r for r in data["records"] if r["label"]]
        n = len(records)
        print(f"### {s['stage']} — {data['dataset']} (n={n} labeled)\n")
        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        for label in classify.LABELS:
            k = counts.get(label, 0)
            lo, hi = _wilson(k, n)
            print(f"- {label:22s} {k / n * 100 if n else 0:5.1f}%  (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
        print()


def main():
    ap = argparse.ArgumentParser(prog="trainspotting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in [("facts", cmd_facts), ("sources", cmd_sources), ("report", cmd_report)]:
        p = sub.add_parser(name)
        p.add_argument("model", help=f"one of: {', '.join(sorted(registry.MODELS))}")
        p.set_defaults(fn=fn)
        if name == "sources":
            p.add_argument("--json", action="store_true", help="also write results/<model>.sources.json")

    p = sub.add_parser("ask", help="score sampled prompts against a free-form yes/no question")
    p.add_argument("model")
    p.add_argument("question")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("context", help="store the full training example behind each sampled prompt")
    p.add_argument("model")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("classify")
    p.add_argument("model")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.set_defaults(fn=cmd_classify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
