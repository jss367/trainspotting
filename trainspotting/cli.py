import argparse
import json
import math
import re
import sys
from pathlib import Path

from . import classify, context, extract, hf, languages, pretrain, registry

RESULTS = Path(__file__).resolve().parent.parent / "results"
# The committed half of the bulk artifacts: gitignored under results/, shipped
# here for the site, and so the only copy present in a fresh clone.
SITE_DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


def _positive_int(value: str) -> int:
    """argparse type for counts. Zero divides by zero deep inside the sampler and
    a negative one silently returns nothing; both should be a usage error."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _fmt_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    return f"{n:,}"


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. k and n may be non-integer effective counts."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _cluster_wilson(records: list[dict], key: str = "shard") -> tuple[float, float, float]:
    """Wilson interval widened by the design effect of clustering by `key`.

    Documents drawn from one shard share a topic cluster, so they are not
    independent observations and a binomial interval over the document count is
    too narrow. Rescaling the match count to the number of distinct shards, which
    is what this used to do, is not an interval over shards either: a shard
    contributing five matches and one contributing a single non-match would round
    to "two successes out of two clusters", hiding the disagreement between them.

    So take the design effect properly, from the Taylor-linearised variance of the
    ratio estimator over clusters:

        Var(p) = C / ((C - 1) · M²) · Σ (y_c - p·m_c)²

    where cluster c holds m_c documents of which y_c match, and M = Σ m_c. Divide
    by the binomial variance to get the design effect, and evaluate Wilson at the
    effective sample size n/deff. Wilson is kept rather than a normal interval
    because it still behaves at rates near 0 and 1, which several of these are.

    Returns (lo, hi, n_effective). With one document per shard every cluster has
    size one, the design effect is C/(C-1) ≈ 1, and this is the ordinary interval.
    """
    n = len(records)
    if n == 0:
        return 0.0, 0.0, 0.0
    clusters: dict[str, list[int]] = {}
    for r in records:
        clusters.setdefault(r.get(key) or "", []).append(1 if r["match"] else 0)
    C = len(clusters)
    p = sum(r["match"] for r in records) / n
    if C < 2 or p in (0.0, 1.0):
        # No between-cluster variance to estimate (or none to find): fall back to
        # the document-level interval rather than inventing a design effect.
        return (*_wilson(p * n, n), float(n))
    ss = sum((sum(ys) - p * len(ys)) ** 2 for ys in clusters.values())
    var_cluster = C / ((C - 1) * n**2) * ss
    var_binomial = p * (1 - p) / n
    deff = max(1.0, var_cluster / var_binomial) if var_binomial else 1.0
    n_eff = max(1.0, n / deff)
    return (*_wilson(p * n_eff, n_eff), n_eff)


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
        elif s.get("sample_dataset"):
            line += f" — samplable ({s['sample_dataset']})"
        print(line)
        if s.get("note"):
            print(f"    {s['note']}")


def cmd_sources(args):
    model = registry.get_model(args.model)
    out = {}
    for s in registry.post_training_stages(model):
        freqs = hf.column_frequencies(s["hf_dataset"], s["source_columns"])
        total = hf.num_rows(s["hf_dataset"])
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
            "total": total,
            "columns": freqs,
            "links": links,
        }
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

    # Warn before spending anything. The post-training stages cost an API call
    # per batch, and finding out afterwards that the pretraining half has no
    # sample to score is a slow way to learn it.
    if args.pretrain:
        missing = [
            st["stage"]
            for st in registry.pretrain_stages(model)
            if _pretrain_docs_source(args.model, st["stage"]) is None
        ]
        if missing:
            print(
                f"warning: no document sample for {', '.join(missing)}"
                f" — run `trainspotting pretrain {args.model}` first;"
                " scoring the post-training stages anyway\n",
                file=sys.stderr,
            )

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

    if not args.pretrain:
        return

    # Pretraining documents are judged from the file `pretrain` wrote, not
    # re-sampled, so asking a second question scores the same documents and
    # costs nothing but the API call.
    for s in registry.pretrain_stages(model):
        docs_path = _pretrain_docs_source(args.model, s["stage"])
        if docs_path is None:
            print(
                f"{s['stage']}: no sample yet"
                f" (`trainspotting pretrain {args.model} --stage {s['stage']}`)",
                file=sys.stderr,
            )
            continue
        data = json.loads(docs_path.read_text())
        docs = data["records"]
        # Judge an excerpt spanning the whole document, not its first 1,500
        # characters. A corpus document does not announce itself the way a
        # prompt does, and the long-context mixes run past 200k characters, so
        # truncating would report a rate over opening boilerplate. Bigger inputs
        # mean fewer per request.
        labels = classify.classify_prompts(
            # Already stored as an excerpt spanning the whole document, so this
            # judges precisely the text the site shows.
            [d["text"] for d in docs],
            model=args.classifier,
            question=args.question,
            system=classify.ASK_DOC_SYSTEM,
            max_chars=extract.MAX_DOCUMENT_CHARS,
            batch_size=5,
        )
        records = [
            {
                "prompt": d["text"],
                "match": lab == "yes",
                "source": d["source"],
                "topic": d["topic"],
                "shard": d["shard"],
            }
            for d, lab in zip(docs, labels)
            if lab
        ]
        k, n = sum(r["match"] for r in records), len(records)
        lo, hi, n_eff = _cluster_wilson(records)
        path = RESULTS / f"{args.model}.{s['stage']}.ask-{slug}.json"
        path.write_text(
            json.dumps(
                {
                    "question": args.question,
                    "dataset": data["dataset"],
                    "stage": s["stage"],
                    "sample": data["sample"],
                    "seed": data["seed"],
                    "classifier": args.classifier,
                    "scope": data.get("scope"),
                    "caveat": data.get("caveat"),
                    "judged_chars": extract.MAX_DOCUMENT_CHARS,
                    "n_effective": round(n_eff, 2),
                    # Stored, not recomputed by the site: the cluster correction
                    # lives in one place so the page and the CLI cannot drift.
                    "ci": [lo, hi],
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


def _pretrain_docs_path(model_name: str, stage: str) -> Path:
    """Where `pretrain` writes a document sample."""
    return RESULTS / f"{model_name}.{stage}.docs.json"


def _pretrain_docs_source(model_name: str, stage: str) -> Path | None:
    """Where to read one back, or None if this checkout has neither copy.

    `results/*.docs.json` is gitignored — it is a regenerable cache — so on a
    fresh clone the only copy of a committed sample is the one under docs/data/
    that the site serves. Reading only from results/ would tell someone who just
    cloned the repo that the sample shipped with it does not exist.
    """
    for path in (
        _pretrain_docs_path(model_name, stage),
        SITE_DATA / f"{model_name}.{stage}.docs.json",
    ):
        if path.exists():
            return path
    return None


def cmd_pretrain(args):
    """Sample documents from a stage's Dolma 3 shard repo.

    The datasets-server cannot serve these corpora (it indexes only the first
    ~5 GB and the shards are topic-ordered), so this reads the repo files by
    range request instead. No model is called; this is the deterministic half,
    and `ask --pretrain` scores whatever it wrote.
    """
    model = registry.get_model(args.model)
    stages = registry.pretrain_stages(model)
    if args.stage:
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no pretraining stage {args.stage!r} for {args.model}")
    RESULTS.mkdir(exist_ok=True)
    for s in stages:
        dataset = s["sample_dataset"]
        print(f"listing shards in {dataset} ...", file=sys.stderr)
        shards, revision = pretrain.list_shards(dataset)
        groups = pretrain.group_sizes(shards)
        total_bytes = sum(x["size"] for x in shards)
        print(
            f"  {len(shards):,} shards, {total_bytes / 1e9:.0f} GB compressed,"
            f" {len(groups)} source/topic groups at {revision[:7]}",
            file=sys.stderr,
        )

        def progress(i, n, path):
            print(f"\r  fetching shard {i}/{n} ", end="", file=sys.stderr, flush=True)

        docs, barren = pretrain.sample_documents(
            dataset,
            args.sample,
            seed=args.seed,
            revision=revision,
            shards=shards,
            docs_per_shard=args.docs_per_shard,
            progress=progress,
        )
        print(file=sys.stderr)
        records = [
            {
                "id": d["id"],
                # An excerpt spanning the document, not its first 12k characters.
                # These run past 200k in the long-context mixes, and a prefix
                # would be the nav bar and the abstract — unrepresentative both
                # to read on the site and to classify. `chars` keeps the true
                # length so nothing pretends the excerpt is the whole document.
                "text": extract.excerpt(d["text"]),
                "chars": len(d["text"]),
                "source": d["source"],
                "topic": d["topic"],
                "shard": d["shard"],
                "metadata": d["metadata"],
            }
            for d in docs
        ]
        path = _pretrain_docs_path(args.model, s["stage"])
        if len(records) < args.sample:
            # A corpus can genuinely fail to fill the request — 55 huge shards
            # cannot yield 300 documents at one apiece — so say so rather than
            # letting "sample" claim a size the file does not have.
            print(
                f"  note: asked for {args.sample}, corpus yielded {len(records)}",
                file=sys.stderr,
            )
        path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    # The exact commit the composition and documents came from.
                    # "main" moves; a result file that cites exact byte shares
                    # has to say which revision it counted.
                    "revision": revision,
                    "stage": s["stage"],
                    "name": s["name"],
                    "sample": len(records),
                    "requested": args.sample,
                    "seed": args.seed,
                    "docs_per_shard": args.docs_per_shard,
                    # Shard draws replaced because they yielded nothing usable.
                    # Non-zero means the sample is weighted by reachable unique
                    # documents as well as by size.
                    "barren_draws": barren,
                    "scope": s.get("sample_scope"),
                    "caveat": pretrain.sampling_caveat(args.docs_per_shard),
                    "shards": len(shards),
                    "bytes": total_bytes,
                    "groups": groups,
                    "records": records,
                },
                indent=2,
            )
        )
        print(
            f"{s['stage']}: {len(records)} documents -> {path}"
            f" ({path.stat().st_size / 1e6:.1f} MB)"
            + (f", {barren} barren draw(s) replaced" if barren else ""),
            file=sys.stderr,
        )


def cmd_languages(args):
    """Detect which natural language each sampled prompt is written in.

    Sampling is deterministic in (sample, seed), so the defaults pull exactly
    the rows a classify run labeled and the drill-down lines up. Detection is
    local — no API key, no cost — because the Dolci datasets carry no language
    column of their own and the only label that comes close is Instruct-SFT's
    single `Multilingual` domain bucket.
    """
    model = registry.get_model(args.model)
    stages = registry.post_training_stages(model)
    if args.stage:
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no post-training stage {args.stage!r} for {args.model}")
    RESULTS.mkdir(exist_ok=True)
    for s in stages:
        sample, seed = args.sample, args.seed
        if args.from_labels:
            # The same (sample, seed) draws the same rows, so a committed classify
            # run already holds the prompts verbatim — no reason to re-fetch them.
            labels_path = RESULTS / f"{args.model}.{s['stage']}.labels.json"
            if not labels_path.exists():
                sys.exit(f"{labels_path} not found — drop --from-labels to sample from HuggingFace")
            prior = json.loads(labels_path.read_text())
            prompts = [r["prompt"] for r in prior["records"]]
            sample, seed = prior["sample"], prior["seed"]
            print(f"reusing {len(prompts)} prompts from {labels_path.name}", file=sys.stderr)
        else:
            print(f"sampling {args.sample} rows from {s['hf_dataset']} ...", file=sys.stderr)
            rows = hf.sample_rows(s["hf_dataset"], args.sample, seed=args.seed)
            prompts = [extract.extract_prompt(r, s["prompt_path"]) for r in rows]
            # Clip before detecting, not after. A classify run stores the clipped
            # prompt, so detecting the full text here would make --from-labels
            # disagree with this path on the handful of prompts past the cutoff.
            prompts = [extract.clip(p) for p in prompts if p]
        records = []
        for p in prompts:
            code, conf = languages.detect(p)
            records.append(
                {"prompt": p, "label": code, "confidence": round(conf, 3)}
            )
        path = RESULTS / f"{args.model}.{s['stage']}.languages.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": s["hf_dataset"],
                    "sample": sample,
                    "seed": seed,
                    "detector": "py3langid",
                    "records": records,
                },
                indent=2,
            )
        )
        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        n = len(records)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        summary = ", ".join(f"{languages.name(c)} {k / n * 100:.1f}%" for c, k in top)
        print(f"{s['stage']}: n={n} — {summary} -> {path}", file=sys.stderr)


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

    print("\n## Language (sampled, detected locally)\n")
    for s in registry.post_training_stages(model):
        path = RESULTS / f"{args.model}.{s['stage']}.languages.json"
        if not path.exists():
            print(f"- {s['stage']}: no language run yet (`trainspotting languages {args.model} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = data["records"]
        n = len(records)
        counts = {}
        for r in records:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        non_en = n - counts.get("en", 0) - counts.get(languages.UNDETERMINED, 0)
        lo, hi = _wilson(non_en, n)
        print(f"### {s['stage']} — {data['dataset']} (n={n} detected)\n")
        print(f"- not English: {non_en / n * 100 if n else 0:.1f}%  (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
        for code, k in sorted(counts.items(), key=lambda kv: -kv[1]):
            if code in ("en", languages.UNDETERMINED):
                continue
            print(f"  - {languages.name(code):20s} {k / n * 100:5.1f}%  ({k}/{n})")
        und = counts.get(languages.UNDETERMINED, 0)
        if und:
            print(f"- undetermined: {und / n * 100:.1f}%  ({und}/{n}) — too short, too much code, or too evenly mixed to call")
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
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.add_argument(
        "--pretrain",
        action="store_true",
        help="also score pretraining documents sampled by `trainspotting pretrain`",
    )
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("pretrain", help="sample documents from the Dolma 3 pretraining shards")
    p.add_argument("model")
    p.add_argument("--stage", help="only this stage (pretrain/midtrain/long-context)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--docs-per-shard",
        type=_positive_int,
        default=1,
        help="documents kept per sampled shard; >1 is faster but the documents "
        "are correlated, which widens any interval computed over them",
    )
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser("context", help="store the full training example behind each sampled prompt")
    p.add_argument("model")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("languages", help="detect the natural language of sampled prompts (local, no API key)")
    p.add_argument("model")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-labels", action="store_true",
                   help="read prompts from the committed classify run instead of re-sampling HuggingFace")
    p.set_defaults(fn=cmd_languages)

    p = sub.add_parser("classify")
    p.add_argument("model")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.set_defaults(fn=cmd_classify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
