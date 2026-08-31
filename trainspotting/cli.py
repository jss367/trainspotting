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

# Every command takes one of these. A model walks its whole pipeline; a dataset
# is a single samplable dataset with no pipeline around it, and the layers that
# read rows cannot tell the difference (see registry.resolve).
TARGET_HELP = "model or dataset: " + ", ".join(registry.targets())


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
        # The design effect is unestimable here, not 1. With a single cluster
        # there is nothing to compare it against; with a unanimous outcome the
        # observed between-cluster variance is zero, which is 0/0 rather than
        # evidence of independence. Either way, falling back to the document
        # count would hand a clustered run the narrow interval for n independent
        # observations — and "no matches at all" is a likely answer to a pointed
        # question, so that branch fires exactly when the number matters. Use the
        # cluster count instead: assume documents sharing a shard told us one
        # thing, not m_c things. At one document per shard C is n and nothing
        # changes.
        return (*_wilson(p * C, C), float(C))
    ss = sum((sum(ys) - p * len(ys)) ** 2 for ys in clusters.values())
    var_cluster = C / ((C - 1) * n**2) * ss
    var_binomial = p * (1 - p) / n
    deff = max(1.0, var_cluster / var_binomial) if var_binomial else 1.0
    n_eff = max(1.0, n / deff)
    return (*_wilson(p * n_eff, n_eff), n_eff)


def _select_stages(args, stages_of, family):
    """The stages a command runs over: every one of `family`, narrowed by `--stage`.

    `family` names the group in the error message, so asking for a pretraining
    stage by a post-training name fails with the right suggestion.

    A target with none of the family at all is an error too, not an empty loop:
    a dataset has no pretraining corpora behind it, and `pretrain wildchat-1m`
    exiting silently having written nothing reads exactly like a sample that
    came back empty.
    """
    target = registry.resolve(args.target)
    stages = stages_of(target)
    if not stages:
        sys.exit(f"{args.target} has no {family} stages")
    if getattr(args, "stage", None):
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no {family} stage {args.stage!r} for {args.target}")
    return stages


def _sample_rows(stage, sample, seed):
    """(row, prompt) for each row of a deterministic (sample, seed) draw that has one.

    Rows carrying no user prompt drop out here, so the result is usually shorter
    than `sample`. The row travels with its prompt because part of what an
    example teaches is in the row rather than the text — an RL row's verifier
    settles its taxonomy label outright.
    """
    print(f"sampling {sample} rows from {stage['hf_dataset']} ...", file=sys.stderr)
    rows = hf.sample_rows(stage["hf_dataset"], sample, seed=seed)
    pairs = ((r, extract.extract_prompt(r, stage["prompt_path"])) for r in rows)
    return [(r, p) for r, p in pairs if p]


def _sample_prompts(stage, sample, seed):
    """Just the prompts, for the callers with no use for the row."""
    return [p for _, p in _sample_rows(stage, sample, seed)]


def _write_json(path, payload):
    """Write a result file, creating results/ if this is the first one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _counts(records):
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return counts


def _print_match_rate(stage, k, n, lo, hi, path):
    print(
        f"{stage}: {k}/{n} match = {k / n * 100 if n else 0:.1f}%"
        f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%) -> {path}",
        file=sys.stderr,
    )


def cmd_facts(args):
    target = registry.resolve(args.target)
    print(f"# {args.target} ({target['hf_model'] or target['name']})\n")
    for s in target["stages"]:
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
    target = registry.resolve(args.target)
    out = {}
    for s in registry.post_training_stages(target):
        freqs, counted, partial = hf.column_frequencies(
            s["hf_dataset"], s["source_columns"]
        )
        total = hf.num_rows(s["hf_dataset"])
        # Shares are over the rows the stats API actually scanned, which on a
        # big dataset is not all of them.
        counted = counted or total
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


def _label_post_training(args, question=None, slug=None):
    """sample → extract → classify → write, for each selected post-training stage.

    `question` selects the label mode. Without one, each prompt gets a single
    label from the fixed HHH taxonomy and the run lands in
    <target>.<stage>.labels.json. With one, each prompt gets a yes/no judgment of
    that question and the run lands in <target>.<stage>.ask-<slug>.json with the
    match rate and its interval. Everything else — which rows are drawn, which
    prompts survive extraction, what the envelope records about the run — is the
    same in both modes, and `classify` and `ask` sharing this loop is what keeps
    it that way.

    Taxonomy mode has one shortcut: a row whose verifier already settles its
    label is never sent to the model, which would answer about the prompt's
    topic instead. A free-form question gets no such shortcut — knowing what the
    reward checks does not answer it.
    """
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        rows = _sample_rows(s, args.sample, args.seed)
        prompts = [p for _, p in rows]
        fixed = [
            classify.verifier_label(row, registry.stage_kind(s)) if question is None else None
            for row, _ in rows
        ]
        ask = [p for p, f in zip(prompts, fixed) if not f]
        settled = len(prompts) - len(ask)
        print(
            f"classifying {len(ask)} prompts with {args.classifier}"
            + (f" ({settled} labeled by their verifier)" if settled else "")
            + " ...",
            file=sys.stderr,
        )
        asked = iter(
            classify.classify_prompts(
                ask,
                model=args.classifier,
                question=question,
                # A chat log is not a training example, so it is not judged as
                # one. Every model stage gets None here and takes the default.
                system=classify.system_for(registry.stage_kind(s), question),
            )
        )
        labels = [f or next(asked) for f in fixed]
        run = {
            "dataset": s["hf_dataset"],
            "sample": args.sample,
            "seed": args.seed,
            "classifier": args.classifier,
        }
        if question is None:
            records = []
            for p, lab, f in zip(prompts, labels, fixed):
                rec = {"prompt": extract.clip(p), "label": lab}
                if f:
                    rec["by"] = "verifier"
                records.append(rec)
            path = _write_json(
                RESULTS / f"{args.target}.{s['stage']}.labels.json",
                {**run, "records": records},
            )
            print(f"{s['stage']}: {_counts(records)}  -> {path}", file=sys.stderr)
        else:
            records = [
                {"prompt": extract.clip(p), "match": lab == "yes"}
                for p, lab in zip(prompts, labels)
                if lab
            ]
            path = _write_json(
                RESULTS / f"{args.target}.{s['stage']}.ask-{slug}.json",
                {"question": question, **run, "records": records},
            )
            k, n = sum(r["match"] for r in records), len(records)
            _print_match_rate(s["stage"], k, n, *_wilson(k, n), path)


def _label_pretrain_docs(args, question, slug):
    """Score the documents `pretrain` wrote against `question`.

    Judged from that file rather than re-sampled, so asking a second question
    scores the same documents and costs nothing but the API call.
    """
    for s in registry.pretrain_stages(registry.resolve(args.target)):
        docs_path = _pretrain_docs_source(args.target, s["stage"])
        if docs_path is None:
            print(
                f"{s['stage']}: no sample yet"
                f" (`trainspotting pretrain {args.target} --stage {s['stage']}`)",
                file=sys.stderr,
            )
            continue
        data = json.loads(docs_path.read_text())
        docs = data["records"]
        labels = classify.classify_prompts(
            # Stored as an excerpt spanning the whole document, so this judges
            # precisely the text the site shows. A corpus document does not
            # announce itself the way a prompt does, and the long-context mixes
            # run past 200k characters, so judging a 1,500-character prefix would
            # report a rate over opening boilerplate. Bigger inputs, fewer per
            # request.
            [d["text"] for d in docs],
            model=args.classifier,
            question=question,
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
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.ask-{slug}.json",
            {
                "question": question,
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
        )
        _print_match_rate(s["stage"], k, n, lo, hi, path)


def _warn_missing_pretrain_samples(args):
    """Warn before spending anything.

    The post-training stages cost an API call per batch, and finding out
    afterwards that the pretraining half had no sample to score is a slow way to
    learn it.
    """
    missing = [
        s["stage"]
        for s in registry.pretrain_stages(registry.resolve(args.target))
        if _pretrain_docs_source(args.target, s["stage"]) is None
    ]
    if missing:
        print(
            f"warning: no document sample for {', '.join(missing)}"
            f" — run `trainspotting pretrain {args.target}` first;"
            " scoring the post-training stages anyway\n",
            file=sys.stderr,
        )


def cmd_classify(args):
    """Label sampled prompts from every post-training stage with the HHH taxonomy."""
    _label_post_training(args)


def cmd_ask(args):
    """Score sampled prompts from every post-training stage against a free-form question."""
    # One short name ties a question's post-training and pretraining files together.
    slug = args.slug or re.sub(r"[^a-z0-9]+", "-", args.question.lower()).strip("-")[:60]
    print(f"question: {args.question}\n", file=sys.stderr)
    if args.pretrain:
        # A dataset has no corpora to score. Accepting the flag and quietly
        # scoring only the prompts would answer half the question asked.
        if not registry.pretrain_stages(registry.resolve(args.target)):
            sys.exit(f"--pretrain: {args.target} has no pretraining stages")
        _warn_missing_pretrain_samples(args)
    _label_post_training(args, question=args.question, slug=slug)
    if args.pretrain:
        _label_pretrain_docs(args, args.question, slug)


def _pretrain_docs_path(target_name: str, stage: str) -> Path:
    """Where `pretrain` writes a document sample."""
    return RESULTS / f"{target_name}.{stage}.docs.json"


def _pretrain_docs_source(target_name: str, stage: str) -> Path | None:
    """Where to read one back, or None if this checkout has neither copy.

    `results/*.docs.json` is gitignored — it is a regenerable cache — so on a
    fresh clone the only copy of a committed sample is the one under docs/data/
    that the site serves. Reading only from results/ would tell someone who just
    cloned the repo that the sample shipped with it does not exist.
    """
    for path in (
        _pretrain_docs_path(target_name, stage),
        SITE_DATA / f"{target_name}.{stage}.docs.json",
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
    for s in _select_stages(args, registry.pretrain_stages, "pretraining"):
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

        docs, short = pretrain.sample_documents(
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
        if len(records) < args.sample:
            # A corpus can genuinely fail to fill the request — 55 huge shards
            # cannot yield 300 documents at one apiece — so say so rather than
            # letting "sample" claim a size the file does not have.
            print(
                f"  note: asked for {args.sample}, corpus yielded {len(records)}",
                file=sys.stderr,
            )
        path = _write_json(
            _pretrain_docs_path(args.target, s["stage"]),
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
                # Shard draws that contributed fewer documents than asked
                # for. Non-zero means the sample is weighted by reachable
                # document density as well as by size.
                "short_draws": short,
                "scope": s.get("sample_scope"),
                "caveat": pretrain.sampling_caveat(args.docs_per_shard),
                "shards": len(shards),
                "bytes": total_bytes,
                "groups": groups,
                "records": records,
            },
        )

        print(
            f"{s['stage']}: {len(records)} documents -> {path}"
            f" ({path.stat().st_size / 1e6:.1f} MB)"
            + (f", {short} short draw(s) made up by others" if short else ""),
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
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        sample, seed = args.sample, args.seed
        if args.from_labels:
            # The same (sample, seed) draws the same rows, so a committed classify
            # run already holds the prompts verbatim — no reason to re-fetch them.
            labels_path = RESULTS / f"{args.target}.{s['stage']}.labels.json"
            if not labels_path.exists():
                sys.exit(f"{labels_path} not found — drop --from-labels to sample from HuggingFace")
            prior = json.loads(labels_path.read_text())
            prompts = [r["prompt"] for r in prior["records"]]
            sample, seed = prior["sample"], prior["seed"]
            print(f"reusing {len(prompts)} prompts from {labels_path.name}", file=sys.stderr)
        else:
            # Clip before detecting, not after. A classify run stores the clipped
            # prompt, so detecting the full text here would make --from-labels
            # disagree with this path on the handful of prompts past the cutoff.
            prompts = [extract.clip(p) for p in _sample_prompts(s, args.sample, args.seed)]
        records = []
        for p in prompts:
            code, conf = languages.detect(p)
            records.append(
                {"prompt": p, "label": code, "confidence": round(conf, 3)}
            )
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.languages.json",
            {
                "dataset": s["hf_dataset"],
                "sample": sample,
                "seed": seed,
                "detector": "py3langid",
                "records": records,
            },
        )
        counts = _counts(records)
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
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        print(f"re-fetching {args.sample} sampled rows from {s['hf_dataset']} ...", file=sys.stderr)
        rows = hf.sample_rows_with_index(s["hf_dataset"], args.sample, seed=args.seed)
        records = []
        for row_index, row in rows:
            prompt = extract.extract_prompt(row, s["prompt_path"])
            if prompt:
                records.append(
                    context.build(row, registry.stage_kind(s), prompt, row_index)
                )
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.context.json",
            {
                "dataset": s["hf_dataset"],
                "stage": s["stage"],
                "sample": args.sample,
                "seed": args.seed,
                "records": records,
            },
        )
        print(f"{s['stage']}: {len(records)} records -> {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def cmd_report(args):
    target = registry.resolve(args.target)
    kind = "Training-data audit" if target["is_model"] else "Dataset audit"
    print(f"# {kind}: {args.target}\n")
    if target["is_model"]:
        print("## Stage sizes\n")
        for s in target["stages"]:
            if s.get("tokens"):
                print(f"- {s['stage']}: {s['name']}, {_fmt_tokens(s['tokens'])} tokens")
            else:
                print(f"- {s['stage']}: {s['name']} ({s['hf_dataset']})")
    elif target.get("note"):
        print(target["note"])
    # The same seven labels mean different things by kind, and the heading is
    # the only place the report says which.
    print(
        "\n## HHH classification (sampled)\n"
        if target["is_model"]
        else "\n## What the prompts ask for (sampled)\n"
    )
    for s in registry.post_training_stages(target):
        path = RESULTS / f"{args.target}.{s['stage']}.labels.json"
        if not path.exists():
            print(f"- {s['stage']}: no classification run yet (`trainspotting classify {args.target} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = [r for r in data["records"] if r["label"]]
        n = len(records)
        # Verifier-labeled rows never reached the classifier, so name both
        # counts rather than crediting the model for all of them.
        v = sum(1 for r in records if r.get("by") == "verifier")
        by = f", {v} by their verifier" if v else ""
        print(f"### {s['stage']} — {data['dataset']} (n={n} labeled{by})\n")
        counts = _counts(records)
        for label in classify.LABELS:
            k = counts.get(label, 0)
            lo, hi = _wilson(k, n)
            print(f"- {label:22s} {k / n * 100 if n else 0:5.1f}%  (95% CI {lo * 100:.1f}–{hi * 100:.1f}%)")
        print()

    print("\n## Language (sampled, detected locally)\n")
    for s in registry.post_training_stages(target):
        path = RESULTS / f"{args.target}.{s['stage']}.languages.json"
        if not path.exists():
            print(f"- {s['stage']}: no language run yet (`trainspotting languages {args.target} --stage {s['stage']}`)")
            continue
        data = json.loads(path.read_text())
        records = data["records"]
        n = len(records)
        counts = _counts(records)
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
        p.add_argument("target", help=TARGET_HELP)
        p.set_defaults(fn=fn)
        if name == "sources":
            p.add_argument("--json", action="store_true", help="also write results/<target>.sources.json")

    p = sub.add_parser("ask", help="score sampled prompts against a free-form yes/no question")
    p.add_argument("target", help=TARGET_HELP)
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
    p.add_argument("target", help=TARGET_HELP)
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
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("languages", help="detect the natural language of sampled prompts (local, no API key)")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-labels", action="store_true",
                   help="read prompts from the committed classify run instead of re-sampling HuggingFace")
    p.set_defaults(fn=cmd_languages)

    p = sub.add_parser("classify")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.set_defaults(fn=cmd_classify)

    args = ap.parse_args()
    # Canonicalize once, here, so every result path and every lookup downstream
    # agrees. `resolve` accepts case variants; writing the raw argument into the
    # filename meant `classify WildChat-1M` produced a file the site — which
    # indexes the registry key — never asks for, and the run silently didn't
    # exist.
    try:
        args.target = registry.resolve(args.target)["target"]
    except KeyError as e:
        sys.exit(e.args[0])
    args.fn(args)


if __name__ == "__main__":
    main()
