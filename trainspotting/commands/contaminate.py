"""`contaminate`: is a benchmark's test set in the training data, and on which side?"""

import hashlib
import json
import sys

from .. import benchmarks, contamination, grep, hf, infinigram, lookup, registry
from ..paths import RESULTS
from .common import _filename_part, _fmt_bytes, _stamp, _write_json
from .grep import _grep_plan


def _contam_probes(spec, items, words):
    """One probe per probable part of each item, and why the rest were not.

    A part the server cut is not probed: the window would be cut from a
    fragment, and might land on the cut itself. A part too short for a window is
    not probed either. Both are counted so the summary can say how many items
    the check actually reached.

    The items with a cut part come back as well, by index. Their other parts
    are still probed and a hit on one stands, but a miss on them is not a miss
    on the item — the cut part may be the one that was copied — so the rollups
    keep such an item out of the denominator unless it hit.
    """
    probes, skipped, cut = [], {"truncated": 0, "short": 0}, set()
    for it in items:
        for part in benchmarks.parts(spec):
            if benchmarks.column(spec, part) in it["truncated"]:
                skipped["truncated"] += 1
                cut.add(it["index"])
                continue
            text = benchmarks.item_text(spec, it["row"], part)
            pr = benchmarks.probe(text or "", words)
            if pr is None:
                skipped["short"] += 1
                continue
            probes.append({"id": len(probes), "item": it["index"], "part": part, **pr})
    return probes, skipped, sorted(cut)


# The settings that decide what a contamination run measures, at their defaults.
# The parser takes its defaults from here and `_contam_slug` compares against
# it, so a run at the defaults is recognized as one however the flags are spelt.
CONTAM_DEFAULTS = {
    "items": 200,
    "seed": 0,
    "words": benchmarks.WORDS,
    "field": None,
    "case_sensitive": False,
}


def _contam_settings(args) -> dict:
    """The settings that change which probes are cut or where they are searched.

    `--stage` and `--index` are left out because each already names its own
    file; `--examples` and the byte cap change what is kept, not what is found.
    The field list is sorted and deduplicated so the spelling of the flags does
    not make a new run.
    """
    return {
        "items": args.items,
        "seed": args.seed,
        "words": args.words,
        "field": sorted(set(args.field)) if args.field else None,
        "case_sensitive": bool(args.case_sensitive),
    }


def _contam_slug(benchmark: str, settings: dict) -> str:
    """The name a contamination run's result files carry: the benchmark id at
    the default settings, else the id and a hash of the settings.

    A run with `--field prompt` or `--items 20` cuts different probes or reads
    a different side, and writing it to the file of the default run overwrote
    a full measurement with a narrow one and left no sign. The default keeps
    the bare id, so the committed runs keep their names and a diagnostic run
    cannot overwrite them; `--slug` names a run instead, as it does for `grep`.
    """
    if settings == CONTAM_DEFAULTS:
        return benchmark
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    return f"{benchmark}-{digest}"


def _contam_unscanned(all_stages: list[dict], scanned: list[dict], corpus_only: bool) -> list[str]:
    """The stage names the summary must call unscanned rather than let read as clean.

    With --corpus-only no mix is read at all, so every stage is unscanned — the
    stages the user selected included. Filtering on the selection alone came
    back empty there, and a corpus-only summary then said nothing about the
    stages it had not looked at, which is the one silence this command exists
    to refuse.
    """
    if corpus_only:
        return [s["stage"] for s in all_stages]
    return [s["stage"] for s in all_stages if s not in scanned]


def _contam_refusal(args, has_stages: bool, index: str | None) -> str | None:
    """Why this run would measure nothing, or None when it would measure something.

    Two sides, and each can be taken away by the target or by a flag: a dataset
    has no corpus index — nothing was pretrained on it — and --corpus-only
    removes its scan; a base model has no post-training stages and --no-corpus
    removes its corpus side. Left alone, either combination fetched the
    benchmark, wrote no result and exited 0: a successful run that made no
    measurement. Decided before any row is fetched, so refusing costs nothing.
    """
    scan = has_stages and not args.corpus_only
    count = bool(index) and not args.no_corpus
    if scan or count:
        return None
    why = []
    if not has_stages:
        why.append(f"{args.target} has no post-training stages to scan")
    elif args.corpus_only:
        why.append("--corpus-only skips the post-training scans")
    if not index:
        why.append(f"{args.target} has no corpus index (nothing was pretrained on a dataset)")
    elif args.no_corpus:
        why.append("--no-corpus skips the corpus side")
    return "nothing to measure: " + "; ".join(why)


def _contam_index(args, target: dict) -> str | None:
    """The index the corpus side counts in, or None when the target has no corpus side.

    `--index` moves a model to a different corpus — the full OLMo 2 index for the
    comparison in the README, or a Dolma 3 index the day Ai2 publishes one. It
    cannot give a dataset one. Nothing was pretrained on a dataset, so there is
    no corpus behind it to count in, and `registry.infinigram_index` says None
    for exactly that reason; letting the flag override that would count the
    probes in some model's corpus and file the result under the dataset's name,
    as if that corpus were part of its training. Refused here, before the
    benchmark is fetched, the same way `_contam_refusal` refuses a run with
    nothing to measure.
    """
    if args.index and not target["is_model"]:
        sys.exit(
            f"--index {args.index}: {args.target} is a dataset, and nothing was pretrained "
            f"on a dataset. There is no corpus behind it to count in; a count over "
            f"{args.index} would describe a model nobody named. Drop --index, or name a model."
        )
    return args.index or registry.infinigram_index(target)


def cmd_contaminate(args):
    """Is a benchmark's test set in the training data, where, and on which side?

    Every layer this needs already exists: `grep` reads every row of a mix and
    keeps the sides apart, `lookup` counts an exact string in a corpus index, and
    `benchmarks` says which text is the question and which the answer. What this
    adds is the join — one read per mix for all the probes, and the hits handed
    back to the items they came from.
    """
    try:
        spec = benchmarks.resolve(args.benchmark)
    except KeyError as e:
        # The message names the known benchmarks; a traceback would bury it.
        sys.exit(e.args[0])
    target = registry.resolve(args.target)
    all_stages = registry.post_training_stages(target)
    index = _contam_index(args, target)
    refusal = _contam_refusal(args, bool(all_stages), index)
    if refusal:
        sys.exit(refusal)
    # Resolved before the rows are read, so the stamp names the tree the items
    # were cut from rather than one published while the pages were fetched —
    # the same order every other row-drawing command here uses.
    bench_revision = hf.dataset_revision(spec["hf_dataset"])
    total = benchmarks.total_items(spec)
    indices = benchmarks.pick_indices(total, args.items, args.seed)
    items = benchmarks.fetch_items(spec, indices)
    # And again after. The pages are served from whatever the tree is at the
    # time, so a benchmark republished mid-fetch would leave probes cut from two
    # versions of it under one SHA, and a probe nobody can find in the recorded
    # revision. Same check and same field as the paged samplers.
    bench_moved = hf.dataset_revision(spec["hf_dataset"])
    if bench_revision and bench_moved and bench_moved != bench_revision:
        print(
            f"# note: {spec['hf_dataset']} moved from {bench_revision[:7]} to "
            f"{bench_moved[:7]} while its rows were fetched; the items may straddle both trees",
            file=sys.stderr,
        )
    probes, skipped, cut = _contam_probes(spec, items, args.words)
    n_q = sum(1 for p in probes if p["part"] == "question")
    n_c = sum(1 for p in probes if p["part"] == "choices")
    n_a = len(probes) - n_q - n_c
    # Same rule as `grep`: `--slug` is a filename component, not a path.
    settings = _contam_settings(args)
    slug = _filename_part(args.slug) if args.slug else _contam_slug(spec["id"], settings)
    print(
        f"# contaminate {spec['name']} ({spec['hf_dataset']} {spec['config']}/{spec['split']}) "
        f"on {args.target}\n"
        f"# {len(items):,} of {total:,} items"
        + (f" (seed {args.seed})" if len(items) < total else "")
        + f", {args.words}-word probes: {n_q} question"
        + (f", {n_c} choices" if spec.get("choices") else "")
        + f", {n_a} answer"
        + (f"; {skipped['short']} part(s) too short to probe" if skipped["short"] else "")
        + (f"; {skipped['truncated']} cut by the server" if skipped["truncated"] else "")
        + f"\n# result files: *.contam-{slug}.json"
        + ("" if args.slug or slug == spec["id"] else
           " (settings differ from the defaults, so the default run's files are left alone)")
        + "\n",
        file=sys.stderr,
    )
    if not probes:
        sys.exit("nothing to probe")

    bench = {
        "id": spec["id"], "name": spec["name"], "hf_dataset": spec["hf_dataset"],
        "config": spec["config"], "split": spec["split"], "revision": bench_revision,
        **({"revision_moved_to": bench_moved}
           if bench_revision and bench_moved and bench_moved != bench_revision else {}),
        "total_items": total, "question_field": spec["question"],
        "choices_field": spec.get("choices"), "answer_field": spec.get("answer"),
    }
    common = {
        "benchmark": bench,
        "slug": slug,
        "items_requested": args.items,
        "seed": args.seed,
        "words": args.words,
        # What --field asked for, as distinct from `fields`, which is what a
        # stage's mix let the scan read. Together with the keys above this is
        # every setting the slug hashes, so a file can say why it has its name.
        "fields_requested": settings["field"],
        # Which items the rate is over is each side's own: `items_probed` and
        # `items_unresolved` come from its rollup, which settles the items in
        # `items_cut` — a part cut by the server — only when they hit.
        "items_cut": cut,
        "skipped": skipped,
        "case_sensitive": args.case_sensitive,
        "probes": probes,
    }

    # --- the post-training mixes, every row -------------------------------
    stages = all_stages
    if args.stage:
        stages = [s for s in stages if s["stage"] == args.stage]
        if not stages:
            sys.exit(f"no post-training stage {args.stage!r} for {args.target}")
    unscanned = _contam_unscanned(all_stages, stages, args.corpus_only)
    stage_runs = []
    if stages and not args.corpus_only:
        con = grep.connect()
        args.by = None
        plan = _grep_plan(con, args, stages)
        total_bytes = sum(p["bytes"] for p in plan)
        print(f"# {len(plan)} stage(s), {_fmt_bytes(total_bytes)} to read", file=sys.stderr)
        for p in plan:
            print(
                f"- {p['stage']['stage']:6s} {p['rows']:>9,} rows  {_fmt_bytes(p['bytes']):>9}"
                f"  {'/'.join(p['exprs'])}  ({p['stage']['hf_dataset']})",
                file=sys.stderr,
            )
        cap = int(args.max_gb * 1e9)
        if total_bytes > cap and not args.yes:
            sys.exit(
                f"\nthat is {_fmt_bytes(total_bytes)}, over the {args.max_gb} GB cap, and nothing has "
                f"been read yet. Narrow it (--stage, --field) or allow it (--max-gb "
                f"{total_bytes / 1e9:.1f}, or --yes)."
            )
        for p in plan:
            s = p["stage"]
            if p["listing"]["partial"]:
                print(f"\n{s['stage']}: WARNING — the server converted only part of this repo, "
                      "so every count below is a floor", file=sys.stderr)
            if p["unsearched"]:
                print(f"\n{s['stage']}: not searched: {', '.join(p['unsearched'])}", file=sys.stderr)
            print(f"\nscanning {s['stage']} ({_fmt_bytes(p['bytes'])}) ...", file=sys.stderr)
            result = contamination.scan(
                con, grep.read_parquet_sql(p["listing"]["urls"]), p["exprs"], p["source"],
                probes, case_sensitive=args.case_sensitive, examples=args.examples,
            )
            rollup = contamination.stage_items(probes, result["probe_hits"], cut)
            hit = rollup["items"]
            totals = (
                grep.source_totals(con, grep.read_parquet_sql(p["listing"]["urls"]), p["source"])
                if p["source"] else {}
            )
            payload = {
                "dataset": s["hf_dataset"],
                "stage": s["stage"],
                **common,
                "has_answer_probes": n_a > 0,
                "fields": list(p["exprs"]),
                "available_fields": p["available"],
                "source_column": p["source_column"],
                **_stamp(s["hf_dataset"], revision=p["revision"]),
                "parquet_revision": p["listing"]["revision"],
                "partial": p["listing"]["partial"],
                "shards": len(p["listing"]["urls"]),
                "bytes_read": p["bytes"],
                "unsearched_columns": p["unsearched"],
                "total_rows": p["rows"],
                "rows_by_source": totals,
                **rollup,
                **result,
            }
            path = _write_json(
                RESULTS / f"{args.target}.{s['stage']}.contam-{slug}.json", payload
            )
            stage_runs.append(payload)
            print(f"{s['stage']}: {len(hit['any'])}/{len(rollup['items_probed'])} items seen, "
                  f"{result['matched']:,} rows  -> {path}", file=sys.stderr)
            for src, n in list(result["by_source"].items())[:8]:
                of = totals.get(src)
                rate = f" = {n / of * 100:5.2f}% of it" if of else ""
                print(f"  {n:>7,} / {of or p['rows']:>9,}{rate}  {src}", file=sys.stderr)
    elif not stages:
        print(f"{args.target} has no post-training stages to scan", file=sys.stderr)

    # --- the corpus, by exact string ---------------------------------------
    corpus_run = None
    if index and not args.no_corpus:
        caveat = registry.infinigram_caveat(target, index)
        print(f"\ncounting {len(probes)} probes in {index} ...", file=sys.stderr)
        counts = []
        for i, p in enumerate(probes, 1):
            try:
                c = lookup.count(index, p["literal"])
            except lookup.LookupError_ as e:
                # The index did not answer — it rejected the query, or five
                # attempts got no reply. That is not zero occurrences, and it
                # must not be written as one: a run of hundreds of requests will
                # see a transient failure sooner or later, and a copied item
                # whose probe happened to be the one that failed would otherwise
                # come out clean. `occurrences` is null, and `corpus_items`
                # keeps the probe out of every count and every denominator.
                counts.append({"probe": p["id"], "occurrences": None, "approx": False,
                               "error": str(e)})
            else:
                counts.append({"probe": p["id"], "occurrences": c["occurrences"],
                               "approx": c["approx"]})
            print(f"\r  {i}/{len(probes)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        rollup = contamination.corpus_items(probes, counts, cut)
        if rollup["errors"]:
            print(f"  {len(rollup['errors'])} probe(s) could not be counted — not zeros",
                  file=sys.stderr)
        # `items_probed` here is the corpus's own — the items it settled. An
        # unanswered probe unsettles an item here where a stage scan, reading
        # every row, has none; a cut part unsettles it on both sides.
        corpus_run = {
            "stage": "corpus",
            "index": index,
            "covers": infinigram.INDEXES.get(index),
            "caveat": caveat,
            **common,
            **_stamp(),
            "counts": counts,
            **rollup,
        }
        # The index is in the stage slot, as the mix is for a stage run: two
        # `--index` runs describe two corpora and must not overwrite each other.
        # The slug is the stage files', since the same settings cut its probes.
        path = _write_json(
            RESULTS / f"{args.target}.corpus-{_filename_part(index)}.contam-{slug}.json",
            corpus_run,
        )
        print(f"  -> {path}", file=sys.stderr)

    print(file=sys.stderr)
    # A corpus side the flag removed is named, as a stage --corpus-only removed
    # is; a dataset has none to name, and _contam_index already said so.
    skipped = "--no-corpus" if index and args.no_corpus else None
    for line in contamination.summary(stage_runs, corpus_run, unscanned, corpus_skipped=skipped):
        print(line, file=sys.stderr)
