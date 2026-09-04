import argparse
import hashlib
import json
import math
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from . import (
    behavior,
    benchmarks,
    bif,
    budget,
    casestudy,
    classify,
    contamination,
    context,
    extract,
    grep,
    hf,
    infinigram,
    influence,
    languages,
    lookup,
    pretrain,
    registry,
    search,
    stance,
    steps,
)
from . import paths
from .paths import RESULTS
from .stats import cluster_wilson as _cluster_wilson, wilson as _wilson

# Every command takes one of these. A model walks its whole pipeline; a dataset
# is a single samplable dataset with no pipeline around it, and the layers that
# read rows cannot tell the difference (see registry.resolve).
TARGET_HELP = "model or dataset: " + ", ".join(registry.targets())


def _count_int(value: str) -> int:
    """argparse type for a count where zero is a real choice — `lookup --docs 0`
    asks for counts without documents. A negative one is not: it reaches the
    sampler as a negative budget, skips retrieval, and reports "0 documents from
    0 draws" for a phrase with thousands of occurrences."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {n}")
    return n


def _docs_arg(value: str) -> int | str:
    """argparse type for `lookup --docs`: a count, or `all` to walk every
    occurrence by rank instead of sampling ten at a time. The word rather than
    a separate flag, because "how many documents" is the one question and
    "all of them" is one of its answers."""
    if value.strip().lower() == "all":
        return "all"
    return _count_int(value)


def _positive_int(value: str) -> int:
    """argparse type for counts. Zero divides by zero deep inside the sampler and
    a negative one silently returns nothing; both should be a usage error."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {n}")
    return n


def _probe_words(value: str) -> int:
    """argparse type for `contaminate --words`. A probe shorter than
    `benchmarks.MIN_WORDS` matches by chance — a one-word window is a common
    word, and a chance match reads as contamination — so the floor `probe()`
    applies to items applies to the window too."""
    n = _positive_int(value)
    if n < benchmarks.MIN_WORDS:
        raise argparse.ArgumentTypeError(
            f"a probe needs at least {benchmarks.MIN_WORDS} words, got {n}: a shorter "
            "window matches by chance, and a chance match reads as contamination"
        )
    return n


def _draws(value: str) -> int:
    """argparse type for `bif --draws`. One retained draw is a series of one
    observation: centering it gives every covariance, correlation and partial
    exactly zero, and the file written would rank the candidates in an arbitrary
    tie and look like a result."""
    n = _positive_int(value)
    if n < 2:
        raise argparse.ArgumentTypeError(
            f"a covariance needs at least 2 retained draws, got {n}"
        )
    return n


def _positive_float(value: str) -> float:
    """argparse type for the sampler's real-valued settings. A negative or
    non-finite step size reaches `math.sqrt` inside the sampler after the
    checkpoint has been loaded; a negative γ makes the prior repulsive and a
    negative nβ drives the chain up the loss, and either writes a result for a
    distribution that is not the documented posterior."""
    x = float(value)
    if not (x > 0 and math.isfinite(x)):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value}")
    return x


def _nonnegative_int(value: str) -> int:
    """argparse type for `--examples`. Zero is meaningful — count without keeping
    any evidence — but a negative limit reaches the example heap with nothing in
    it and raises IndexError, after the multi-gigabyte scan has already run."""
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be zero or a positive integer, got {n}")
    return n


def _fmt_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    return f"{n:,}"


# Sentinel for "look the revision up now". Distinct from None, which is a
# caller saying it knows the revision and the answer is "unknown" — reusing an
# old result file that predates this field, say. Collapsing the two would stamp
# today's `main` onto rows drawn from a revision nobody recorded.
_RESOLVE = object()


def _stamp(dataset: str | None = None, revision=_RESOLVE) -> dict:
    """Provenance every result file carries: when it was written, and which
    commit of the dataset it was computed over.

    A dataset id alone does not identify what was counted. `main` moves — Ai2
    has republished these mixes — so a file reporting "8.1% harmlessness, n=300"
    without a revision cannot be checked later, or told apart from the same
    figure over different rows.

    Callers that draw rows resolve the revision *before* the draw and pass it
    here, so the stamp names the tree the rows actually came from; a lookup
    after a long labeling run could name a revision published while it ran. A
    lookup that fails records null rather than failing a run whose API calls are
    already paid for.
    """
    out = {"generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    if dataset:
        out["revision"] = hf.dataset_revision(dataset) if revision is _RESOLVE else revision
    return out


def _unlabeled_note(labels: list, reasons: dict[str, int]) -> str:
    """One line naming what the classifier never labeled, for stderr."""
    n = sum(1 for label in labels if label is None)
    if not n:
        return ""
    detail = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
    return f"  [{n} unlabeled ({detail or 'reason unrecorded'})]"


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
    """(index, row, prompt) for each row of a deterministic (sample, seed) draw
    that has one.

    Rows carrying no user prompt drop out here, so the result is usually shorter
    than `sample`. The row travels with its prompt because part of what an
    example teaches is in the row rather than the text — an RL row's verifier
    settles its taxonomy label outright.

    The index is the row's absolute position in the split, and it is what every
    result record stores to address its training example. Joining on the prompt
    instead cannot tell two rows apart that open with the same 400 characters,
    which is rare in a curated mix and routine in a chat log: 64 of WildChat's
    299 sampled prompts share an opening with another, 39 of them the same
    Midjourney prompt-generator preamble in front of 39 different conversations.
    """
    print(f"sampling {sample} rows from {stage['hf_dataset']} ...", file=sys.stderr)
    rows = hf.sample_rows_with_index(stage["hf_dataset"], sample, seed=seed)
    triples = (
        (i, r, extract.extract_prompt(r, stage["prompt_path"])) for i, r in rows
    )
    return [(i, r, p) for i, r, p in triples if p]


def _sample_prompts(stage, sample, seed):
    """(index, prompt), for the callers with no use for the row itself."""
    return [(i, p) for i, _, p in _sample_rows(stage, sample, seed)]


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


def _print_match_rate(stage, k, n, lo, hi, path, note=""):
    print(
        f"{stage}: {k}/{n} match = {k / n * 100 if n else 0:.1f}%"
        f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%) -> {path}{note}",
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
            route = registry.sample_route(s)
            how = "by shard" if route == "shards" else "in full, uniformly"
            line += f" — samplable {how} ({s['sample_dataset']})"
        print(line)
        if s.get("note"):
            print(f"    {s['note']}")


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


def _label_post_training(args, question=None, slug=None, stages=None):
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
    for s in stages or _select_stages(args, registry.post_training_stages, "post-training"):
        # Before the draw, not after: labeling 300 prompts takes minutes, and a
        # revision resolved at the end could name a tree published while it ran.
        revision = hf.dataset_revision(s["hf_dataset"])
        rows = _sample_rows(s, args.sample, args.seed)
        indices = [i for i, _, _ in rows]
        prompts = [p for _, _, p in rows]
        fixed = [
            classify.verifier_label(row, registry.stage_kind(s)) if question is None else None
            for _, row, _ in rows
        ]
        ask = [p for p, f in zip(prompts, fixed) if not f]
        settled = len(prompts) - len(ask)
        print(
            f"classifying {len(ask)} prompts with {args.classifier}"
            + (f" ({settled} labeled by their verifier)" if settled else "")
            + " ...",
            file=sys.stderr,
        )
        # A chat log is not a training example, so it is not judged as one.
        # Every model stage gets None here and takes the default rubric.
        system = classify.system_for(registry.stage_kind(s), question)
        asked_labels, reasons = classify.classify_prompts(
            ask, model=args.classifier, question=question, system=system
        )
        # The datasets-server takes no revision — /rows serves its own build of
        # whatever the dataset is now — so the stamp can only be the tree the
        # hub pointed at when the draw started. Read it again now that the slow
        # part is over: if it moved while this ran, the rows may straddle two
        # trees, and the file should say so rather than name one of them and
        # sound certain. Only this path checks, because only this path is slow
        # enough for the window to matter.
        moved = hf.dataset_revision(s["hf_dataset"])
        asked = iter(asked_labels)
        # A verifier-settled row is never None, so what stays unlabeled is what
        # the classifier was asked about and did not answer.
        labels = [f or next(asked) for f in fixed]
        note = _unlabeled_note(labels, reasons)
        run = {
            "dataset": s["hf_dataset"],
            **_stamp(s["hf_dataset"], revision=revision),
            "sample": args.sample,
            "seed": args.seed,
            "classifier": args.classifier,
            # The taxonomy (or the question) is the instrument: rewording a
            # label moves every share under it, so the file says which wording
            # produced these labels.
            "system_sha": classify.system_id(classify.build_system(question, system)),
            # Prompts the classifier never labeled, and why. Every rate here is
            # over the labeled ones, so this is the part of the sample those
            # rates do not describe.
            "unlabeled": sum(1 for label in labels if label is None),
            "unlabeled_reasons": reasons,
        }
        # `revision` is None when the pre-draw lookup failed, and a SHA now is
        # not evidence the tree moved — it is the first reading we got. Saying
        # so would also crash on revision[:7] and throw away a run that has
        # already been paid for.
        if revision and moved and moved != revision:
            run["revision_moved_to"] = moved
            print(
                f"  note: {s['hf_dataset']} moved from {revision[:7]} to"
                f" {moved[:7]} while this ran; rows may straddle both",
                file=sys.stderr,
            )
        if question is None:
            records = []
            for i, p, lab, f in zip(indices, prompts, labels, fixed):
                # `row` is what the site joins to the context record. See
                # _sample_rows: the prompt text is not a key.
                rec = {"row": i, "prompt": extract.clip(p), "label": lab}
                if f:
                    rec["by"] = "verifier"
                records.append(rec)
            path = _write_json(
                RESULTS / f"{args.target}.{s['stage']}.labels.json",
                {**run, "records": records},
            )
            print(f"{s['stage']}: {_counts(records)}  -> {path}{note}", file=sys.stderr)
        else:
            records = [
                {"row": i, "prompt": extract.clip(p), "match": lab == "yes"}
                for i, p, lab in zip(indices, prompts, labels)
                if lab
            ]
            path = _write_json(
                RESULTS / f"{args.target}.{s['stage']}.ask-{slug}.json",
                {"question": question, **run, "records": records},
            )
            k, n = sum(r["match"] for r in records), len(records)
            _print_match_rate(s["stage"], k, n, *_wilson(k, n), path, note)


def _label_pretrain_docs(args, question, slug, stages=None):
    """Score the documents `pretrain` wrote against `question`.

    Judged from that file rather than re-sampled, so asking a second question
    scores the same documents and costs nothing but the API call.
    """
    for s in stages or registry.pretrain_stages(registry.resolve(args.target)):
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
        labels, reasons = classify.classify_prompts(
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
                # What the interval clusters on: the unit this document was
                # drawn in, which is the shard for a shard-route sample and the
                # page of ten adjacent rows for a rows-route one. `shard` is the
                # fallback so a sample written before `cluster` existed — every
                # committed Olmo one — clusters exactly as it did before.
                "cluster": d.get("cluster") or d["shard"],
                # The document's true length, not the excerpt's. `budget` weighs
                # a corpus rate by length — a 200k-character long-context PDF is
                # not one 500-character web snippet's worth of training — and
                # carrying the number here means that rollup never has to reopen
                # the multi-megabyte document sample to find it.
                "chars": d["chars"],
            }
            for d, lab in zip(docs, labels)
            if lab
        ]
        k, n = sum(r["match"] for r in records), len(records)
        lo, hi, n_eff = _cluster_wilson(records, key="cluster")
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.ask-{slug}.json",
            {
                "question": question,
                "dataset": data["dataset"],
                # The revision the documents were sampled at, carried over from
                # the sample rather than looked up now: these documents came
                # from that tree, whatever `main` points at today. A sample
                # written before this field existed records null, not today's.
                **_stamp(data["dataset"], revision=data.get("revision")),
                # And where it went, when the draw outlasted a republish. The
                # sample records this; without carrying it here the site's
                # revision link on an ask card shows the starting SHA alone and
                # drops the warning that these documents may span two trees.
                # Re-detecting it now would be a different question — the window
                # that matters closed when the documents were drawn.
                **(
                    {"revision_moved_to": data["revision_moved_to"]}
                    if data.get("revision_moved_to")
                    else {}
                ),
                # How these documents were drawn, taken from the sample rather
                # than looked up in the registry when this run is read back.
                # `budget` weighs a rows-drawn rate by document length and a
                # shard-drawn one not at all, so a stage whose `sample_via`
                # changes after the fact would otherwise have its stored runs
                # silently reinterpreted under a design that did not produce
                # them. A sample written before `route` existed carries none,
                # and that rollup falls back to the registry as it did before.
                **({"route": data["route"]} if data.get("route") else {}),
                "stage": s["stage"],
                "sample": data["sample"],
                "seed": data["seed"],
                "classifier": args.classifier,
                "system_sha": classify.system_id(
                    classify.build_system(question, classify.ASK_DOC_SYSTEM)
                ),
                "unlabeled": sum(1 for label in labels if label is None),
                "unlabeled_reasons": reasons,
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
        _print_match_rate(s["stage"], k, n, lo, hi, path, _unlabeled_note(labels, reasons))


def _warn_missing_pretrain_samples(args, stages=None):
    """Warn before spending anything.

    The post-training stages cost an API call per batch, and finding out
    afterwards that the pretraining half had no sample to score is a slow way to
    learn it.
    """
    missing = [
        s["stage"]
        for s in stages or registry.pretrain_stages(registry.resolve(args.target))
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


# Characters of a derived name kept before the disambiguating hash. Well under
# the 255-byte basename limit once the model, stage and suffix are added.
MAX_SLUG_CHARS = 60


def _slug(text: str) -> str:
    """A filename-safe short name derived from a question or a search pattern.

    Two reductions lose enough to name a different run: text with no ASCII
    letters or digits — a Chinese phrase, a punctuation-only expression —
    reduces to nothing at all, and two long inputs can agree on their first 60
    characters. Either would write over an unrelated result file without saying
    so, so both get a hash of the original appended.

    Questions differing only in case or punctuation still share a file, which
    for prose is two spellings of one question. A regex is not prose — see
    `_pattern_slug`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    if not slug:
        return f"pattern-{digest}"
    if len(slug) > MAX_SLUG_CHARS:
        return f"{slug[:MAX_SLUG_CHARS].rstrip('-')}-{digest}"
    return slug


def _pattern_slug(
    pattern: str, case_sensitive: bool = False, regex: bool = False
) -> str:
    """A stable name for a pattern and the modes that change what it matches.

    Punctuation in a regex is syntax, not spelling: `a.b` and `a+b` match
    different text, and so does `a b` with one space or two. Case and literal
    versus regex mode matter the same way: the same text under either pair of
    modes is a different search. A literal keeps the plain slug only when it
    already *is* that slug and the search was case-insensitive; anything else
    carries a hash of the pattern and its mode. Pass `--slug` for a readable
    name.
    """
    base = re.sub(r"[^a-z0-9]+", "-", pattern.lower()).strip("-")
    # The readable shortcut still has to produce a filename: the empty pattern
    # is not a name, and a 300-character literal is its own slug but not a
    # basename any filesystem will take — which would spend the whole sampling
    # run and then fail on the write.
    if (
        base
        and pattern == base
        and len(base) <= MAX_SLUG_CHARS
        and not case_sensitive
        and not regex
    ):
        return base
    mode = f"{'cs' if case_sensitive else 'ci'}{'-regex' if regex else ''}"
    digest = hashlib.sha1(
        f"{pattern}\n{mode}".encode()
    ).hexdigest()[:8]
    if not base:
        return f"pattern-{digest}"
    return f"{base[:MAX_SLUG_CHARS].rstrip('-')}-{digest}"


def cmd_ask(args):
    """Score sampled examples against a free-form question, either half of the pipeline.

    Unlike every other command, `--stage` here selects across two families:
    post-training stages are sampled and judged as prompts, corpus stages are
    read out of a committed document sample and judged as documents. So the
    selection is resolved once, here, rather than by `_select_stages` inside
    each half — which would exit on `--stage pretrain` before the pretraining
    half ever ran.

    `--pretrain-only` exists because the two halves cost very different things.
    A question already answered over post-training should be extendable to the
    corpora — which is where 99% of the tokens are — without re-paying for nine
    stages of prompt labeling that are already committed.
    """
    # One short name ties a question's post-training and pretraining files together.
    slug = args.slug or _slug(args.question)
    print(f"question: {args.question}\n", file=sys.stderr)
    target = registry.resolve(args.target)
    want_pretrain = args.pretrain or args.pretrain_only
    post = [] if args.pretrain_only else registry.post_training_stages(target)
    pre = registry.pretrain_stages(target) if want_pretrain else []
    if want_pretrain and not registry.pretrain_stages(target):
        # A dataset has no corpora to score. Accepting the flag and quietly
        # scoring only the prompts would answer half the question asked.
        sys.exit(f"--pretrain: {args.target} has no pretraining stages")
    if not post and not pre:
        # A base model asked without --pretrain. There are no prompts anywhere
        # in its pipeline, so the run below would do nothing and exit 0 — the
        # same silent success `cmd_sources` used to hand this target. Name the
        # flag that makes the question answerable instead.
        sys.exit(
            f"{args.target} has no post-training stages; pass --pretrain to"
            " score its pretraining documents instead"
        )
    if args.stage:
        post = [s for s in post if s["stage"] == args.stage]
        pre = [s for s in pre if s["stage"] == args.stage]
        if not post and not pre:
            hint = "" if want_pretrain else " (pass --pretrain to reach a corpus stage)"
            sys.exit(f"no stage {args.stage!r} to ask about for {args.target}{hint}")
    if pre:
        _warn_missing_pretrain_samples(args, pre)
    if post:
        _label_post_training(args, question=args.question, slug=slug, stages=post)
    elif pre:
        # Say what is being skipped rather than letting a corpus-only run read
        # as a whole-pipeline answer.
        print(
            f"{args.target} has no post-training stages — scoring its"
            " pretraining documents only.",
            file=sys.stderr,
        )
    if pre:
        _label_pretrain_docs(args, args.question, slug, stages=pre)


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
    return paths.find(f"{target_name}.{stage}.docs.json")


def _pretrain_rows(args, s, dataset):
    """Sample a corpus the datasets-server has indexed in full.

    Returns the documents and the corpus facts to store alongside them. There is
    no shard listing here, so the composition is the registry's published one
    rather than a breakdown this run counted, and the site reads `route` to know
    not to claim a measured one it does not have.
    """
    print(f"sampling {args.sample} documents from {dataset} ...", file=sys.stderr)
    # Resolved before the draw, so the stamp names the tree the rows came from
    # rather than one published while the run was in flight.
    revision = hf.dataset_revision(dataset)
    docs, total = pretrain.sample_rows_documents(
        dataset, args.sample, seed=args.seed, text_column=s.get("text_column", "text")
    )
    # And again after it. Unlike the shard route — whose range requests name a
    # pinned revision in the URL, so its rows cannot straddle one — this route's
    # thirty /rows requests are served from whatever the tree is at the time. A
    # republish mid-draw leaves a sample split across two corpora under a single
    # SHA, which is the one thing the stamp is supposed to rule out. Same check,
    # same field name, and the same reason, as every other paged sampler here.
    moved = hf.dataset_revision(dataset)
    print(f"  {total:,} documents in the corpus", file=sys.stderr)
    # A SHA now with no SHA before is the first reading we got, not evidence of
    # a move — and `revision[:7]` below would raise on it.
    if revision and moved and moved != revision:
        print(
            f"  note: {dataset} moved from {revision[:7]} to {moved[:7]} while this"
            " ran; the documents may straddle both trees",
            file=sys.stderr,
        )
    return docs, {
        **_stamp(dataset, revision=revision),
        **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
        "route": "rows",
        "rows_total": total,
        "caveat": pretrain.rows_sampling_caveat(),
    }


def cmd_pretrain(args):
    """Sample documents from a stage's pretraining corpus.

    Two routes, chosen by `registry.sample_route`. Dolma 3 goes by shard: the
    datasets-server indexes only the first ~5 GB of those repos and the shards
    are topic-ordered, so this reads the repo files by range request instead.
    A corpus the server *has* indexed in full — the deduplicated Pile — is paged
    directly, which is both simpler and a better sample.

    Either way no model is called; this is the deterministic half, and
    `ask --pretrain` scores whatever it wrote.
    """
    for s in _select_stages(args, registry.pretrain_stages, "pretraining"):
        dataset = s["sample_dataset"]
        if registry.sample_route(s) == "rows":
            docs, corpus_facts = _pretrain_rows(args, s, dataset)
            _write_pretrain_docs(args, s, dataset, docs, corpus_facts)
            continue
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
        _write_pretrain_docs(
            args,
            s,
            dataset,
            docs,
            {
                # The exact commit the composition and documents came from.
                # "main" moves; a result file that cites exact byte shares
                # has to say which revision it counted.
                **_stamp(dataset, revision=revision),
                "route": "shards",
                "docs_per_shard": args.docs_per_shard,
                # Shard draws that contributed fewer documents than asked
                # for. Non-zero means the sample is weighted by reachable
                # document density as well as by size.
                "short_draws": short,
                "caveat": pretrain.sampling_caveat(args.docs_per_shard),
                "shards": len(shards),
                "bytes": total_bytes,
                "groups": groups,
            },
            note=f", {short} short draw(s) made up by others" if short else "",
        )


def _write_pretrain_docs(args, s, dataset, docs, corpus_facts, note=""):
    """Store one stage's document sample, whichever route drew it.

    The route-specific facts arrive already assembled in `corpus_facts` and are
    merged in whole, so a shard run keeps its shard count, byte total, group
    breakdown and pinned revision, and a rows run carries the corpus row count
    instead of pretending to any of them. `route` is what the site branches on.
    """
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
            # The correlated unit this document was drawn in, when it is not the
            # shard. Only the rows route sets one — the shard route's cluster is
            # its `shard`, and writing that value twice under two names would
            # give the next reader two places to keep in step.
            **({"cluster": d["cluster"]} if d.get("cluster") else {}),
            **({"row": d["row"]} if d.get("row") is not None else {}),
            # A cell the server shortened: `chars` is then the length of what
            # arrived, not of the document, and the site says so rather than
            # letting a clipped document read as a short one.
            **({"truncated": True} if d.get("truncated") else {}),
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
            "stage": s["stage"],
            "name": s["name"],
            "sample": len(records),
            "requested": args.sample,
            "seed": args.seed,
            "scope": s.get("sample_scope"),
            **corpus_facts,
            "records": records,
        },
    )
    print(
        f"{s['stage']}: {len(records)} documents -> {path}"
        f" ({path.stat().st_size / 1e6:.1f} MB)" + note,
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
            # A classify run written before result records carried their row
            # index has no row to reuse; those records keep joining to their
            # context by prompt text, as they did before.
            pairs = [(r.get("row"), r["prompt"]) for r in prior["records"]]
            sample, seed = prior["sample"], prior["seed"]
            # Reusing a run's prompts means reusing its revision: those rows came
            # from that tree. A file written before the field existed has none,
            # and null is the honest answer — not today's `main`, which is a
            # different tree from the one nobody recorded.
            revision = prior.get("revision")
            # And its ambiguity, if it had any. These are that run's rows, so a
            # language file that named only the first tree would state as settled
            # what the classification run recorded as unresolved.
            moved = prior.get("revision_moved_to")
            print(f"reusing {len(pairs)} prompts from {labels_path.name}", file=sys.stderr)
        else:
            revision = hf.dataset_revision(s["hf_dataset"])
            # Clip before detecting, not after. A classify run stores the clipped
            # prompt, so detecting the full text here would make --from-labels
            # disagree with this path on the handful of prompts past the cutoff.
            pairs = [
                (i, extract.clip(p))
                for i, p in _sample_prompts(s, args.sample, args.seed)
            ]
            # Detection is local, but the draw feeding it is thirty paged
            # requests, so this path has the same republish window as `context`.
            moved = hf.dataset_revision(s["hf_dataset"])
        records = []
        for i, p in pairs:
            code, conf = languages.detect(p)
            rec = {"prompt": p, "label": code, "confidence": round(conf, 3)}
            if i is not None:
                rec = {"row": i, **rec}
            records.append(rec)
        path = _write_json(
            RESULTS / f"{args.target}.{s['stage']}.languages.json",
            {
                "dataset": s["hf_dataset"],
                **_stamp(s["hf_dataset"], revision=revision),
                # From a reused run, this is the ambiguity it recorded; from a
                # fresh draw, movement observed across that draw. Either way it
                # takes two known and different readings: a first lookup that
                # failed leaves the tree unknown, and a SHA read afterwards is
                # the first answer we got, not evidence of a republish.
                **({"revision_moved_to": moved} if revision and moved and moved != revision else {}),
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


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _grep_plan(con, args, stages):
    """What each selected stage would cost to scan, before anything is read.

    Resolving the whole plan first is what lets the byte cap be one decision
    rather than a surprise partway through: a Think SFT mix is tens of gigabytes
    of message text where the DPO and RL mixes are one or two, and the difference
    only shows up here.
    """
    plan = []
    for s in stages:
        listing = grep.parquet_listing(s["hf_dataset"])
        schema = grep.schema(con, listing["urls"][0])
        exprs, _, unsearched = grep.text_fields(schema, args.field)
        # What the mix holds, as opposed to what this run reads. A result file
        # with only the second cannot say whether an absent response count means
        # `--field` narrowed the search or the mix has no response column, and
        # those are opposite readings of the same blank. Schema only, no reads.
        available, _, _ = grep.text_fields(schema, None)
        if not exprs:
            sys.exit(
                f"{s['stage']}: no text columns for field(s) {', '.join(args.field or grep.GROUPS)}"
                f" in {s['hf_dataset']}"
            )
        source, source_column = grep.source_expr(
            schema, [args.by] if args.by else s["source_columns"]
        )
        if args.by and not source:
            sys.exit(f"{s['stage']}: no text column {args.by!r} in {s['hf_dataset']}")
        leaves = grep.plan_leaves(schema, args.field, source_column)
        plan.append({
            "stage": s,
            "listing": listing,
            # Resolved before the scan, like every other layer that draws rows:
            # a lookup afterwards could name a revision published while a
            # multi-gigabyte read was in flight.
            "revision": hf.dataset_revision(s["hf_dataset"]),
            "schema": schema,
            "exprs": exprs,
            "source": source,
            "source_column": source_column,
            "unsearched": unsearched,
            "available": list(available),
            "rows": grep.total_rows(con, listing["urls"]),
            "bytes": grep.byte_cost(con, listing["urls"], leaves),
        })
    return plan


def cmd_grep(args):
    """Count rows of every post-training mix whose text contains a pattern.

    Exact, over all rows, which is the half of the question sampling cannot do.
    `classify` and `ask` estimate an unconditional rate from 300 prompts; a
    pattern that occurs in 0.1% of a mix is expected to miss such a sample
    entirely, and no interval around zero tells you it is there.
    """
    con = grep.connect()
    stages = _select_stages(args, registry.post_training_stages, "post-training")
    plan = _grep_plan(con, args, stages)

    total_bytes = sum(p["bytes"] for p in plan)
    print(f"# grep {args.pattern!r} — {len(plan)} stage(s), {_fmt_bytes(total_bytes)} to read\n", file=sys.stderr)
    for p in plan:
        fields = "/".join(p["exprs"])
        print(
            f"- {p['stage']['stage']:6s} {p['rows']:>9,} rows  {_fmt_bytes(p['bytes']):>9}"
            f"  {fields}  ({p['stage']['hf_dataset']})",
            file=sys.stderr,
        )
    cap = int(args.max_gb * 1e9)
    if total_bytes > cap and not args.yes:
        sys.exit(
            f"\nthat is {_fmt_bytes(total_bytes)}, over the {args.max_gb} GB cap, and nothing has been "
            f"read yet. Narrow it (--stage, --field) or allow it (--max-gb {total_bytes / 1e9:.1f}, or --yes)."
        )

    # `slugify` already yields a filename-safe slug, but an explicit `--slug`
    # is raw user input that lands in a path, and `_write_json` creates parent
    # directories: `--slug a/b` would quietly file a multi-gigabyte scan where
    # neither `_grep_traces` nor the site export looks, and `../..` would write
    # outside results/ entirely. Same treatment `find` gives its components.
    slug = _filename_part(args.slug) if args.slug else grep.slugify(args.pattern)
    written = []
    for p in plan:
        s = p["stage"]
        if p["listing"]["partial"]:
            print(
                f"\n{s['stage']}: WARNING — the server converted only part of this repo, "
                "so every count below is a lower bound",
                file=sys.stderr,
            )
        if p["unsearched"]:
            print(
                f"\n{s['stage']}: not searched: {', '.join(p['unsearched'])}"
                " — text columns this layer does not recognise as prompt, response or reference",
                file=sys.stderr,
            )
        print(f"\nscanning {s['stage']} ({_fmt_bytes(p['bytes'])}) ...", file=sys.stderr)
        result = grep.scan(
            con,
            grep.read_parquet_sql(p["listing"]["urls"]),
            p["exprs"],
            p["source"],
            args.pattern,
            regex=args.regex,
            case_sensitive=args.case_sensitive,
            examples=args.examples,
        )
        rows = p["rows"]
        k = result["matched"]
        print(f"{s['stage']}: {k:,}/{rows:,} rows = {k / rows * 100 if rows else 0:.3f}%", file=sys.stderr)
        for group, n in result["by_group"].items():
            print(f"  {group:10s} {n:,}", file=sys.stderr)
        # Unconditional, though a zero-match stage has no percentages to
        # compute. Skipping it when `k == 0` saved one read of a label column and
        # cost the plan its meaning: the priced bytes always included both source
        # reads, so a pattern absent from the mix was charged for a query that
        # never ran. Running it always makes the quoted figure the figure, and it
        # earns its keep — a stage with no matches still reports what the
        # denominators were, so "0 of 48,398 rlvr_general_mix rows" is sayable
        # rather than just "0".
        totals = (
            grep.source_totals(con, grep.read_parquet_sql(p["listing"]["urls"]), p["source"])
            if p["source"] else {}
        )
        shown = list(result["by_source"].items())[:12]
        for src, n in shown:
            of = totals.get(src)
            rate = f" = {n / of * 100:5.2f}% of it" if of else ""
            print(f"  {n:>7,} / {of or rows:>9,}{rate}  {src}", file=sys.stderr)
        if len(result["by_source"]) > len(shown):
            rest = len(result["by_source"]) - len(shown)
            print(f"  … and {rest} more source(s), all of them in the result file", file=sys.stderr)
        payload = {
            "dataset": s["hf_dataset"],
            "stage": s["stage"],
            "pattern": args.pattern,
            "slug": slug,
            "regex": args.regex,
            "case_sensitive": args.case_sensitive,
            "fields": list(p["exprs"]),
            "available_fields": p["available"],
            "source_column": p["source_column"],
            # `_stamp` carries `generated` and the *dataset* revision every
            # other result file records. The Parquet-branch revision is a
            # second, different tree — the server's conversion of that
            # dataset — so it travels under its own name rather than
            # overwriting the one the rest of the tool means by "revision".
            **_stamp(s["hf_dataset"], revision=p["revision"]),
            "parquet_revision": p["listing"]["revision"],
            "partial": p["listing"]["partial"],
            "shards": len(p["listing"]["urls"]),
            "bytes_read": p["bytes"],
            "unsearched_columns": p["unsearched"],
            "total_rows": rows,
            "rows_by_source": totals,
            **result,
        }
        path = _write_json(RESULTS / f"{args.target}.{s['stage']}.grep-{slug}.json", payload)
        written.append(payload)
        print(f"  -> {path}", file=sys.stderr)

    # The counts above are one mix each. Read together they are a claim about
    # where a string most plausibly entered the model, which is a different
    # question from how many rows hold it — and one that needs the stages
    # nobody scanned named as such rather than left out.
    target = registry.resolve(args.target)
    # Only for a model. A dataset target is a corpus, not a pipeline: nothing was
    # trained on WildChat's chat log, so ranking its one stage as where a phrase
    # entered a model produced "Most plausibly chat" — a training-origin verdict
    # about a target that has no training. The counts above are exactly as useful
    # for a corpus; it is the ranking that has nothing to rank.
    if target["is_model"]:
        trace = influence.compare(written, target["stages"])
        print("", file=sys.stderr)
        for line in influence.render(trace, args.target, note=True):
            print(line, file=sys.stderr)


# Checkpoints the exposure table prints. The file carries all 154; on a terminal
# five spread across the run say the shape, and the rest is a straight line.
_SHOWN_CHECKPOINTS = (1000, 10_000, 50_000, 100_000, 143_000)


def cmd_steps(args):
    """Count a pattern in sampled training batches, in the order the model saw them.

    Only for a target whose order is published, which is Pythia. Each sampled
    step is one 4.2 MB range request into the preshuffled token stream, decoded
    with the run's tokenizer and searched sequence by sequence. The output is a
    rate along the run, the same rate over equal slices of it, and what that rate
    says the model had seen by each saved checkpoint.
    """
    target = registry.resolve(args.target)
    found = registry.training_order(target)
    if not found:
        sys.exit(
            f"{args.target} has no published training order. This layer reads the batches in "
            "the sequence the optimizer took them, which of the registered targets only Pythia "
            "publishes; for Olmo the corpus is public and the order is not."
        )
    if not args.pattern:
        sys.exit("an empty pattern matches every position of every sequence")
    s, order = found
    steps.check_layout(order)
    try:
        # `--at` is purposive: it adds a batch for inspection, not another
        # random draw. Keep the probability sample separate so a checkpoint
        # selected precisely because it looks unusual cannot tilt the corpus
        # rate, its interval, or every checkpoint exposure estimate.
        sampled_steps = steps.draw_steps(order["steps"], args.sample, args.seed)
        picks = steps.draw_steps(order["steps"], args.sample, args.seed, args.at or ())
    except ValueError as e:
        sys.exit(str(e))
    rx = steps.compile_pattern(args.pattern, args.regex, args.case_sensitive)
    revision = pretrain.resolve_revision(order["dataset"])
    tokenizer_revision = steps.resolve_tokenizer_revision(order["tokenizer"])
    decode = steps.decoder(order, tokenizer_revision)
    to_read = steps.step_bytes(order) * len(picks)
    explicit = len(picks) - len(sampled_steps)
    read_description = f"{len(sampled_steps)} sampled"
    if explicit:
        read_description += f" + {explicit} explicit"
    print(
        f"# steps {args.pattern!r} — {read_description} of {order['steps']:,} steps, "
        f"{_fmt_bytes(to_read)} to read from {order['dataset']} at {revision[:7]}",
        file=sys.stderr,
    )

    def progress(i, n, step):
        print(f"\r  step {step:>7,}  ({i}/{n})   ", end="", file=sys.stderr, flush=True)

    per_step, examples = steps.scan(
        order,
        picks,
        revision,
        rx,
        decode,
        examples_limit=args.examples,
        progress=progress,
        priority_steps=args.at or (),
    )
    print(file=sys.stderr)

    sampled_set = set(sampled_steps)
    estimate_steps = [count for count in per_step if count["step"] in sampled_set]
    summary = steps.summarize(estimate_steps)
    slices = steps.by_slice(
        estimate_steps, order["steps"], args.slices, deff=summary["design_effect"]
    )
    exposure = steps.exposure(summary, order["checkpoints"], order["sequences_per_step"])
    second = steps.second_pass_step(order)

    n, k = summary["sequences"], summary["matched"]
    print(
        f"{s['stage']}: {k:,}/{n:,} sequences hold it = {k / n * 100 if n else 0:.3f}%"
        f" (95% CI {summary['lo'] * 100:.3f}–{summary['hi'] * 100:.3f}%),"
        f" {summary['occurrences']:,} occurrences, {len(estimate_steps)} sampled steps",
        file=sys.stderr,
    )
    print(f"  by stretch of the run ({args.slices} slices of {order['steps']:,} steps):", file=sys.stderr)
    for sl in slices:
        if sl["sequences"]:
            rate = (
                f"{sl['matched']:>5,}/{sl['sequences']:>7,} = {sl['rate'] * 100:6.3f}%"
                f"  ({sl['lo'] * 100:.3f}–{sl['hi'] * 100:.3f}%)  {sl['steps']} step(s)"
            )
        else:
            rate = "no step sampled"
        print(f"    {sl['from_step']:>7,}–{sl['to_step']:<7,}  {rate}", file=sys.stderr)
    if second:
        print(
            f"  from about step {second:,} the run is re-reading the corpus"
            f" (~{_fmt_tokens(order['corpus_tokens'])} tokens against a"
            f" {_fmt_tokens(_run_tokens(order))} budget)",
            file=sys.stderr,
        )
    print("  expected sequences holding it, seen by checkpoint (if the rate holds along the run):", file=sys.stderr)
    shown = {e["step"]: e for e in exposure}
    for step in _SHOWN_CHECKPOINTS:
        e = shown.get(step)
        if e:
            print(
                f"    step {step:>7,}   ~{_fmt_est(e['expected']):>5}   ({_fmt_est(e['lo'])}–{_fmt_est(e['hi'])})",
                file=sys.stderr,
            )
    for ex in examples[:3]:
        flat = " ".join(ex["snippet"].split())
        print(f"  step {ex['step']:,} seq {ex['sequence']}: …{flat}…", file=sys.stderr)

    slug = (
        _filename_part(args.slug)
        if args.slug
        else _pattern_slug(args.pattern, args.case_sensitive, regex=args.regex)
    )
    payload = {
        "dataset": order["dataset"],
        "stage": s["stage"],
        "name": s["name"],
        "pattern": args.pattern,
        "slug": slug,
        "regex": args.regex,
        "case_sensitive": args.case_sensitive,
        **_stamp(order["dataset"], revision=revision),
        "tokenizer": order["tokenizer"],
        "tokenizer_revision": tokenizer_revision,
        "steps_total": order["steps"],
        "sequence_tokens": order["sequence_tokens"],
        "sequences_per_step": order["sequences_per_step"],
        "sample": len(sampled_steps),
        "seed": args.seed,
        "at": sorted(set(args.at or [])),
        "bytes_read": to_read,
        "second_pass_step": second,
        "caveat": steps.CAVEAT,
        **summary,
        "slices": slices,
        "exposure": exposure,
        "per_step": per_step,
        "examples": examples,
    }
    path = _write_json(RESULTS / f"{args.target}.{s['stage']}.steps-{slug}.json", payload)
    print(f"  -> {path}", file=sys.stderr)


def _run_tokens(order: dict) -> int:
    """The run's token budget as the registry counts it: new tokens per step,
    which is one fewer than a sequence holds."""
    return order["steps"] * order["sequences_per_step"] * (order["sequence_tokens"] - 1)


def _phrase_slug(phrase: str) -> str:
    """A filename-safe slug that two different phrases cannot share.

    Distinctness is the requirement, because the slug names the result file
    and a collision silently overwrites an earlier search — and an exact-match
    search distinguishes phrases the normalization folds together ("Climate
    change" and "climate change" tokenize differently and have different
    counts). Normalization is lossy several ways at once: case and punctuation
    fold, non-ASCII drops entirely, length truncates at 60. Rather than
    enumerating which lossy path applied, every derived slug carries a hash of
    the exact phrase; `--slug` is there to pick a fully readable name instead.
    """
    digest = hashlib.sha1(phrase.encode()).hexdigest()
    norm = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")[:60].rstrip("-")
    return f"{norm}-{digest[:8]}" if norm else digest[:12]


def _filename_part(part: str) -> str:
    """One user-supplied component of a result filename, made separator-free.

    Not a security boundary — it is the user's own results/ directory — but a
    stray "/" in --slug or --index would scatter files outside it, away from
    where the user and the site look for results. Squash anything that is not
    a plain filename character.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("._-") or "x"


def _viewer_search_url(dataset: str, query: str, split: str = "train") -> str:
    """The Hub viewer showing the rows a full-text query matched.

    The viewer's `?q=` runs the same datasets-server index `hf.search_count`
    counts with, so this lands on the rows behind a count rather than on a
    sample that might contain one. That distinction is why `trace` links here
    instead of at `trainspotting search`, which draws 300 random rows and so
    finds none of the matches for the rare strings a trace is made of.
    """
    return (
        f"{hf.HUB}/datasets/{dataset}/viewer/default/{split}"
        f"?q={urllib.parse.quote(query)}"
    )


def cmd_find(args):
    """Exact-string search over an open training corpus, via infini-gram.

    The inverse of `ask`: instead of sampling documents and judging each one,
    take a string the caller already has and get its exact occurrence count
    plus example documents. The count doubles as a duplication count, which
    matters on its own — memorization scales with how many times a string
    appears in training. No model is called and nothing is downloaded; both
    steps are API lookups against a prebuilt suffix-array index.
    """
    covers = infinigram.INDEXES.get(args.index, "not in the known-index list; passed through as-is")
    print(f"index: {args.index} — {covers}", file=sys.stderr)
    caveat = infinigram.caveat_for(args.index)
    if caveat:
        print(f"note: {caveat}", file=sys.stderr)
    print(file=sys.stderr)

    found = infinigram.find(args.index, args.phrase)
    count = found["cnt"]
    # What was matched is the token sequence, not the raw string — say so, so a
    # surprising count can be traced to a surprising tokenization.
    print(f"matched as tokens: {' | '.join(found['tokens'])}")
    print(f"occurrences: {count:,}")

    # Ranks count occurrences, not documents: a phrase repeated within one
    # document holds several ranks, all resolving to the same doc_ix, so picks
    # are deduplicated by doc_ix while fetching. The first pass asks for
    # exactly --docs evenly spread picks; each shortfall re-asks at double the
    # resolution, which refines the spread without moving it (every pick at k
    # recurs at 2k), so retries visit new, still-evenly-spread ranks whether
    # or not the match count clamped the draw. The loop ends when enough
    # distinct documents are in hand, every match has been tried, or the
    # lookup budget is spent — bounded so a million-fold-duplicated string
    # costs a bounded number of API calls, not a crawl of them all.
    docs, seen, tried = [], set(), set()
    budget = 10 * args.docs
    k = args.docs
    while count and len(docs) < args.docs and len(tried) < budget:
        picks = [
            p
            for p in infinigram.spread_picks(found["segment_by_shard"], k)
            if p not in tried
        ]
        if not picks:
            break  # every match has been tried
        for s, rank in picks:
            if len(docs) >= args.docs or len(tried) >= budget:
                break
            tried.add((s, rank))
            doc = infinigram.get_doc(args.index, args.phrase, s, rank, args.maxlen)
            # doc_ix numbers a document within its suffix-array shard, so the
            # shard belongs in the identity — two documents in different
            # shards may share a doc_ix without being the same document.
            if (s, doc.get("doc_ix")) in seen:
                continue
            seen.add((s, doc.get("doc_ix")))
            prov = infinigram.doc_provenance(doc)
            docs.append(
                {
                    "index_shard": s,
                    "doc_ix": doc.get("doc_ix"),
                    "doc_len": doc.get("doc_len"),
                    "blocked": doc.get("blocked", False),
                    **prov,
                    "snippet": infinigram.snippet(doc),
                }
            )
        k *= 2
    for d in docs:
        where = " ".join(str(d[k]) for k in ("source", "path", "url") if d.get(k))
        print(f"\n--- doc {d['doc_ix']} ({d['doc_len']:,} tokens) {where}")
        print(d["snippet"] if not d["blocked"] else "[blocked by the index owner]")
    if count and len(docs) < args.docs:
        # The budget ran out or the matches did — either way, say what was
        # actually retrieved rather than letting --docs claim a size the
        # output does not have.
        print(
            f"\n(asked for {args.docs} documents; {len(tried):,} lookups"
            f" yielded {len(docs)} distinct ones)"
        )
    elif count > len(docs):
        print(
            f"\n({len(docs)} distinct documents shown out of {count:,} occurrences,"
            " spread evenly across the index)"
        )

    if args.json:
        slug = args.slug or _phrase_slug(args.phrase)
        # The index is part of what was measured — the same phrase has a
        # different count in every corpus — so it belongs in the filename,
        # or comparing indexes would overwrite one result with the next.
        path = _write_json(
            RESULTS / f"find.{_filename_part(args.index)}.{_filename_part(slug)}.json",
            {
                # The index name plays the role `dataset`+`revision` play in
                # the other result files: infini-gram indexes are immutable
                # builds, so the name alone says what was counted.
                **_stamp(),
                "phrase": args.phrase,
                "index": args.index,
                "covers": covers,
                "caveat": caveat,
                "tokens": found["tokens"],
                "count": count,
                "docs": docs,
            },
        )
        print(f"\nwrote {path}", file=sys.stderr)



def _grep_traces(model_name, model):
    """Committed `grep` runs for one model, grouped by the search they ran.

    The slug is where the stages of one sweep line up, and it has to be, because
    a pattern too long to name itself is stored under a `--slug`. But a slug is a
    filename rather than a promise: rerun one stage with a refined regex under
    the same slug and the directory holds two different searches with one name.
    So the group key is the slug *and* the search definition, and a slug that
    turns out to name more than one gets each rendered separately rather than
    ranked against each other under whichever pattern sorted first.
    """
    groups = {}
    for path in sorted(RESULTS.glob(f"{model_name}.*.grep-*.json")):
        slug = path.name.split(".grep-", 1)[1][: -len(".json")]
        run = json.loads(path.read_text())
        # The filename wins over the recorded slug, which is only a note of what
        # was passed. Grouping keys on the filename and `--slug` is what decides
        # the filename, so a rerun needs the name to land back in this group —
        # and after a collision rename the payload still carries the contested
        # slug it was moved away from.
        run["slug"] = slug
        key = (slug, run.get("pattern"), bool(run.get("regex")), bool(run.get("case_sensitive")))
        groups.setdefault(key, []).append(run)
    # Collision is a property of one slug, not of the directory: marking every
    # trace because some other slug is contested would strip a valid `--slug`
    # from commands that need it to land in their own group.
    per_slug = {}
    for slug, *_ in groups:
        per_slug[slug] = per_slug.get(slug, 0) + 1
    taken = {slug for slug, *_ in groups}

    out = []
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        slug = key[0]
        trace = influence.compare(runs, model["stages"])
        trace["slug_collides"] = per_slug[slug] > 1
        if trace["slug_collides"]:
            # Dropping `--slug` is not enough: two searches differing only in
            # `--regex` or `--case-sensitive` share a pattern, so `grep` would
            # derive the same filename for both. Hand out a free one instead,
            # skipping any slug already on disk.
            n = 1
            while f"{slug}-{n}" in taken:
                n += 1
            trace["slug_suggest"] = f"{slug}-{n}"
            taken.add(trace["slug_suggest"])
        out.append((slug, trace["slug_collides"], trace))
    return out

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


def _fmt_est(n: float | None) -> str:
    """An estimated token count, at the resolution the estimate supports.

    Three significant figures at most: these come from a 300-draw rate times a
    mean length, and printing 41,283,915 would claim precision the sample has
    nowhere near.
    """
    if n is None:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            v = n / scale
            text = f"{v:.0f}" if v >= 100 else f"{v:.1f}".removesuffix(".0")
            return text + suffix
    return f"{n:.0f}"


# Why the rate column is not one rule. The correction that is right for one
# sampling design is a double count under another, and which applies is a
# property of how a stage was *drawn* rather than of what kind of stage it is —
# a corpus the datasets-server indexes in full is paged uniformly over documents
# and takes the same length weighting a post-training mix takes. Keyed by the
# `weighting` string `budget` records, longest prefix first, so the table
# explains the rules it actually used and no others.
_WEIGHTING_RULES = [
    (
        "fit characters — rows",
        "A corpus paged uniformly over documents is weighed by fit characters:\n"
        "without that its rate is a share of documents rather than of training.",
    ),
    (
        "fit characters",
        'Post-training rows are drawn uniformly, so their rate is weighed by fit\n'
        'characters — otherwise it answers "what fraction of examples" rather than\n'
        '"what fraction of training".',
    ),
    (
        "none",
        "Corpus documents drawn by shard come from shards drawn with probability\n"
        "proportional to size, which already weights by tokens, so their document rate\n"
        "is used unchanged; weighing it by length would apply that a second time.",
    ),
    (
        "document count",
        "One corpus stage stores no document lengths to weigh by, so its unweighed\n"
        "document rate reads its matches as if every document were the same size.",
    ),
]

_WEIGHTING_FOOTNOTE = """
Fit tokens are what the model was trained to produce, at {cpt:g} characters per token.
Rate: {rules}
This weighs tokens, not learning: a post-training token and a pretraining token
are not equally formative, and nothing here corrects for that."""


def _weighting_footnote(est: dict) -> str:
    """The rate-column footnote, naming only the weightings this estimate used."""
    used = {s["weighting"] for s in est["stages"] if s.get("weighting")}
    rules = []
    for prefix, text in _WEIGHTING_RULES:
        if any(w.startswith(prefix) for w in used) and text not in rules:
            rules.append(text)
            used = {w for w in used if not w.startswith(prefix)}
    return _WEIGHTING_FOOTNOTE.format(
        cpt=budget.CHARS_PER_TOKEN, rules=" ".join(rules) if rules else "n/a"
    )


def _warn_mixed_questions(est: dict) -> bool:
    """Say so, on stdout, when a slug covers more than one wording — and report
    whether it does, so callers can withhold the total.

    A slug is not a question: `--slug` takes any string and a generated one is
    truncated to 60 characters, so stages sharing a slug can have been scored
    against different words. Summing them produces a number no single question
    ever measured. The warning goes to stdout with the table rather than to
    stderr beside it, because the table is what gets piped into a document and
    the caveat has to travel with it.
    """
    if not est.get("mixed"):
        return False
    variants = est.get("question_variants") or []
    judges = est.get("classifiers") or []
    conflict = est.get("rubric_conflict") or []
    # Three ways to be mixed, and they read very differently to someone deciding
    # whether the withheld total was worth withholding — so say which it was.
    what = []
    if len(variants) > 1:
        what.append(f"{len(variants)} different wordings of the question")
    if len(judges) > 1:
        what.append(f"{len(judges)} different classifiers ({', '.join(judges)})")
    if conflict:
        what.append(f"a rubric that changed between {', '.join(conflict)} stages")
    if not what:
        what.append("different judging instruments")
    print(
        f"WARNING: the stages under slug {est['slug']!r} were scored with"
        f" {' and '.join(what)}, so they do not add up to one measurement."
        " No total is shown.\n"
    )
    for q in variants:
        print(f"  - {q}")
    print()
    return True


def _share_phrase(t: dict) -> str:
    """The whole-pipeline share, said as precisely as it is true.

    Three cases, and only one of them is a bound:

    - every stage sized, every stage measured — the share, flat.
    - every stage sized, some unmeasured — a genuine lower bound. Those stages
      are already in the denominator, so measuring one can only add matches.
    - some stage unsized — not a bound in either direction. `totals()` drops an
      unsized stage from the denominator *and* the numerator, so sizing it later
      moves both, and if its own rate is below this aggregate the share falls.
      Saying "at least" there is arithmetic nobody can defend.
    """
    pct = _fmt_share(t["share"])
    if t["unsized"]:
        return f"{pct} of the {_fmt_est(t['size_tokens'])} that could be sized"
    return f"at least {pct}" if t["measured"] < t["stages"] else pct


def _fmt_share(share: float) -> str:
    """A share as a percentage, with enough digits to be a number.

    A question answered only over post-training is a rounding error against
    5.93T pretraining tokens, and "0.00%" would read as "none" rather than as
    the three-orders-of-magnitude gap that is the actual finding.
    """
    pct = share * 100
    return f"{pct:.2f}%" if pct >= 0.01 or pct == 0 else f"{pct:.2g}%"


BUDGET_COLS = f"{'stage':<14}{'fit tokens':>11}  {'sampled':>13}  {'rate':>9}  {'matching tokens':>17}"


def _budget_row(s: dict) -> str:
    size = _fmt_est(s.get("size_tokens")) + ("*" if s.get("size_is_floor") else " ")
    if not s.get("measured"):
        # "never asked" and "asked, but nothing came back that could be weighed"
        # are different facts about the stage, and collapsing them would read as
        # a gap in the run rather than a gap in the data.
        why = s.get("unusable") or "not measured"
        return f"{s['stage']:<14}{size:>12}  {why}"
    sampled = f"{s['matched']}/{s['n']}  {s['count_rate'] * 100:4.1f}%"
    matching = _fmt_est(s.get("matching_tokens"))
    ci = s.get("matching_tokens_ci")
    if ci:
        matching += f" ({_fmt_est(ci[0])}–{_fmt_est(ci[1])})"
    # `rate` is the estimator the stage's sampling design calls for, which is
    # not the same rule for both halves of the pipeline — see the footnote.
    return (
        f"{s['stage']:<14}{size:>12}  {sampled:>13}  {s['rate'] * 100:8.1f}%"
        f"  {matching:>17}"
    )


# Exit code for "there is nothing here to add up yet" — no ask run, or only
# unusable ones. Distinct from 1 because an uncaught exception also exits 1, and
# a caller that tolerates a missing measurement must not thereby tolerate a
# traceback. `scripts/human_life_value.sh` is that caller.
NO_MEASUREMENT = 3


def cmd_budget(args):
    """Roll an `ask` question up into a share of the training budget.

    Reads committed runs only. Every stage the question was never asked of
    prints "not measured" rather than dropping out, because a total that
    silently excludes 5.93T tokens of pretraining is the exact error this
    command exists to prevent.
    """
    est = budget.estimate(args.target, args.slug)
    measured = [s for s in est["stages"] if s.get("measured")]
    if not measured:
        # An artifact that exists and cannot be used is not a missing artifact,
        # and telling someone to re-run the command that produced it sends them
        # in a circle. Each stage already recorded why it failed; print that.
        unusable = [s for s in est["stages"] if s.get("unusable")]
        if unusable:
            print(
                f"every ask run for {args.target} under slug {args.slug!r} is unusable:",
                file=sys.stderr,
            )
            for s in unusable:
                print(f"  {s['stage']}: {s['unusable']}", file=sys.stderr)
            for s in unusable:
                for note in s.get("notes", []):
                    print(f"  {s['stage']}: {note}", file=sys.stderr)
            sys.exit(NO_MEASUREMENT)
        print(
            f"no ask run for {args.target} with slug {args.slug!r}"
            f" — run `trainspotting ask {args.target} \"...\" --slug {args.slug}`",
            file=sys.stderr,
        )
        sys.exit(NO_MEASUREMENT)
    print(f"# Training budget — {args.target}\n")
    mixed = _warn_mixed_questions(est)
    if not mixed:
        print(f"question: {est['question']}\n")

    print(BUDGET_COLS)
    print("-" * len(BUDGET_COLS))
    for s in est["stages"]:
        print(_budget_row(s))

    print()
    for family, label in (("pretrain", "pretraining"), ("post-training", "post-training"), ("all", "whole pipeline")):
        t = est["totals"][family]
        if not t["stages"]:
            continue
        line = f"{label:<16}{_fmt_est(t['size_tokens']):>10} fit tokens"
        # A measured stage that could not be sized is dropped from both the
        # denominator and the matching sum, so the share is as partial as an
        # unasked one — `unsized` has to count here too.
        partial = t["measured"] < t["stages"] or bool(t["unsized"])
        if mixed:
            line += "  →  no single total (see above)"
        elif t["measured"]:
            # The denominator is every sized stage, asked or not, so with one
            # still unasked this is a lower bound on the whole pipeline — not a
            # share of the part that was measured. Saying "at least" is the
            # difference between a number and a wrong number: the corpora are
            # 99.7% of these tokens. `_share_phrase` also knows when it is not
            # a bound at all.
            line += f"  →  {_fmt_est(t['matching_tokens'])} matching  ({_share_phrase(t)})"
        if partial:
            line += f"  [{t['stages'] - t['measured']} stage(s) not measured"
            # Naming the measured size is what stops the share above reading as
            # a share of the whole thing — but with nothing measured at all,
            # "0 of it was" is noise on top of "not measured".
            line += (
                f"; {_fmt_est(t['measured_size_tokens'])} of it was]"
                if t["measured"]
                else "]"
            )
        print(line)

    floors = est["totals"]["all"]["floor"]
    unsized = est["totals"]["all"]["unsized"]
    if floors:
        print(
            f"\n* {', '.join(floors)} sized at one reference rollout per prompt — a floor."
            " The rollouts the policy was actually fit to are not in the published mix."
        )
    if unsized:
        print(f"\nno size for: {', '.join(unsized)} — excluded from every total above")
    notes = [(s["stage"], n) for s in est["stages"] for n in s.get("notes", [])]
    if notes:
        print()
        for stage, note in notes:
            print(f"note ({stage}): {note}")
    print(_weighting_footnote(est))
    if args.json:
        path = _write_json(RESULTS / f"{args.target}.budget-{args.slug}.json", est)
        print(f"\nwrote {path}", file=sys.stderr)


def cmd_trace(args):
    """Turn an observed behavior into searches and rank stages by where it hits.

    The entry point for someone who has a transcript, not a grep string: paste
    the text the model produced, and this pulls the distinctive phrases out of
    it, counts how many rows of each post-training stage contain each phrase
    (full-text search over the whole split, nothing sampled and nothing
    downloaded), and ranks the stages by how densely the behavior appears.

    This is provenance by near-verbatim match, so it only answers when the
    behavior left a distinctive string. When it finds nothing — the phrases are
    generic, or the behavior is a disposition with no signature words — the fix is
    `trainspotting ask`, which judges the meaning of sampled examples instead of
    matching their text. The closing line says so.

    Three things a count here is not. The server's index stops at the first
    5 GB of a split, which the two 36 GB Think SFT mixes are well past, so those
    stages are reported beside the ranking as lower bounds rather than placed in
    it. The match is a stemmed AND over the query's tokens, not the literal
    phrase, so the count is an upper bound on verbatim occurrences. And it
    counts rows, not sides — so a run ends by linking the matched rows in the
    dataset viewer, which runs the same index, rather than at `trainspotting
    search`, whose 300-row draw finds none of the matches for a string this
    rare.
    """
    text = sys.stdin.read() if args.text == "-" else args.text
    queries = behavior.distinctive_ngrams(text, max_queries=args.max_queries)
    if not queries:
        sys.exit(
            "no distinctive phrase to search on — no window of this text is"
            " anchored on a number, a name-shaped token, or a mid-sentence"
            " capital.\nTry `trainspotting ask` to judge what sampled examples"
            " teach instead of matching their text."
        )
    print("searching for:", file=sys.stderr)
    for q in queries:
        print(f"  {q!r}", file=sys.stderr)
    print(file=sys.stderr)

    results = []
    for s in _select_stages(args, registry.post_training_stages, "post-training"):
        # Before the row count, like every other paged path, and checked again
        # after: a trace holds a stage open longer than any of them, because a
        # cold split spends minutes building its index before answering. If
        # `main` moves in that window, the matches and the row count they are
        # divided by describe different trees, and the density is a ratio
        # between two datasets.
        revision = hf.dataset_revision(s["hf_dataset"])
        total = hf.num_rows(s["hf_dataset"])
        per_query, partial = {}, False
        for q in queries:
            print(f"  {s['stage']}: searching {q!r} ...", file=sys.stderr)
            per_query[q], q_partial = hf.search_count(s["hf_dataset"], q)
            partial = partial or q_partial
        moved = hf.dataset_revision(s["hf_dataset"])
        # Summed across queries, so a row matching two of them counts twice —
        # this ranks where the behavior concentrates, it is not a distinct-row
        # count. The per-query lines below let a single dominant phrase be told
        # apart from a genuinely dense stage.
        hits = sum(per_query.values())
        results.append(
            {
                "stage": s["stage"],
                "dataset": s["hf_dataset"],
                "revision": revision,
                "revision_moved_to": moved if revision and moved and moved != revision else None,
                "total": total,
                "hits": hits,
                "density": hits / total * 1e6 if total else 0.0,
                # The server indexed only the first 5 GB of this split, so the
                # count is over a prefix of it while `total` is the whole thing.
                "partial": partial,
                "per_query": per_query,
            }
        )

    # Three groups, by what each stage's number is worth rather than by size.
    #
    # `ranked` is an exact density over the whole split. `bounded` is a lower
    # bound: the index stopped at 5 GB, so a 36 GB mix with the behavior all
    # through it can print a smaller figure than a small mix with none of it,
    # and ordering the two together would name the wrong stage as the place to
    # look. `crossed` is worse than either — the dataset was republished
    # between the row count and the searches, so the ratio's halves describe
    # different trees and it estimates nothing at all. A bound is at least
    # directionally true; a crossed figure has no known relation to any real
    # quantity, so it is reported for what it is and kept out of the ranking
    # and the recommendation both.
    by_rank = lambda r: -r["density"]  # noqa: E731
    crossed = sorted((r for r in results if r["revision_moved_to"]), key=by_rank)
    rest = [r for r in results if not r["revision_moved_to"]]
    ranked = sorted((r for r in rest if not r["partial"]), key=by_rank)
    bounded = sorted((r for r in rest if r["partial"]), key=by_rank)

    def show(r):
        if r["revision_moved_to"]:
            # No density, and no "in N rows" either: that row count is not this
            # figure's denominator, it was read off a different tree. Printing
            # the ratio anyway had the section text contradicting its own
            # stages one line further down.
            print(
                f"## {r['stage']} — no density"
                f"  ({r['hits']} matches, split moved mid-search, {r['dataset']})"
            )
            print(
                f"  republished mid-search: {r['revision'][:7]} ->"
                f" {r['revision_moved_to'][:7]}"
                + ("; only the first 5 GB is indexed" if r["partial"] else "")
            )
        else:
            print(
                f"## {r['stage']} — {'≥' if r['partial'] else ''}{r['density']:.1f}/M"
                f"  ({r['hits']} matches in {r['total']:,} rows, {r['dataset']})"
            )
        for q, c in sorted(r["per_query"].items(), key=lambda kv: -kv[1]):
            # `!r`, like the stderr echo above: a query is a slice of text the
            # user pasted, and an escape sequence survives tokenization (`\x1b`
            # is not whitespace and `\w+` matches the rest of the sequence), so
            # printing it raw would let a transcript recolour or overwrite the
            # report that is quoting it back.
            print(f"  {c:>7,}  {q!r}")
        print()

    print(f"\n# Behavior trace: {args.target}\n")
    if ranked:
        print("Stages ranked by matches per million rows.\n")
        for r in ranked:
            show(r)
    if bounded:
        print("## Not ranked: only part of these splits is indexed\n")
        print(
            "The server's full-text index stops at the first 5 GB of a split, so"
            " the matches below are from a prefix of the rows they are divided"
            " by. Each figure is a lower bound: the true density is at least"
            " this, which settles the comparison against a ranked stage whose"
            " exact density is smaller and settles nothing against a larger one,"
            " because the rows nobody searched could hold any number of"
            " matches.\n"
        )
        for r in bounded:
            show(r)
    if crossed:
        print("## Not ranked: these splits were republished mid-search\n")
        print(
            "Each stage's row count was read before its searches and the"
            " dataset moved before they finished, so the two halves of the"
            " ratio describe different trees and it is not an estimate of"
            " anything. The match counts below are what the run saw, so they"
            " are printed; the density they would have been divided into is"
            " not, and neither is a place in the ranking. Re-run these"
            " stages.\n"
        )
        for r in crossed:
            show(r)

    # Only stages whose figure means something. `crossed` is excluded outright:
    # a number with no known relation to the split cannot recommend where to
    # read, and a zero from one cannot say the phrases are absent either.
    found = [r for r in ranked + bounded if r["hits"]]
    if not found:
        print(
            # Only stages searched end to end can support the flat claim. With a
            # bound or a crossed stage in the run the headline has to be the
            # weaker one, or it contradicts the caveats printed under it.
            (
                "No stage that was searched in full contained these phrases."
                if bounded or crossed
                else "No stage contained these phrases."
            )
            + " The behavior may be a disposition"
            " with no signature string, or the phrases may be paraphrased in"
            " training.\nTry `trainspotting ask` to judge what sampled examples"
            " teach — it reads meaning, not verbatim text."
            # A zero over an indexed prefix is not a zero over the split, so it
            # cannot join the others in a flat "nothing here".
            + (
                "\nNote that "
                + ", ".join(r["stage"] for r in bounded)
                + " was only partly indexed, so the phrases could be in the part"
                " that was never searched."
                if bounded
                else ""
            )
            + (
                "\nAnd "
                + ", ".join(r["stage"] for r in crossed)
                + " went unmeasured, having been republished mid-search — this"
                " says nothing about those stages either way."
                if crossed
                else ""
            )
        )
    else:
        # The largest number, bounds included. A bound *above* every exact
        # density is the one comparison a bound settles — the true density is at
        # least that, so the stage really is the densest — and dropping it here
        # would point at a stage this run has evidence is not the leader. It is
        # a bound *below* an exact figure that says nothing, and that one loses
        # the max anyway.
        top = max(found, key=lambda r: r["density"])
        print(
            f"Read the rows behind {top['stage']} in the dataset viewer, which"
            " searches the same index:\n"
            f"  {_viewer_search_url(top['dataset'], max(top['per_query'], key=top['per_query'].get))}"
        )
        # Not `trainspotting search`: it draws 300 random rows, so at the
        # densities a signature string produces — 100/M is a 3% chance of one
        # hit in that draw — it answers "how common is this" and almost never
        # "here is one". Pointing at it for a rare phrase would send the reader
        # to a confident zero.
        print(
            "`trainspotting search` reports which side of an example a hit lands"
            " on, but over a 300-row random draw rather than the matched rows, so"
            " for a phrase this rare it will usually find none of them."
        )
        # `top` is the largest number, which is only the largest *density* when
        # every stage was fully indexed. Say so rather than letting a bound that
        # happens to sort lower read as a stage with less of the behavior.
        others = [r["stage"] for r in bounded if r["hits"] and r is not top]
        if others:
            print(
                "Worth reading either way: " + ", ".join(others) + " reported a"
                " lower bound, so the true density there may be higher than"
                " anything ranked above it."
            )
        # A stage left out of the comparison entirely is not a stage with less
        # of the behavior, and the recommendation would read as if it were.
        if crossed:
            print(
                "Not compared at all: "
                + ", ".join(r["stage"] for r in crossed)
                + " was republished mid-search, so it could hold more of this"
                " than anything above. Re-run those stages before concluding."
            )


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
    if not registry.post_training_stages(target):
        # Pythia is the case: a base model with no post-training at all. Both
        # prompt sections below would print an empty heading each, which says
        # "we have not run this yet" — a very different claim from "this model
        # has no such stage to run it on". Say the latter once and skip them.
        print(
            f"\n{target['hf_model'] or target['name']} has no post-training"
            " stages — it was released as a base model, so there are no prompts"
            " to classify, no responses to classify them against, and no"
            " language to detect. `trainspotting pretrain` and `ask --pretrain`"
            " are what apply here."
        )
        # Not a stop, though: the corpus layers are what apply, and they are all
        # downstream of here. `ask --pretrain` on this target produces a rate per
        # corpus stage and a training-budget rollup over them, so returning at
        # this point would drop the one audit layer a base model *can* support
        # from the very report that just recommended running it.
        _report_influence(args.target)
        _report_questions(args.target, target)
        return
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
        # Every share below is over the labeled prompts. Refusals fall on
        # jailbreak-style content, so an unreported gap reads as a smaller
        # harmlessness share rather than as missing data.
        unlabeled = data.get("unlabeled", len(data["records"]) - n)
        if unlabeled:
            reasons = data.get("unlabeled_reasons") or {}
            detail = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
            print(
                f"- {unlabeled} of {n + unlabeled} sampled prompts went unlabeled"
                + (f" ({detail})" if detail else "")
                + " — excluded from every share below\n"
            )
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

    # String traces before the budget: main puts the budget last on purpose,
    # because the rates above it are per stage and not comparable to each other.
    if target["is_model"]:
        traces = _grep_traces(args.target, target)
        print("\n## String traces\n")
        if not traces:
            print(f"- no `grep` run yet (`trainspotting grep {args.target} \"some string\"`)")
        else:
            # Only true of stages the server converted in full, so it is said of
            # those rather than of the section: a partial conversion's
            # `total_rows` is the converted subset, and claiming otherwise here
            # would contradict the lower-bound warning printed under the stage.
            partial = sorted({f"{r['stage']} ({t['pattern']!r})"
                              for _, _, t in traces for r in t["stages"] if r["partial"]})
            print("Every count below is over all rows of the stage named, not a sample, so a "
                  "zero is the string being absent rather than merely unlikely — and a stage "
                  "listed as unsearched or inconclusive is neither.\n")
            if partial:
                print("Except where noted per stage: the datasets-server converted only part "
                      "of " + ", ".join(partial) + ", so those counts and denominators cover "
                      "the converted subset alone.\n")
            print(influence.BASIS_NOTE + "\n")
            if any(split for _, split, _ in traces):
                print("One slug below names more than one search — a pattern or a matching "
                      "flag was changed without changing the slug. Each is rendered on its "
                      "own; the stages under one heading are the stages that ran that exact "
                      "search.\n")
            for _, _, trace in traces:
                for line in influence.render(trace, args.target):
                    print(line)

    _report_influence(args.target)
    _report_questions(args.target, target)


def _report_influence(target_name: str) -> None:
    """The committed Bayesian-influence runs, if any.

    Placed after the string traces and before the questions because it answers
    the caveat the traces end on: a rate says where a phrase is, and this says
    which of the sampled examples the model's loss on it actually moves with.
    Nothing prints when nothing has run — the layer needs weights and a GPU, so
    its absence is the normal state of a checkout rather than an omission.
    """
    runs = bif.committed(target_name)
    if not runs:
        return
    print("\n## Bayesian influence\n")
    print("Posterior covariance between the model's loss on a query and its loss on each "
          "committed sampled example, sampled by SGLD around the released weights. "
          "Positive means training harder on the example would lower the query's loss. "
          "See `trainspotting bif --help`.\n")
    for res in runs:
        for line in bif.render(res):
            print(line)
        print()


def _report_questions(target_name: str, target: dict) -> None:
    """The free-form layers: what was asked, which way it pushes, what it costs
    as a share of training.

    Ordered so the last thing a reader sees is the budget. The rates above it
    are per stage and not comparable to each other — that is the whole reason
    the budget table exists — so leading with them and stopping would leave the
    report saying "6% of DPO prompts" as if it answered how much training the
    model got.
    """
    asks = paths.runs(target_name, "ask")
    stances = paths.runs(target_name, "stance")
    corpus_names = {x["stage"] for x in registry.pretrain_stages(target)}
    if not asks and not stances:
        return
    order = [s["stage"] for s in target["stages"]]

    if asks:
        print("\n## Custom questions (sampled)\n")
        for slug, stages in asks.items():
            data = {st: budget.load(f"{target_name}.{st}.ask-{slug}.json") for st in stages}
            print(f"### {slug}\n")
            # Grouped by the wording each run actually stored, not by slug. A
            # slug is not a question — see `_warn_mixed_questions` — and
            # printing one question over every stage's rate attributes the
            # others' measurements to words they were never scored against.
            # Keyed on the classifier as well as the wording, as the site's ask
            # cards are: the same words put to two judges are two measurements,
            # and a block that lists both rates under one heading names neither.
            groups: dict[tuple, list[str]] = {}
            for st in sorted(stages, key=lambda x: order.index(x) if x in order else 99):
                if data[st]:
                    groups.setdefault((data[st]["question"], data[st].get("classifier")), []).append(st)
            for (question, classifier), group in groups.items():
                print(f"> {question}\n")
                if classifier:
                    print(f"judged by {classifier}\n")
                for st in group:
                    d = data[st]
                    records = d["records"]
                    k, n = sum(bool(r["match"]) for r in records), len(records)
                    # A corpus run stores its own cluster-corrected interval; the
                    # binomial one would be too narrow for documents drawn by shard.
                    lo, hi = d["ci"] if d.get("ci") else _wilson(k, n)
                    print(
                        f"- {st:14s} {k / n * 100 if n else 0:5.1f}%  ({k}/{n},"
                        f" 95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"
                    )
                print()
            if len(groups) > 1:
                differ = "wording" if len({q for q, _ in groups}) > 1 else "classifier"
                if len({q for q, _ in groups}) > 1 and len({c for _, c in groups}) > 1:
                    differ = "wording and classifier"
                print(
                    f"({len(groups)} instruments share the slug {slug!r}, differing by"
                    f" {differ}; the rates above are grouped under the one each stage was"
                    " actually scored by.)\n"
                )
            # The rubric is named rather than used as a grouping key, because a
            # corpus stage and a post-training stage are scored under different
            # rubrics on every `--pretrain` run by design — keying on it would
            # split every such block in two. What is worth saying is a rubric
            # that moved between stages judged the same way, which is the same
            # rule `budget.mixing` applies.
            for question, group in groups.items():
                by_family: dict[str, dict[str, list[str]]] = {}
                for st in group:
                    sha = data[st].get("system_sha")
                    if sha:
                        fam = "pretrain" if st in corpus_names else "post-training"
                        by_family.setdefault(fam, {}).setdefault(sha, []).append(st)
                for fam, shas in by_family.items():
                    if len(shas) > 1:
                        detail = "; ".join(
                            f"{', '.join(sts)} under {sha[:12]}" for sha, sts in shas.items()
                        )
                        print(
                            f"(the {fam} stages above were scored under"
                            f" {len(shas)} different rubrics — {detail} — so their rates"
                            " are not directly comparable.)\n"
                        )

    if stances:
        print("\n## Which way each example pushes (whole examples, sampled)\n")
        for slug, stages in stances.items():
            print(f"### {slug}\n")
            # Grouped by the instrument each run recorded, the same way the ask
            # section and the site's stance cards are. Printing every stage
            # under the slug alone lets two nets scored against different words,
            # or by different judges, read as one answer — and a net is signed,
            # so averaging incompatible ones by eye is worse than a rate.
            # Keyed on the rubric as well. `stance.SYSTEM` is the instrument
            # here in the way the question is — it is what tells the judge that
            # a DISPREFERRED completion means the model is trained *out* of that
            # text — so rewording it moves toward/away labels while the question
            # and the classifier stay identical. Unlike the budget's
            # family-scoped check, every stance run is one family, so the hash
            # can go straight into the key.
            groups: dict[tuple, list[tuple[str, dict]]] = {}
            for st in sorted(stages, key=lambda x: order.index(x) if x in order else 99):
                d = budget.load(f"{target_name}.{st}.stance-{slug}.json")
                if d:
                    key = (d["question"], d.get("classifier"), d.get("system_sha"))
                    groups.setdefault(key, []).append((st, d))
            shas = {k[2] for k in groups}
            for (question, classifier, sha), group in groups.items():
                print(f"> {question}\n")
                by = f"judged by {classifier}" if classifier else ""
                # Only worth printing when it is what separates two groups —
                # otherwise it is a hash on every report for no reason.
                if sha and len(shas) > 1:
                    by = (by + " " if by else "") + f"under rubric {sha[:12]}"
                if by:
                    print(f"{by}\n")
                for st, d in group:
                    c = d["counts"]
                    n = len(d["records"])
                    if not n:
                        print(f"- {st:14s} every sampled example went unjudged — no direction to show")
                        continue
                    lo, hi = _wilson(c["toward"], n)
                    print(
                        f"- {st:14s} toward {c['toward']}/{n} = {c['toward'] / n * 100:.1f}%"
                        f" (95% CI {lo * 100:.1f}–{hi * 100:.1f}%), away {c['away']},"
                        f" net {d['net']:+d}"
                    )
                print()
            if len(groups) > 1:
                why = "wording, classifier or rubric" if len(shas) > 1 else "wording or classifier"
                print(
                    f"({len(groups)} instruments share the slug {slug!r}, differing by"
                    f" {why}; the nets above are grouped under the one that produced them"
                    " and do not combine.)\n"
                )

    for slug in asks:
        est = budget.estimate(target_name, slug)
        if not any(s.get("measured") for s in est["stages"]):
            continue
        print(f"\n## Training budget — {slug}\n")
        mixed = _warn_mixed_questions(est)
        print(BUDGET_COLS)
        print("-" * len(BUDGET_COLS))
        for st in est["stages"]:
            print(_budget_row(st))
        t = est["totals"]["all"]
        # A measured stage that could not be sized is dropped from both the
        # denominator and the matching sum, so the share is as partial as an
        # unasked one — `unsized` has to count here too.
        print(
            f"\nwhole pipeline: {_fmt_est(t['size_tokens'])} fit tokens — no single"
            " total, see the warning above"
            if mixed
            else f"\nwhole pipeline: {_fmt_est(t['matching_tokens'])} of"
            f" {_fmt_est(t['size_tokens'])} fit tokens ({_share_phrase(t)})"
        )
        # "Never asked" and "asked, and nothing usable came back" need different
        # advice: `--pretrain-only` does not re-run a failed post-training stage,
        # and telling someone to ask a question that already ran and failed sends
        # them in a circle. The rows above already print each stage's reason.
        unasked = [x for x in est["stages"] if not x.get("measured") and not x.get("unusable")]
        unusable = [x for x in est["stages"] if x.get("unusable")]
        if unasked:
            corpora = [x["stage"] for x in unasked if x["family"] == "pretrain"]
            how = (
                " --pretrain-only` to close the gap" if corpora and len(corpora) == len(unasked)
                else "` for the stages below"
            )
            print(
                f"  {len(unasked)} of {t['stages']} stages were never asked this question"
                f" ({', '.join(x['stage'] for x in unasked)}) — run"
                f" `trainspotting ask {target_name} \"...\" --slug {slug}{how}"
            )
        if unusable:
            print(
                f"  {len(unusable)} stage(s) were asked and produced nothing usable"
                f" ({', '.join(x['stage'] for x in unusable)}) — see the reason on each"
                " row above; re-asking without fixing that will fail the same way"
            )
        if t["unsized"]:
            print(
                f"  {', '.join(t['unsized'])} could not be sized, so"
                " the share above is over the stages that could be"
            )
        print()


def cmd_bif(args):
    """Weigh the committed sampled examples against a query by Bayesian influence.

    Everything before the sampler is the same plumbing the other layers use: the
    candidates are the context records and corpus documents already committed
    for the target, filtered by `--match` if a phrase is being chased. The
    sampler needs the checkpoint's weights, so this is the one command that
    imports torch, and it does so after the candidate set is known — a missing
    context file is a cheaper thing to find out than a missing GPU.
    """
    target = registry.resolve(args.target)
    model_id = args.model or target["hf_model"]
    if not model_id:
        sys.exit(
            f"{args.target} is a dataset, not a model: there are no weights to sample around."
            " Pass --model <hf id> to weigh its examples against some checkpoint anyway."
        )
    query = sys.stdin.read() if args.text == "-" else args.text
    if not query.strip():
        sys.exit("the query is empty")
    if args.match:
        try:
            re.compile(args.match)
        except re.error as e:
            sys.exit(f"--match {args.match!r} is not a valid regex: {e}")
    stages = [args.stage] if args.stage else None
    incomplete: dict[str, int] = {}
    cands, skipped = bif.candidates(
        args.target, target, stages=stages, match=args.match, limit=args.limit, seed=args.seed,
        incomplete=incomplete,
    )
    for stage, why in skipped.items():
        print(f"{stage}: not weighed — {why}", file=sys.stderr)
    for stage, n in incomplete.items():
        print(f"{stage}: {n} records skipped — the stored sample does not hold them as the model "
              f"was trained on them (tool use, a turn cut at 4,000 characters, or a document "
              f"stored as an excerpt)", file=sys.stderr)
    if not cands:
        sys.exit("no candidate examples: nothing committed for this target that this layer can weigh")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("this command needs torch and transformers: pip install -e '.[bif]'")
    tokenizer = bif.load_tokenizer(model_id)
    if getattr(tokenizer, "chat_template", None) and not args.prompt:
        sys.exit(
            f"{model_id} is a chat model: its template renders a reply behind a header and the "
            "turn it answers, so the query has to be given as a reply — pass --prompt with what "
            "the model was replying to"
        )
    device = bif.pick_device(args.device)
    print(f"loading {model_id} on {device} ({args.dtype})", file=sys.stderr)
    model, tokenizer, revision = bif.load(model_id, device, args.dtype, tokenizer=tokenizer)
    if model_id != target["hf_model"]:
        print(
            f"note: weighing against {model_id}, not {args.target}'s own checkpoint"
            f" {target['hf_model']}; the result file records which",
            file=sys.stderr,
        )
    encoded, kept, dropped = [], [], 0
    for c in cands:
        e = bif.encode(tokenizer, c["turns"], args.max_tokens)
        if e["fit_tokens"] == 0:
            dropped += 1
            continue
        kept.append(c)
        encoded.append(e)
    if dropped:
        print(f"{dropped} candidates have no fit tokens after encoding and were dropped", file=sys.stderr)
    if not encoded:
        sys.exit("no candidate has any text the model was fit to")
    q = bif.encode(tokenizer, bif.query_candidate(query, args.prompt)["turns"], args.max_tokens)
    if q["fit_tokens"] == 0:
        sys.exit("the query has no tokens to score")
    # The posterior is localized on the text the model was fit *toward*. A DPO
    # rejected completion is what the objective pushed the model off, so it is
    # scored at every draw but never drawn into a minibatch: fitting the chain
    # to it would localize the posterior on the opposite of what training did.
    localize = [i for i, c in enumerate(kept) if c["side"] != "rejected"]
    if not localize:
        sys.exit(
            "every candidate is a rejected completion, so there is no text the model was fit toward "
            "to localize the posterior on; widen --match or add a stage"
        )
    nbeta = args.nbeta if args.nbeta is not None else bif.default_nbeta(args.batch)
    settings = {
        "chains": args.chains,
        "draws": args.draws,
        "burn_in": args.burn_in,
        "every": args.every,
        "lr": args.lr,
        "nbeta": nbeta,
        "gamma": args.gamma,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": device,
        "seed": args.seed,
        "match": args.match,
        "limit": args.limit,
        "stage": args.stage,
    }
    print(
        f"{len(encoded)} candidates ({len(localize)} localized on), query {q['fit_tokens']} fit tokens; "
        f"{args.chains} chains × ({args.burn_in} burn-in + {args.draws} × {args.every} steps)",
        file=sys.stderr,
    )
    run = bif.sample(
        model, encoded, q, device=device, chains=args.chains, draws=args.draws,
        burn_in=args.burn_in, every=args.every, lr=args.lr, nbeta=nbeta, gamma=args.gamma,
        batch=args.batch, eval_batch=args.eval_batch, seed=args.seed, localize=localize,
        log=lambda m: print(m, file=sys.stderr),
    )
    # Through `_filename_part` like every other explicit slug: a slash in it
    # would write a file `bif.committed` never globs, and the run would vanish
    # from the report.
    slug = _filename_part(args.slug) if args.slug else _slug(query)
    res = bif.result(args.target, model_id, revision, query, args.prompt, kept, encoded, run, skipped, settings)
    res = {"slug": slug, **_stamp(), "dropped": dropped, "incomplete": incomplete, **res}
    path = _write_json(RESULTS / f"{args.target}.bif-{slug}.json", res)
    for line in bif.render(res):
        print(line)
    print(f"-> {path}", file=sys.stderr)


def cmd_lookup(args):
    """Count an exact string across the public corpora that have an index.

    The complement to every sampling command here: those ask what a corpus
    contains, this asks whether it contains one specific thing. No model is
    called and nothing is downloaded.
    """
    ids = args.index or [i["id"] for i in lookup.INDEXES]
    unknown = [i for i in ids if i not in lookup.INDEX_BY_ID]
    if unknown:
        sys.exit(
            f"unknown index {', '.join(unknown)}; known: "
            + ", ".join(i["id"] for i in lookup.INDEXES)
        )
    print(f"\n{args.query!r}\n")
    width = max(len(lookup.INDEX_BY_ID[i]["label"]) for i in ids)
    # The index counts token-boundary matches, not character substrings, so a
    # surprising count is sometimes a surprising tokenisation — `find` prints
    # the sequence for exactly this reason. Collected per index because each
    # index tokenises for itself, and printed once when they all agree.
    tokenised: dict[str, list[str]] = {}
    for idx in ids:
        info = lookup.INDEX_BY_ID[idx]
        try:
            r = lookup.probe(idx, args.query, args.docs)
        except lookup.LookupError_ as e:
            print(f"  {info['label']:<{width}}  — {e}")
            continue
        n = r["occurrences"]
        tokenised.setdefault(" | ".join(r.get("tokens") or []), []).append(info["label"])
        # Occurrences and documents are different numbers and the gap is the
        # whole point, so print both whenever documents were pulled rather than
        # letting one stand in for the other.
        detail = ""
        if args.docs and n:
            docs = r["documents"]
            # Three different claims, and the parenthetical has to say which:
            # every occurrence was seen; a random sample was drawn; or the
            # census asked for every rank and the index answered fewer.
            if r["exhaustive"]:
                how = f" (all {r['drawn']:,} occurrences)" if args.docs == "all" else ""
            elif args.docs == "all":
                how = f" ({r['drawn']:,} of {n:,} occurrences returned a document)"
            else:
                how = f" (sampled from {r['drawn']} draws)"
            detail = f"  in {len(docs)} document{'s' if len(docs) != 1 else ''}{how}"
        print(f"  {info['label']:<{width}}  {n:>9,} occurrence{'s' if n != 1 else ' '}{'~' if r['approx'] else ''}{detail}")
        for d in r.get("documents", []):
            bits = [b for b in [d["subset"], d["snapshot"], f"{d['tokens']:,} tok" if d["tokens"] else None] if b]
            if d.get("occurrences_drawn", 1) > 1:
                bits.append(f"×{d['occurrences_drawn']}" if args.docs == "all" else f"{d['occurrences_drawn']} draws")
            print(f"      {d['url'] or d['shard'] or '?'}")
            print(f"        {' · '.join(bits)}")
    for seq, labels in tokenised.items():
        if not seq:
            continue
        where = "" if len(tokenised) == 1 else f"  ({', '.join(labels)})"
        print(f"\n  matched as tokens: {seq}{where}")
    print()


def cmd_case_study(args):
    """Run a committed lookup study and write its result file for the site."""
    if args.slug not in casestudy.CASE_STUDIES:
        sys.exit(f"unknown case study {args.slug!r}; known: {', '.join(casestudy.CASE_STUDIES)}")
    RESULTS.mkdir(exist_ok=True)

    def progress(query, index):
        line = f"  {lookup.INDEX_BY_ID[index]['label']} · {query}"
        print(f"\r{line[:78]:<78}", end="", file=sys.stderr, flush=True)

    out = casestudy.run(args.slug, progress=progress)
    print(file=sys.stderr)
    path = RESULTS / f"case-study.{args.slug}.json"
    path.write_text(json.dumps(out, indent=2))

    probe, spread = out["probe"], out["spread"]
    print(
        f"{probe['query']!r}: {probe['occurrences']} occurrence(s) in "
        f"{len(probe['documents'])} document(s)"
        + ("" if probe["exhaustive"] else " (sampled)"),
        file=sys.stderr,
    )
    top = spread["domains"][:3]
    print(
        f"{spread['query']!r}: {spread['occurrences']:,} occurrences; of "
        f"{spread['drawn']} drawn, "
        + ", ".join(f"{d['domain']} {d['share'] * 100:.0f}%" for d in top),
        file=sys.stderr,
    )
    print(f"-> {path}", file=sys.stderr)


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
    p.add_argument(
        "--stage",
        help="only this stage — a post-training one (sft/dpo/rlvr), or with --pretrain "
        "a corpus one (pretrain/midtrain/long-context)",
    )
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.add_argument(
        "--pretrain",
        action="store_true",
        help="also score pretraining documents sampled by `trainspotting pretrain`",
    )
    p.add_argument(
        "--pretrain-only",
        action="store_true",
        help="score only the pretraining documents — for extending a question already "
        "answered over post-training without paying for those stages again",
    )
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser(
        "stance",
        help="judge which way each stored training example pushes on a question "
        "(toward / away / neither) — reads whole examples, not prompts",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("question")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr)")
    p.add_argument("--classifier", default="claude-opus-5")
    p.add_argument("--slug", help="short name for the result files (default: derived from the question)")
    p.set_defaults(fn=cmd_stance)

    p = sub.add_parser(
        "budget",
        help="roll an ask question up into a share of the training budget, in tokens",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("slug", help="the --slug of the ask runs to roll up")
    p.add_argument("--json", action="store_true", help="also write results/<target>.budget-<slug>.json")
    p.set_defaults(fn=cmd_budget)

    p = sub.add_parser(
        "find",
        help="exact occurrence count + example documents for a phrase, via infini-gram",
    )
    p.add_argument("phrase", help="the exact string to look up (matched on token boundaries)")
    p.add_argument(
        "--index",
        default=infinigram.DEFAULT_INDEX,
        help="infini-gram index to search; known: "
        + ", ".join(infinigram.INDEXES)
        + " (no Dolma 3 / OLMo 3 index exists publicly yet; other names are passed through)",
    )
    p.add_argument("--docs", type=_positive_int, default=5, help="example documents to retrieve")
    p.add_argument(
        "--maxlen",
        type=_positive_int,
        default=200,
        help="tokens of each document to display around the match",
    )
    p.add_argument("--json", action="store_true", help="also write results/find.<index>.<slug>.json")
    p.add_argument("--slug", help="short name for the result file (default: derived from the phrase)")
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser(
        "trace",
        help="find which stages a pasted behavior's distinctive phrases occur in",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("text", help="transcript or description to trace ('-' reads stdin)")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument(
        "--max-queries",
        type=_positive_int,
        default=6,
        help="most distinctive phrases to extract and search for",
    )
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser(
        "bif",
        help="weigh the committed sampled examples against a query by Bayesian influence "
        "(needs torch, transformers and the model's weights)",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("text", help="what the model said — the text whose loss is being explained ('-' reads stdin)")
    p.add_argument("--prompt", help="what it was replying to, scored as context rather than as target")
    p.add_argument("--stage", help="only this stage's committed sample")
    p.add_argument("--match", help="keep only candidates whose text holds this regex (case-insensitive)")
    p.add_argument("--limit", type=_positive_int, help="at most this many candidates per stage, drawn at random")
    p.add_argument("--model", help="HuggingFace id of the checkpoint to sample around (default: the target's own)")
    p.add_argument("--chains", type=_positive_int, default=4)
    p.add_argument("--draws", type=_draws, default=100, help="retained draws per chain (at least 2)")
    p.add_argument("--burn-in", type=_nonnegative_int, default=50, help="SGLD steps discarded per chain")
    p.add_argument("--every", type=_positive_int, default=1, help="SGLD steps between retained draws")
    p.add_argument("--lr", type=_positive_float, default=5e-8,
                   help="SGLD step size ε; the report says if the chains climbed away from w*, in which case lower it")
    p.add_argument("--nbeta", type=_positive_float, help="inverse temperature nβ (default: batch / ln batch)")
    p.add_argument("--gamma", type=_positive_float, default=100.0, help="localization strength γ")
    p.add_argument("--batch", type=_positive_int, default=8, help="candidates per SGLD minibatch")
    p.add_argument("--eval-batch", type=_positive_int, default=16, help="examples per forward pass when recording losses")
    p.add_argument("--max-tokens", type=_positive_int, default=512, help="tokens kept per example (the front is dropped)")
    # No float16: its exponent range is narrow enough that small likelihood
    # gradients underflow in `backward()` before the float32 master ever sees
    # them, and a chain fed those gradients samples the prior and the noise
    # rather than the posterior. Curing that needs a loss scaler; bfloat16 has
    # float32's exponent range and needs none.
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="weights precision; bfloat16 halves memory and keeps float32's exponent range")
    p.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slug", help="short name for the result file (default: derived from the text)")
    p.set_defaults(fn=cmd_bif)

    p = sub.add_parser("pretrain", help="sample documents from a model's pretraining corpora")
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

    p = sub.add_parser(
        "search",
        help="find a regex anywhere in the sampled examples — prompt, response, "
        "and for DPO which side of the pair",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="Python regex, case-insensitive unless --case-sensitive")
    p.add_argument("--stage", help="only this stage (sft/dpo/rlvr for a model; a dataset has one)")
    p.add_argument("--sample", type=_positive_int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--slug", help="short name for the result files (default: derived from the pattern)")
    p.add_argument("--show", type=int, default=3, help="matching rows to print (0 for none)")
    p.set_defaults(fn=cmd_search)

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

    p = sub.add_parser("lookup", help="count an exact string in the public corpora that have an index")
    p.add_argument("query", help="the exact string to look for")
    p.add_argument(
        "--index",
        action="append",
        help=f"restrict to one index (repeatable); default all of: {', '.join(i['id'] for i in lookup.INDEXES)}",
    )
    p.add_argument(
        "--docs",
        type=_docs_arg,
        default=0,
        metavar="N|all",
        help="also pull up to N documents behind each count (the index caps a single call at 10), "
        "or `all` to fetch every occurrence one request at a time",
    )
    p.set_defaults(fn=cmd_lookup)

    p = sub.add_parser("case-study", help="run a committed lookup study and write its result file")
    p.add_argument("slug", nargs="?", default="marginal-revolution",
                   help=f"one of: {', '.join(casestudy.CASE_STUDIES)}")
    p.set_defaults(fn=cmd_case_study)

    p = sub.add_parser("grep", help="exact string search over every row of each post-training mix")
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="literal substring, or a regex with --regex")
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument(
        "--field",
        action="append",
        choices=list(grep.GROUPS),
        help="which part of the example to search; repeatable, default all three",
    )
    p.add_argument(
        "--by",
        help="column to break the counts down by; default is the registry's "
        "source_columns, so the breakdown matches `sources`",
    )
    p.add_argument("--regex", action="store_true", help="treat the pattern as an RE2 regex")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep in the result file; 0 counts without keeping any")
    p.add_argument(
        "--max-gb",
        type=float,
        default=5.0,
        help="refuse to read more than this over the network; the plan is printed either way",
    )
    p.add_argument("--yes", action="store_true", help="scan whatever it costs")
    p.add_argument("--slug", help="short name for the result files (default: derived from the pattern)")
    p.set_defaults(fn=cmd_grep)

    p = sub.add_parser(
        "contaminate",
        help="is a benchmark's test set in the training data — which stage, which side",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("benchmark", help="one of: " + ", ".join(sorted(benchmarks.BENCHMARKS)))
    p.add_argument("--stage", help="only this post-training stage (sft/dpo/rlvr)")
    p.add_argument("--items", type=_positive_int, default=CONTAM_DEFAULTS["items"],
                   help="test items to probe; a seeded draw when the set is larger")
    p.add_argument("--seed", type=int, default=CONTAM_DEFAULTS["seed"])
    p.add_argument("--words", type=_probe_words, default=CONTAM_DEFAULTS["words"],
                   help="consecutive words per probe, cut from the middle of the item; "
                        f"at least {benchmarks.MIN_WORDS}")
    p.add_argument(
        "--field",
        action="append",
        choices=list(grep.GROUPS),
        help="which part of the example to search; repeatable, default every side",
    )
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep per stage; 0 counts without keeping any")
    p.add_argument("--max-gb", type=float, default=5.0,
                   help="refuse to read more than this over the network; the plan is printed either way")
    p.add_argument("--yes", action="store_true", help="scan whatever it costs")
    p.add_argument("--index", help="infini-gram index for the corpus side (default: the "
                   "registry's closest index for the model; a dataset has no corpus side "
                   "and takes none)")
    # Together these would fetch the benchmark, read nothing, and exit 0 with a
    # summary of stages not scanned — a run that asked no question.
    side = p.add_mutually_exclusive_group()
    side.add_argument("--no-corpus", action="store_true", help="skip the corpus side")
    side.add_argument("--corpus-only", action="store_true", help="skip the post-training scans")
    p.add_argument("--slug", help="short name for the result files (default: the benchmark, "
                   "with a hash of --items/--seed/--words/--field/--case-sensitive when any "
                   "is not at its default)")
    p.set_defaults(fn=cmd_contaminate)

    p = sub.add_parser(
        "steps",
        help="count a string in sampled training batches, in the order the model saw them (Pythia)",
    )
    p.add_argument("target", help=TARGET_HELP)
    p.add_argument("pattern", help="literal substring, or a regex with --regex")
    p.add_argument(
        "--sample",
        type=_positive_int,
        default=64,
        help="training steps to read, one from each equal slice of the run; 4.2 MB each",
    )
    p.add_argument("--at", type=_nonnegative_int, action="append", help="also read this exact step; repeatable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slices", type=_positive_int, default=8, help="stretches of the run to report the rate over")
    p.add_argument("--regex", action="store_true", help="treat the pattern as a Python regex")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--examples", type=_nonnegative_int, default=20,
                   help="matching snippets to keep in the result file; 0 counts without keeping any")
    p.add_argument("--slug", help="short name for the result file (default: derived from the pattern)")
    p.set_defaults(fn=cmd_steps)

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
    # Only the commands that take one: `find`, `lookup` and `case-study` are
    # about corpora rather than a registered target, and canonicalizing an
    # argument they never parsed raised AttributeError before their handler ever
    # ran. `find` has been unusable on main since this canonicalization landed.
    if getattr(args, "target", None) is not None:
        try:
            args.target = registry.resolve(args.target)["target"]
        except KeyError as e:
            sys.exit(e.args[0])
    args.fn(args)


if __name__ == "__main__":
    main()
