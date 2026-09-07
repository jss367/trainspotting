"""`classify` and `ask`: sample → extract → judge → write, in both label modes."""

import json
import sys
from pathlib import Path

from .. import classify, extract, hf, registry, paths
from ..paths import RESULTS
from ..stats import cluster_wilson as _cluster_wilson, wilson as _wilson
from .common import _counts, _print_match_rate, _sample_rows, _select_stages, _slug, _stamp, _unlabeled_note, _write_json


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


def _pretrain_docs_source(target_name: str, stage: str) -> Path | None:
    """Where to read one back, or None if this checkout has neither copy.

    `results/*.docs.json` is gitignored — it is a regenerable cache — so on a
    fresh clone the only copy of a committed sample is the one under docs/data/
    that the site serves. Reading only from results/ would tell someone who just
    cloned the repo that the sample shipped with it does not exist.
    """
    return paths.find(f"{target_name}.{stage}.docs.json")
