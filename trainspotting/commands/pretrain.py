"""`pretrain`: sample documents from a model's pretraining corpora."""

import sys
from pathlib import Path

from .. import extract, hf, pretrain, registry
from ..paths import RESULTS
from .common import _select_stages, _stamp, _write_json


def _pretrain_docs_path(target_name: str, stage: str) -> Path:
    """Where `pretrain` writes a document sample."""
    return RESULTS / f"{target_name}.{stage}.docs.json"


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
