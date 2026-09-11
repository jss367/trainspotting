"""`find`: count an exact string in the closest public infini-gram index."""

import hashlib
import re
import sys

from .. import infinigram
from ..paths import RESULTS
from .common import _filename_part, _stamp, _write_json


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
