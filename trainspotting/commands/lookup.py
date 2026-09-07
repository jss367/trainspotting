"""`lookup`: count an exact string in the public corpora that have an index."""

import sys

from .. import lookup


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
