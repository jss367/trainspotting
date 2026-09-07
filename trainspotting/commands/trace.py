"""`trace`: from an observed behaviour to the stages that most densely hold its phrases."""

import sys
import urllib.parse

from .. import behavior, hf, registry
from .common import _select_stages


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
