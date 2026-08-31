"""A worked example of looking one writer's work up in a pretraining corpus.

`lookup` gives anyone a count. A count on its own is the part people misread —
both numbers in this study looked like the opposite of what they were on first
inspection — so the site ships one study that runs the queries, keeps the
documents behind them, and shows what the number turned out to mean.

The spec below is data rather than page copy so a second study is a new entry
here plus a re-run, not an edit to the HTML. What each entry has to carry, and
the reason each field exists:

  * `selection` on every query group. Some of these queries were chosen from
    knowing the blog, and some were found by reading documents already sampled
    out of the corpus. The second kind cannot measure how often a post is
    present — it was selected for being present — and a study that let those two
    kinds share a table would be quietly reporting a hit rate of 100%.
  * `note` on every query, saying what a hit or a miss would mean. Without it a
    reader supplies their own interpretation, which is the failure this study
    exists to prevent.
"""

import datetime

from . import lookup

# Read the module docstring before adding one. The `selection` field is the part
# that keeps this honest, and it is not a formality.
CASE_STUDIES = {
    "marginal-revolution": {
        "title": "Is one blogger's work in the pretraining data?",
        "subject": "Marginal Revolution",
        "byline": "Tyler Cowen and Alex Tabarrok, daily since 2003",
        "site": "marginalrevolution.com",
        "question": (
            "Take a blog that has published every day for two decades. Is all of "
            "it in there? Once each, or a hundred times?"
        ),
        "answer": (
            "Not all of it, about once each where it is there at all, and most of "
            "what looks like duplication belongs to somebody else's site."
        ),
        "groups": [
            {
                "name": "Phrases the blog repeats",
                "selection": "independent",
                "explain": (
                    "Chosen from how the blog writes, before looking at any "
                    "corpus. Each occurrence is one post's use of the phrase, so "
                    "the count is a ceiling on how many such posts are present."
                ),
                "queries": [
                    {
                        "q": "For the pointer I thank",
                        "note": "The standard credit line when a reader sends in a link.",
                    },
                    {
                        "q": "Solve for the equilibrium",
                        "note": "A recurring one-line post, and a phrase the wider web has since borrowed.",
                    },
                    {
                        "q": "Assorted links",
                        "note": "The near-daily link roundup, published under this heading for twenty years.",
                    },
                    {
                        "q": "That was then, this is now",
                        "note": "A recurring series, and also an ordinary English sentence — a reminder that a common phrase measures nothing.",
                    },
                ],
            },
            {
                "name": "Individual posts",
                "selection": "found-in-corpus",
                "explain": (
                    "Post titles taken from blog pages that were already sampled "
                    "out of the corpus, so every one of them is present by "
                    "construction. These measure what a present post looks like, "
                    "not how often a post is present."
                ),
                "queries": [
                    {
                        "q": "There is no great stagnation, cereal edition",
                        "note": "A November 2012 link post.",
                    },
                    {
                        "q": "The Inuit ear-pulling game",
                        "note": "A December 2012 link post.",
                    },
                    {
                        "q": "Garett Jones on the top economic stories of 2012",
                        "note": "A December 2012 link post.",
                    },
                    {
                        "q": "Pseudo-placebo effects in RCTs",
                        "note": "A December 2012 link post.",
                    },
                ],
            },
        ],
        # The one post opened all the way up. Chosen for having a small enough
        # count that the document list is exhaustive rather than sampled.
        "probe": {
            "query": "There is no great stagnation, cereal edition",
            "index": "v4_dolma-v1_7_llama",
            "explain": (
                "Few enough occurrences that the index returns every one of them, "
                "so this is the whole of what the corpus holds for this post."
            ),
        },
        # The name of the blog, which appears far more often than any of its
        # posts do — and mostly not on the blog.
        "spread": {
            "query": "Marginal REVOLUTION",
            "index": "v4_dolma-v1_7_llama",
            "draws": 60,
            "explain": (
                "The site's own title, as it renders in the page. Too common for "
                "an exhaustive list, so the index draws occurrences at random and "
                "this is where they turned out to live."
            ),
        },
    }
}

# What a re-run cannot reproduce, and why. Committed alongside the numbers
# because the file is a snapshot of a live index, not a derivation from
# something pinned.
CAVEAT = (
    "Counts are occurrences of the exact string, not documents: one page that "
    "repeats a phrase contributes more than one. Documents behind a count are "
    "complete only where the count is at or under ten, which is the index's "
    "per-call cap; above it the index draws occurrences at random and a re-run "
    "returns different ones. None of these corpora is Dolma 3, so none of this "
    "describes what OLMo 3 was trained on — no public index covers Dolma 3."
)


def run(slug: str, indexes: list[str] | None = None, progress=None) -> dict:
    """Run a study's queries and return the result file's contents.

    Counts run across every index; the document pulls run against the single
    index each probe names. A probe is a claim about one corpus, and running it
    across five would produce five sets of documents whose shard paths and
    filtering fields mean different things.
    """
    spec = CASE_STUDIES[slug]
    ids = indexes or [i["id"] for i in lookup.INDEXES]

    counts = []
    for group in spec["groups"]:
        rows = []
        for q in group["queries"]:
            by_index = {}
            for idx in ids:
                if progress:
                    progress(q["q"], idx)
                by_index[idx] = lookup.count(idx, q["q"])
            rows.append({**q, "by_index": by_index})
        counts.append({k: v for k, v in group.items() if k != "queries"} | {"rows": rows})

    p = spec["probe"]
    if progress:
        progress(p["query"], p["index"])
    probe_count = lookup.count(p["index"], p["query"])
    probe_docs = lookup.sample_documents(
        p["index"], p["query"], probe_count["occurrences"], lookup.MAX_DOCS_PER_CALL
    )

    s = spec["spread"]
    if progress:
        progress(s["query"], s["index"])
    spread_count = lookup.count(s["index"], s["query"])
    spread_docs = lookup.sample_documents(
        s["index"], s["query"], spread_count["occurrences"], s["draws"]
    )

    domains = lookup.domain_shares(spread_docs["documents"])
    site = spec["site"].removeprefix("www.")
    on_site = sum(d["occurrences"] for d in domains if d["domain"] == site)

    return {
        "slug": slug,
        **{k: v for k, v in spec.items() if k not in ("groups", "probe", "spread")},
        # A live index with no revision to pin, so the date is the whole of the
        # provenance. Anything read off this file is what the index said then.
        "run_on": datetime.date.today().isoformat(),
        "api": lookup.API,
        "caveat": CAVEAT,
        "indexes": [lookup.INDEX_BY_ID[i] for i in ids],
        "groups": counts,
        "probe": {**p, **probe_count, **probe_docs},
        "spread": {
            **s,
            **spread_count,
            **spread_docs,
            "domains": domains,
            # The headline of this half: how many of the drawn copies sit on
            # the site that wrote the thing. Derived here so the page states a
            # number the result file already contains rather than one it
            # recomputes from the domain table.
            "on_subject_site": on_site,
        },
    }
