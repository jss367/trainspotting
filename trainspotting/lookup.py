"""Count an exact string in a public pretraining corpus and read the documents it lands in.

Every other layer here answers the unconditional question — what is in this
corpus, at what rate, with an interval. This answers the conditional one: is
*this specific text* in there, and how many times. That is the question someone
asks about their own writing, and sampling cannot answer it: a blog with a
thousand posts is a rounding error in a 6T-token mix, so a 300-document sample
will never contain it.

The index is infini-gram (Liu et al., https://infini-gram.io), a suffix array
over several public corpora. It answers exact-substring counts in milliseconds
and returns the documents behind them with their original filtering metadata
still attached, which is what makes a count interpretable — see `normalize`.

Two properties of the API shape everything downstream, and both are easy to
misread:

  * `count` counts **occurrences, not documents**. A page that repeats a phrase
    twice contributes 2. Reporting a count as "copies in the training data" is
    wrong, and wrong in the direction that inflates it.
  * `search_docs` returns a **uniform random sample of occurrences**, capped at
    ten per call and different on every call. So the documents are exhaustive
    only when the count is at most ten; above that they are a sample, and a
    committed result file is a snapshot rather than something a re-run
    reproduces.

Neither corpus here is Dolma 3, because no public infini-gram index covers it.
Everything this module returns is about a different corpus than the rest of the
site, which is why it is a separate view that names its corpus in every row.
"""

import collections
import itertools
import json
import time
from urllib.parse import urlsplit

import requests

API = "https://api.infini-gram.io/"

# The server rejects anything larger. Ten occurrences per call is also the unit
# the sampler below counts in, so raising this if the API ever allows it is the
# only change needed to draw a bigger sample per round trip.
MAX_DOCS_PER_CALL = 10

# Corpora with a public index, in the order the site lists them. `label` is what
# a reader sees; `note` says whose corpus it is and roughly when, because the
# whole point of showing five of them is that the answer depends on which one a
# model was trained on.
INDEXES = [
    {
        "id": "v4_dolma-v1_7_llama",
        "label": "Dolma 1.7",
        "note": "Ai2's open pretraining corpus, the OLMo 2 generation.",
    },
    {
        "id": "v4_dclm-baseline_llama",
        "label": "DCLM-baseline",
        "note": "A heavily model-filtered Common Crawl derivative.",
    },
    {
        "id": "v4_rpj_llama_s4",
        "label": "RedPajama v1",
        "note": "Together's open reproduction of the LLaMA 1 mix.",
    },
    {
        "id": "v4_c4train_llama",
        "label": "C4",
        "note": "Google's cleaned Common Crawl, one 2019 snapshot.",
    },
    {
        "id": "v4_piletrain_llama",
        "label": "The Pile",
        "note": "EleutherAI's mix, mostly curated sources rather than open web.",
    },
]

INDEX_BY_ID = {i["id"]: i for i in INDEXES}


class LookupError_(RuntimeError):
    """The index rejected the query — unknown index, or a query it cannot serve."""


def _post(payload: dict) -> dict:
    """POST with backoff.

    The API reports its own errors inside a 200 body rather than as a status, so
    a bare `raise_for_status` would hand back `{"error": ...}` as if it were a
    result and the caller would read a missing count as zero occurrences. Check
    the body.
    """
    last: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(API, json=payload, timeout=120)
            # 403 is this API's rate limit, not an auth failure — it is open and
            # takes no key. A study runs dozens of queries back to back and will
            # hit it, so back off rather than reporting "forbidden" and losing
            # the whole run at the last query.
            if r.status_code in (403, 429) or r.status_code >= 500:
                time.sleep(5 * 2**attempt)
                continue
            r.raise_for_status()
            j = r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(2 * 2**attempt)
            continue
        if j.get("error"):
            raise LookupError_(j["error"])
        return j
    raise LookupError_(f"infini-gram unreachable: {last}")


def count(index: str, query: str) -> dict:
    """Occurrences of `query` in `index`, as an exact substring.

    `approx` is the server's own flag: it estimates rather than counts for
    queries whose result set is enormous. Passed through rather than hidden, so
    a number that is not exact never reads as one.
    """
    j = _post({"index": index, "query_type": "count", "query": query})
    return {
        "occurrences": j["count"],
        "approx": bool(j.get("approx")),
        "tokens": j.get("tokens", []),
    }


def _snapshot(deep: dict) -> str | None:
    """The Common Crawl snapshot a document came from, e.g. "CC-MAIN-2021-10".

    Subsets disagree about where they put it: the CCNet-derived buckets bury it
    in a segment path, Falcon keeps a `dump` field, and the curated subsets have
    no snapshot at all because they are not crawls.
    """
    if isinstance(deep.get("dump"), str):
        return deep["dump"]
    for key in ("cc_segment", "segment", "provenance"):
        v = deep.get(key)
        if isinstance(v, str):
            for part in v.replace("/", " ").split():
                if part.startswith("CC-MAIN-"):
                    return part
    return None


def _hostname(url) -> str | None:
    """The host a document's recorded URL points at, or None if it will not parse.

    `hostname` rather than `netloc`: the latter keeps a port and any userinfo,
    so https://marginalrevolution.com:443/x would compare unequal to the
    subject's own site and be tallied as somebody else's — which is the study's
    headline number.

    Anything that is not a parseable string URL is no host. `urlsplit` raises on
    a non-string and on a malformed bracketed IPv6 host, and a crawl's metadata
    is exactly where such a URL turns up; letting it raise here would throw away
    the whole retrieved sample over one bad record.
    """
    if not isinstance(url, str) or not url:
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _quality(attrs: dict) -> float | None:
    """Dolma's quality-classifier score for the document, where it recorded one.

    Stored as a list of [start, end, score] spans over the text. One span covers
    the whole document in every case seen here, and averaging would be a fiction
    anyway, so take the first.

    A score that will not convert is treated as unavailable, the same as one
    that is absent. Everything else this module reads out of a document's
    metadata already works that way, and it has to: the sample is drawn from a
    live index across five corpora whose subsets disagree about every field, so
    one heterogeneous record raising here would discard every document
    retrieved with it rather than losing one number off one of them.
    """
    for key, spans in (attrs or {}).items():
        if "hq" not in key or not isinstance(spans, list) or not spans:
            continue
        first = spans[0]
        if isinstance(first, list) and len(first) == 3:
            try:
                score = float(first[2])
            except (TypeError, ValueError):
                return None
            # NaN and the infinities are floats that will not serialize: `json`
            # writes them as bare `NaN` / `Infinity` tokens, which no browser's
            # parser accepts, so one of them in a study would stop the page
            # loading the file at all rather than showing a missing score.
            return round(score, 5) if -1e308 < score < 1e308 else None
    return None


# Counter for hits with no file and no URL, so each stays its own document.
_UNIDENTIFIED = itertools.count()


def _identity(rec: dict) -> tuple:
    """What makes two hits the same document.

    `doc_ix` numbers a document within one suffix-array shard, so it is not an
    identity on its own — `cli.cmd_find` keys on `(shard, doc_ix)` for exactly
    this reason and says so. The retrieval path there knows its shard because it
    picked it; `search_docs` does not return one (checked against the live API:
    a hit carries `doc_ix`, `doc_len`, `disp_len`, `needle_offset`, `spans`,
    `token_ids`, `metadata`, `blocked`, and nothing else), so the identity here
    is the corpus file the document came from instead. Two documents sharing a
    `doc_ix` across shards are in different files; two hits sharing both are the
    same document seen twice, which is the case this deduplicates.
    """
    if rec.get("shard"):
        return ("shard", rec["shard"], rec.get("doc_ix"))
    # No file to name it by — which is exactly what the malformed-metadata
    # fallback produces, so this is not a rare branch. A URL still identifies
    # the page. With neither, two hits are only known to share a shard-local
    # index, which is not evidence they are the same document, so nothing is
    # merged: over-counting distinct documents costs a slightly higher document
    # count, while merging costs both draws being attributed to one of them's
    # domain, and `domain_shares` is what the study's headline is computed over.
    if rec.get("url"):
        return ("url", rec["url"], rec.get("doc_ix"))
    return ("unidentified", next(_UNIDENTIFIED))


def normalize(doc: dict) -> dict:
    """One search_docs hit, flattened to the fields that make a count readable.

    A raw hit nests three levels of metadata whose shape differs per subset
    (`cc_en_middle` carries CCNet fields, `falcon-refinedweb-filtered` and
    `c4-filtered` carry their own), so every field here is optional and read
    defensively. What matters is that the provenance survives: which corpus
    subset, whose site, which crawl, what the filters thought of it.
    """
    # Decoded the way `doc_provenance` decodes it, and for the same reason: this
    # metadata comes from five heterogeneous corpora, and one document with a
    # malformed string, a JSON null, or an already-decoded object would
    # otherwise raise and take the whole sample with it. A document with
    # unreadable provenance is a document with no provenance, not a failed run.
    meta = doc.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    inner = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    deep = inner.get("metadata") if isinstance(inner.get("metadata"), dict) else {}
    attrs = inner.get("attributes") if isinstance(inner.get("attributes"), dict) else {}

    # Every nesting a subset is known to use, in the order `doc_provenance`
    # reads them — that parser exists because these shapes disagree, and a URL
    # this one fails to find is a document that falls into "(no url recorded)"
    # and drops out of `domain_shares`, which is what the study's headline
    # "0 of 60 are on the blog" is computed over.
    url = (
        inner.get("url")
        or inner.get("WARC-Target-URI")
        or deep.get("url")
        or deep.get("WARC-Target-URI")
        or (inner.get("id") if str(inner.get("id", "")).startswith("http") else None)
    )
    # Only a string is a URL. A subset that recorded an integer id or a nested
    # object under one of these keys would otherwise put that object in the
    # record — where the site renders it as a link and `_identity` puts it in a
    # dict key, which a dict is not allowed to be. Unreadable is unavailable
    # here as everywhere else in this function.
    if not isinstance(url, str) or not url:
        url = None
    # Fold `www.` here rather than at each call site. The subsets disagree —
    # CCNet's recorded `source_domain` keeps the prefix, a domain parsed out of
    # a URL may not — and left alone that splits one site across two rows of
    # every tally, which reads as two smaller sources instead of one large one.
    # `hostname`, not `netloc`: the latter keeps a port and any userinfo, so a
    # URL like https://marginalrevolution.com:443/x would compare unequal to the
    # subject's site and be tallied as somebody else's — which is the study's
    # headline number. `hostname` also lowercases, which the fold below then
    # only has to strip `www.` from.
    # The recorded domain first, the URL's host as the fallback — but only if the
    # recorded one is usable. A plain `or` picked a truthy non-string
    # `source_domain` (an id, a nested object) and the type check below then
    # turned it into None, throwing away a hostname the URL could still have
    # supplied. That is a document dropped from `domain_shares`, which is what
    # the study's headline share is computed over, for a field that was never
    # the better answer anyway.
    recorded = deep.get("source_domain")
    domain = recorded if isinstance(recorded, str) and recorded else _hostname(url)
    domain = domain.lower().removeprefix("www.") if isinstance(domain, str) else None

    # How much of the page survived line-level filtering. This is the number
    # that explains why a long blog post is a short training document, so it is
    # worth carrying even though only the CCNet-derived subsets record it.
    kept, orig = deep.get("nlines"), deep.get("original_nlines")

    # The window around the match comes back as [text, label] spans — the
    # matched needle carries a label, the surrounding context does not — so the
    # readable document is their concatenation.
    #
    # The spans come first, and `text` is only the fallback. They are the window
    # the index centred on the match; `text` is the document from its beginning,
    # and it is then cut to fit, so for anything longer than the cap the excerpt
    # was the opening paragraphs and did not contain the query at all. The
    # committed probe has one: a document retrieved for "There is no great
    # stagnation, cereal edition" whose stored excerpt never says it, which
    # leaves a reader unable to check the one thing the study asks them to.
    spans = "".join(
        str(span[0]) for span in doc.get("spans", []) if isinstance(span, list) and span and span[0]
    )
    text = spans or doc.get("text") or ""
    return {
        "doc_ix": doc.get("doc_ix"),
        # Tokens, not characters: infini-gram measures documents in the Llama
        # tokens its index is built over.
        "tokens": doc.get("doc_len"),
        "subset": (meta.get("path") or "").split("/")[0] or None,
        "shard": meta.get("path"),
        "url": url,
        "domain": domain,
        "title": deep.get("title"),
        "snapshot": _snapshot(deep),
        "crawled": deep.get("date_download") or inner.get("created"),
        "quality": _quality(attrs),
        "lines_kept": kept,
        "lines_original": orig,
        # A window the index centres on the match, not the head of the
        # document — so it routinely opens mid-sentence, and the page has to
        # say so rather than letting it read as the page's opening lines.
        "excerpt": text[:1200],
    }


def sample_documents(index: str, query: str, occurrences: int, want: int = MAX_DOCS_PER_CALL) -> dict:
    """Documents behind `query`, and whether they are all of them.

    Above the ten-per-call cap this makes repeated calls and merges them, the
    last asking for the remainder so a `want` of 11 costs eleven documents
    rather than twenty. The calls are independent uniform draws over
    occurrences, so `drawn` (with repeats) is the denominator for any share
    computed over the sample, while `documents` is deduplicated for display.
    Conflating the two would report "half these copies are on scraper sites"
    off a denominator that had already collapsed a site's five copies into one.

    `exhaustive` is the only claim worth making carefully: it means the sample
    holds every occurrence, which takes both a count at or under the cap (one
    call can see them all) and a `want` that reached for all of them. Anything
    else is a snapshot, including three documents of a phrase that occurs eight
    times.
    """
    # Asking for more documents than there are occurrences does not return
    # fewer — the server samples occurrences with replacement and pads to
    # `maxnum`, so a two-occurrence query answered ten of the same two. Ask for
    # exactly the occurrences that exist and the result is each of them once.
    # Asking for more than the caller wanted is the same error in the other
    # direction: `--docs 11` used to spend two full ten-document calls and hand
    # back twenty, which breaks the "up to N" the CLI promises and, worse, puts
    # twenty in the `drawn` denominator every share is computed over.
    budget = min(want, occurrences)
    # The claim is that the sample holds every occurrence, which needs both: one
    # call can see them all, and this call asked for them all. A caller that
    # asked for three of eight has a sample, not a census, however small the
    # count is.
    exhaustive = occurrences <= MAX_DOCS_PER_CALL and budget == occurrences
    seen: dict[int, dict] = {}
    drawn = 0
    remaining = budget
    while remaining > 0:
        maxnum = min(MAX_DOCS_PER_CALL, remaining)
        j = _post(
            {
                "index": index,
                "query_type": "search_docs",
                "query": query,
                "maxnum": maxnum,
            }
        )
        documents = j.get("documents", [])
        for raw in documents:
            rec = normalize(raw)
            drawn += 1
            key = _identity(rec)
            if key in seen:
                seen[key]["occurrences_drawn"] += 1
            else:
                rec["occurrences_drawn"] = 1
                seen[key] = rec
        # Count what was asked for, not what came back: a short reply means the
        # index has no more to give, and looping on the shortfall would spin.
        remaining -= maxnum
        if not documents:
            break
    # Every call redraws from the whole occurrence list, so the same document
    # recurs across calls. Keep one copy and count the hits: the copy is what a
    # reader opens, the count is what any share over the sample is weighted by.
    return {
        "drawn": drawn,
        # What was asked for is not what arrived. A count of eight can come back
        # with three documents, or none, and the claim "this is every copy" has
        # to answer to the reply rather than to the request — otherwise a short
        # answer from the index reads on the page as an exhaustive list.
        "exhaustive": exhaustive and drawn >= budget,
        "documents": sorted(seen.values(), key=lambda d: -d["occurrences_drawn"]),
    }


def domain_shares(documents: list[dict]) -> list[dict]:
    """Which sites host the sampled occurrences, most first.

    Weighted by occurrences drawn rather than by distinct document, because the
    question is where the copies of a text live: a scraper that reposts it on
    forty pages genuinely is forty copies, and collapsing those to one would
    make the original site look proportionally far more present than it is.
    """
    tally: collections.Counter = collections.Counter()
    for d in documents:
        tally[d["domain"] or "(no url recorded)"] += d.get("occurrences_drawn", 1)
    total = sum(tally.values()) or 1
    return [
        {"domain": dom, "occurrences": n, "share": n / total}
        for dom, n in tally.most_common()
    ]


def probe(index: str, query: str, docs: int = 0) -> dict:
    """A count plus, optionally, the documents behind it."""
    out = count(index, query)
    if docs and out["occurrences"]:
        out.update(sample_documents(index, query, out["occurrences"], docs))
    return out
