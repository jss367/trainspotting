"""Exact string search over open training corpora, via Ai2's infini-gram API.

Sampling (`pretrain` + `ask`) answers the unconditional question — "what is in
here" — with a rate and an interval. This module answers the pointed one: "is
this specific string in here, how many times, and in what documents". The two
are complements; neither substitutes for the other.

infini-gram (https://infini-gram.io) serves exact n-gram counts and document
retrieval over suffix-array indexes of open corpora. The count doubles as a
duplication count, which matters on its own: memorization scales with how many
times a string appears in training.

One honest limitation, stated wherever results are shown: as of 2026-08 the
public API has no index over Dolma 3 / the OLMo 3 corpora this tool samples.
The closest indexes cover OLMo 2's full training data (olmo-mix pretraining +
Dolmino midtraining + Tulu 3 post-training) and Dolma 1.7. Dolma 3 is largely
a re-filtering of the same upstream sources (Common Crawl, arXiv, code), so an
OLMo 2 hit is real evidence the string is in the ecosystem's training text —
but it is a different corpus, and a count here is not a count over what OLMo 3
saw. When Ai2 publishes a Dolma 3 index, adding it to INDEXES is the only
change needed.

Two API quirks the caller should know:
- Queries are tokenized, and matches align to token boundaries: querying "a"
  counts the token " a", not the letter.
- Retrieval is a two-step: `find` returns rank ranges per suffix-array shard,
  `get_doc_by_rank` fetches one document per (shard, rank). Ranks are picked
  evenly across the global range so the examples spread over the corpus
  instead of clustering in one shard.
"""

import json
import time

import requests
import requests.exceptions

API_URL = "https://api.infini-gram.io/"

# The public indexes worth pointing at from here, with what each covers.
# Any other index name is passed through untouched — the API rejects unknown
# ones with a clear error, and this list going stale should not gate access
# to an index Ai2 adds later (a Dolma 3 one, with luck).
INDEXES = {
    "v4_olmo-2-0325-32b-instruct_llama": (
        "OLMo 2 32B full training data: olmo-mix-1124 pretraining + Dolmino "
        "midtraining + Tulu 3 post-training (~4.6T tokens)"
    ),
    "v4_olmo-2-1124-13b-instruct_llama": (
        "OLMo 2 13B full training data (~4.6T tokens)"
    ),
    "v4_olmoe-0125-1b-7b-instruct_llama": (
        "OLMoE full training data (~4.6T tokens)"
    ),
    "v4_olmo-mix-1124_llama": "olmo-mix-1124, OLMo 2 pretraining only",
    "v4_dolma-v1_7_llama": "Dolma 1.7, OLMo 1.7 pretraining (~2.6T tokens)",
}

DEFAULT_INDEX = "v4_olmo-2-0325-32b-instruct_llama"

NO_OLMO3_CAVEAT = (
    "The public infini-gram API has no Dolma 3 / OLMo 3 index, so this count "
    "is over a different corpus than the one this tool samples — the closest "
    "available, not the one the registered models were trained on."
)


def caveat_for(index: str) -> str | None:
    """The corpus caveat that is true of `index`, or None.

    Every index in INDEXES predates Dolma 3, so the no-OLMo-3 caveat holds for
    all of them. An index this module does not know — including the Dolma 3
    one this file hopes Ai2 publishes — must not be characterized: asserting
    "closest available, not what the model saw" about an arbitrary index would
    be exactly wrong the day someone passes the real thing.
    """
    return NO_OLMO3_CAVEAT if index in INDEXES else None


def _post(payload: dict, timeout: int = 90) -> dict:
    """POST with backoff. The API asks callers to expect transient failures,
    and reports its own errors as 200s with an `error` body — surface those as
    exceptions rather than letting a caller index into an error dict."""
    last: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(API_URL, json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * 2**attempt)
                continue
            r.raise_for_status()
            body = r.json()
            if "error" in body:
                raise RuntimeError(f"infini-gram: {body['error']}")
            return body
        except (
            requests.ConnectionError,
            requests.Timeout,
            # A hangup mid-body raises this rather than ConnectionError, and
            # not at the top level of `requests` — same failure mode
            # pretrain._get retries for the Hub.
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last = e
            time.sleep(2 * 2**attempt)
    if last:
        raise last
    raise RuntimeError("infini-gram: retries exhausted")


def find(index: str, phrase: str) -> dict:
    """Locate every occurrence of `phrase` in `index`.

    Returns the API's response: `cnt` (exact occurrence count), `tokens` (how
    the phrase was tokenized, which is what was actually matched), and
    `segment_by_shard` (rank ranges that `get_doc` can resolve to documents).
    """
    return _post({"index": index, "query_type": "find", "query": phrase})


def get_doc(index: str, phrase: str, shard: int, rank: int, max_disp_len: int = 200) -> dict:
    """One matching document by (shard, rank), truncated to `max_disp_len` tokens."""
    return _post(
        {
            "index": index,
            "query_type": "get_doc_by_rank",
            "query": phrase,
            "s": shard,
            "rank": rank,
            "max_disp_len": max_disp_len,
        }
    )


def spread_picks(segment_by_shard: list, k: int) -> list[tuple[int, int]]:
    """Up to `k` (shard, rank) picks spread evenly across every match.

    `segment_by_shard` concatenates into one global range of matches; taking
    evenly spaced positions across it — rather than the first k of shard 0 —
    returns examples from across the corpus, and stays deterministic so a rerun
    retrieves the same documents. Fewer than k matches returns them all.

    Doubling k refines the spread without moving it: every pick at k is also a
    pick at 2k (position 2j·total//2k equals j·total//k). Callers lean on that
    to retry duplicates — re-asking at higher resolution and skipping ranks
    already tried visits new, still-evenly-spread positions.
    """
    sizes = [(s, hi - lo) for s, (lo, hi) in enumerate(segment_by_shard)]
    total = sum(n for _, n in sizes)
    if total == 0:
        return []
    k = min(k, total)
    picks = []
    for i in range(k):
        g = i * total // k  # global match index, evenly spaced from 0
        for s, n in sizes:
            if g < n:
                picks.append((s, segment_by_shard[s][0] + g))
                break
            g -= n
    return picks


def snippet(doc: dict) -> str:
    """The displayed slice of a retrieved document, matches marked with «».

    `spans` pairs each run of text with the clause it matched or None, so the
    marking shows exactly what the engine matched — including a tokenization
    that grabbed more or less than the reader assumed.
    """
    return "".join(
        text if clause is None else f"«{text}»" for text, clause in doc.get("spans", [])
    )


def doc_provenance(doc: dict) -> dict:
    """What the corpus recorded about where a retrieved document came from.

    The API returns each index's raw per-document metadata as a JSON string
    whose shape varies by source (WARC headers for web crawl, plain ids for
    curated sets). Keep the stable, useful part: the shard file, the source
    tag, and a URL when one exists.
    """
    raw = doc.get("metadata")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    inner = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    out = {}
    if raw.get("path"):
        out["path"] = raw["path"]
    source = inner.get("source") or raw.get("source")
    if source:
        out["source"] = source
    url = inner.get("url") or inner.get("WARC-Target-URI")
    if not url and isinstance(inner.get("metadata"), dict):
        url = inner["metadata"].get("url") or inner["metadata"].get("WARC-Target-URI")
    if not url and isinstance(inner.get("id"), str) and inner["id"].startswith("http"):
        url = inner["id"]
    if url:
        out["url"] = url
    return out
