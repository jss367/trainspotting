"""Thin client for the HuggingFace datasets-server API.

Everything here works without downloading the datasets: /info for schema and
row counts, /statistics for exact column value frequencies, /rows for sampling.
"""

import os
import random
import re
import time
from pathlib import Path

import requests

BASE = "https://datasets-server.huggingface.co"
ROWS_PER_PAGE = 100  # server maximum for /rows length


def _token() -> str | None:
    """The user's Hub token, if they have one lying around. Optional — every
    dataset here is public — but anonymous rate limits are low enough that a
    search across nine mixes can exhaust them, and authenticated ones are not."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        return (Path.home() / ".cache" / "huggingface" / "token").read_text().strip() or None
    except OSError:
        return None


HEADERS = {"Authorization": f"Bearer {tok}"} if (tok := _token()) else {}


def _get(path: str, **params) -> dict:
    """GET with backoff, patient about the two errors that mean "later", not "no".

    A 429 means the rate limit needs to drain, so honor retry-after and keep
    trying — six exponential attempts was enough until /search, whose index
    polling (below) is itself request traffic, so a long index load used to eat
    the whole allowance and then die on the first 429. A /search against a
    dataset nobody has searched lately answers 500 "the dataset index is
    loading" while the server pulls a multi-GB index; that too is a flat wait,
    not a fault. Both are bounded by the loop cap (~45 min worst case). Other
    500s keep the short exponential clock: they clear on a retry or not at all.
    """
    attempt = 0
    r = exc = None
    for _ in range(90):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=120, headers=HEADERS)
        except (requests.Timeout, requests.ConnectionError) as e:
            # A cold /search chews on a multi-GB index server-side and can blow
            # straight past any client timeout. Same story as the 429: it means
            # later, not no — and it never carries a response to raise from.
            exc = e
            time.sleep(30)
            continue
        if r.status_code >= 500 and "index is loading" in r.text:
            time.sleep(60)
            continue
        if r.status_code == 429:
            # Polling every few seconds while waiting is what keeps the limit
            # pinned — the wait traffic *is* the traffic — so wait a real minute.
            time.sleep(int(r.headers.get("retry-after", 0)) or 60)
            continue
        if r.status_code >= 500:
            time.sleep(2 * 2**attempt)
            attempt += 1
            if attempt >= 6:
                break
            continue
        r.raise_for_status()
        return r.json()
    if r is None:
        raise exc
    r.raise_for_status()


def search_rows(
    dataset: str,
    query: str,
    offset: int = 0,
    length: int = ROWS_PER_PAGE,
    config: str = "default",
    split: str = "train",
) -> dict:
    """One page of /search full-text matches: rows with indices, plus the exact
    total match count in num_rows_total. The index is word-based — every row
    containing all the query's words, in any string column, any order — so the
    caller must check the rows for the exact phrase itself."""
    return _get(
        "search",
        dataset=dataset, config=config, split=split,
        query=query, offset=offset, length=length,
    )


def dataset_info(dataset: str, config: str = "default") -> dict:
    j = _get("info", dataset=dataset)
    return j["dataset_info"][config]


def num_rows(dataset: str, config: str = "default", split: str = "train") -> int:
    return dataset_info(dataset, config)["splits"][split]["num_examples"]


def column_frequencies(
    dataset: str, columns: list[str], config: str = "default", split: str = "train"
) -> dict[str, dict[str, int]]:
    """Exact value counts for string-label columns, precomputed by HF."""
    j = _get("statistics", dataset=dataset, config=config, split=split)
    out = {}
    for col in j.get("statistics", []):
        if col["column_name"] in columns:
            freq = col["column_statistics"].get("frequencies")
            if freq:
                out[col["column_name"]] = dict(
                    sorted(freq.items(), key=lambda kv: -kv[1])
                )
    return out


def sample_rows_with_index(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[tuple[int, dict]]:
    """Sample ~n rows via random pages of the /rows endpoint, keeping row indices.

    Rows within a page are correlated (adjacent on disk), so we draw many small
    chunks from uniformly random offsets rather than a few full pages. The index
    is the row's absolute position in the split, which addresses it in the HF
    dataset viewer.
    """
    total = num_rows(dataset, config, split)
    rng = random.Random(seed)
    chunk = 10
    n_pages = (n + chunk - 1) // chunk
    offsets = sorted(rng.randrange(max(1, total - chunk)) for _ in range(n_pages))
    rows: list[tuple[int, dict]] = []
    for off in offsets:
        j = _get(
            "rows", dataset=dataset, config=config, split=split, offset=off, length=chunk
        )
        rows.extend((off + i, r["row"]) for i, r in enumerate(j["rows"]))
    rng.shuffle(rows)
    return rows[:n]


def sample_rows(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[dict]:
    """The same sample as sample_rows_with_index, without the indices."""
    return [row for _, row in sample_rows_with_index(dataset, n, seed, config, split)]


HUB = "https://huggingface.co"
_REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")


def dataset_url(value: str) -> str | None:
    """The public hub page for a source label, or None.

    Source-mixture labels are a mix of real dataset repo ids ("hamishivi/
    math_rlvr_mixture_dpo") and bare internal names ("flan_v2_converted") that
    address nothing. Even among the repo-shaped ones some are private, and the
    hub answers 401 for private and missing alike — so only labels that resolve
    anonymously get a link.
    """
    if not _REPO_ID.fullmatch(value):
        return None
    url = f"{HUB}/datasets/{value}"
    try:
        if requests.head(url, timeout=30, allow_redirects=True).status_code == 200:
            return url
    except requests.RequestException:
        return None
    return None
