"""Thin client for the HuggingFace datasets-server API.

Everything here works without downloading the datasets: /info for schema and
row counts, /statistics for exact column value frequencies, /rows for sampling.
"""

import random
import re
import time

import requests

BASE = "https://datasets-server.huggingface.co"
ROWS_PER_PAGE = 100  # server maximum for /rows length


def _get(path: str, **params) -> dict:
    """GET with backoff. The datasets-server rate-limits (429) and occasionally
    500s on a large page; both clear on a retry."""
    for attempt in range(6):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", 0)) or 5 * 2**attempt)
            continue
        if r.status_code >= 500:
            time.sleep(2 * 2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def dataset_info(dataset: str, config: str = "default") -> dict:
    j = _get("info", dataset=dataset)
    return j["dataset_info"][config]


def num_rows(dataset: str, config: str = "default", split: str = "train") -> int:
    return dataset_info(dataset, config)["splits"][split]["num_examples"]


def column_frequencies(
    dataset: str, columns: list[str], config: str = "default", split: str = "train"
) -> tuple[dict[str, dict[str, int]], int, bool]:
    """Value counts for string-label columns, precomputed by HF.

    Returns (frequencies, counted, partial). On a big dataset the server stops
    after a first slice and sets `partial`; the counts are then over `counted`
    rows rather than the whole split, and `counted` is the only honest
    denominator for them. WildChat-1M is the first registry entry where this
    fires — 778,133 of its 837,989 rows — and dividing by the full row count
    would quietly report every share about 7% low, under a heading that says
    "exact".
    """
    j = _get("statistics", dataset=dataset, config=config, split=split)
    out = {}
    for col in j.get("statistics", []):
        if col["column_name"] in columns:
            freq = col["column_statistics"].get("frequencies")
            if freq:
                out[col["column_name"]] = dict(
                    sorted(freq.items(), key=lambda kv: -kv[1])
                )
    return out, j.get("num_examples", 0), bool(j.get("partial"))


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
