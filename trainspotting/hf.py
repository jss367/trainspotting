"""Thin client for the HuggingFace datasets-server API.

Everything here works without downloading the datasets: /info for schema and
row counts, /statistics for exact column value frequencies, /rows for sampling.
"""

import random
import time

import requests

BASE = "https://datasets-server.huggingface.co"
ROWS_PER_PAGE = 100  # server maximum for /rows length


def _get(path: str, **params) -> dict:
    for attempt in range(6):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", 0)) or 5 * 2**attempt)
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


def sample_rows(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[dict]:
    """Sample ~n rows via random pages of the /rows endpoint.

    Rows within a page are correlated (adjacent on disk), so we draw many small
    chunks from uniformly random offsets rather than a few full pages.
    """
    total = num_rows(dataset, config, split)
    rng = random.Random(seed)
    chunk = 10
    n_pages = (n + chunk - 1) // chunk
    offsets = sorted(rng.randrange(max(1, total - chunk)) for _ in range(n_pages))
    rows: list[dict] = []
    for off in offsets:
        j = _get(
            "rows", dataset=dataset, config=config, split=split, offset=off, length=chunk
        )
        rows.extend(r["row"] for r in j["rows"])
    rng.shuffle(rows)
    return rows[:n]
