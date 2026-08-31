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


# Top-up rounds when the draw comes back short of n distinct rows. Bounded so a
# split smaller than the request returns a short sample instead of looping.
MAX_SAMPLE_ROUNDS = 6


def sample_rows_with_index(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[tuple[int, dict]]:
    """Sample ~n *distinct* rows via random pages of the /rows endpoint, keeping
    row indices.

    Rows within a page are correlated (adjacent on disk), so we draw many small
    chunks from uniformly random offsets rather than a few full pages. The index
    is the row's absolute position in the split, which addresses it in the HF
    dataset viewer.

    Offsets are drawn independently, so two of them landing within `chunk` of
    each other return the same rows twice. Keying on the absolute index drops
    the repeats: a duplicated row is a duplicated vote in every rate computed
    over the sample, and taking the first n after a shuffle hid that as a
    slightly small sample rather than a slightly wrong one. Collisions are rare
    on a large split and common on a small one, which is exactly where each row
    carries the most weight. Fresh offsets top the sample back up to n.

    Deterministic in (n, seed). Two changes have moved which rows a given seed
    draws — deduplication, and widening the page-start bound below — so runs are
    only comparable to each other when they were drawn by the same version. Over
    the nine Dolci splits at n=300, 8,997 of 9,000 seeds draw exactly what they
    drew before the bound widened; the exceptions are seed 998 on Dolci-Think-RL-7B
    and seeds 71 and 764 on Dolci-Instruct-RL. Everything committed under
    docs/data/ is seed 0, which is unchanged on all nine, so those files still
    describe the rows they were drawn from. A re-run at a different seed is a
    different sample and does not join against them.
    """
    total = num_rows(dataset, config, split)
    rng = random.Random(seed)
    chunk = 10

    def draw(pages: int) -> list[int]:
        # Inclusive upper bound. `randrange` stops one short, which left the
        # last page start unreachable and with it the final rows of the split:
        # an 11-row split could only ever draw offset 0 and came back
        # permanently short of its own 11 rows.
        return sorted(rng.randrange(max(1, total - chunk + 1)) for _ in range(pages))

    seen: dict[int, dict] = {}
    offsets = draw((n + chunk - 1) // chunk)
    for _ in range(MAX_SAMPLE_ROUNDS):
        for off in offsets:
            # Every row this page would return is already held, so the request
            # would spend a round trip to learn nothing. Skipping it is not the
            # same as stopping: a later offset in the same round can still be
            # fresh, and a round that happens to redraw covered ground says
            # nothing about whether the split has more rows to give.
            if all(i in seen for i in range(off, min(off + chunk, total))):
                continue
            j = _get(
                "rows", dataset=dataset, config=config, split=split, offset=off, length=chunk
            )
            for i, r in enumerate(j["rows"]):
                seen.setdefault(off + i, r["row"])
        if len(seen) >= min(n, total):
            break
        shortfall = n - len(seen)
        offsets = draw((shortfall + chunk - 1) // chunk)
    rows = sorted(seen.items())
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


def dataset_revision(dataset: str, ref: str = "main") -> str | None:
    """The commit SHA `ref` points at, or None if the hub will not say.

    Stamped into every result file so a number stays attributable: `main` moves,
    and a re-upload of a Dolci mix would otherwise turn an old count into a
    claim about a dataset that no longer exists in that form.

    Best-effort on purpose, unlike `pretrain.resolve_revision` which pins a
    cache key and must fail loudly. Provenance is worth one request; it is not
    worth failing a sampling run that has already been paid for.
    """
    try:
        r = requests.get(f"{HUB}/api/datasets/{dataset}/revision/{ref}", timeout=30)
        if r.status_code == 200:
            return r.json().get("sha")
    except (requests.RequestException, ValueError):
        return None
    return None


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
