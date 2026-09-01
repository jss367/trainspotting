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
    dataset here is public — but anonymous rate limits are shared per IP and
    low enough that two sampling runs side by side can exhaust them, while
    authenticated ones are roomy.

    Resolved the way huggingface_hub resolves it, so a login done through any
    of its knobs is found: HF_TOKEN outranks the token file, whose path is
    HF_TOKEN_PATH if set, else <HF_HOME>/token, where HF_HOME itself defaults
    to <XDG_CACHE_HOME or ~/.cache>/huggingface."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    hf_home = os.environ.get("HF_HOME") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    )
    path = os.environ.get("HF_TOKEN_PATH") or str(Path(hf_home) / "token")
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


HEADERS = {"Authorization": f"Bearer {tok}"} if (tok := _token()) else {}


def _get(path: str, server_error_retries: int = 6, **params) -> dict:
    """GET with backoff, patient about the errors that mean "later", not "no".

    A 429 honors retry-after and waits the window out flat — the exponential
    clock capped at six attempts died mid-run whenever the shared per-IP quota
    was already drained by a run next door, which is exactly when the wait is
    worth it. A timeout retries for the same reason: a page the server is slow
    to cut is still coming. Other 500s keep the short exponential clock; they
    clear on a retry or not at all. Everything is bounded by the loop cap.

    `server_error_retries` raises that clock's cap for the one caller whose 500
    is a progress report rather than a fault: the first /search against a large
    split answers "the dataset index is loading" until the server has finished
    building a full-text index over it, which on a multi-million-row mix takes
    minutes and not the ninety seconds six attempts buy. The per-attempt sleep
    is capped so twenty attempts is a bounded wait rather than an exponential
    one.
    """
    attempt = transport = 0
    for _ in range(60):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=120, headers=HEADERS)
        except (requests.Timeout, requests.ConnectionError):
            # A slow page is worth waiting on; a dead network is not. Five
            # tries with growing pauses tells the two apart without letting an
            # offline host consume the whole 60-iteration budget half a minute
            # at a time.
            transport += 1
            if transport >= 5:
                raise
            time.sleep(15 * transport)
            continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", 0)) or 60)
            continue
        if r.status_code >= 500:
            time.sleep(min(30, 2 * 2**attempt))
            attempt += 1
            if attempt >= server_error_retries:
                break
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


# Top-up rounds when the draw comes back short of n distinct rows. Bounded so a
# split smaller than the request returns a short sample instead of looping.
MAX_SAMPLE_ROUNDS = 6

# Rows per /rows request. Small on purpose: it is the size of the correlated
# unit this sampler draws in, so it is also the cluster size any interval over
# the sample has to widen for. `sample_rows_with_pages` reports which page each
# row came from so that widening can happen.
ROW_PAGE = 10

# The name main's `derive.clusters_of` reads this constant by. Same value, one
# definition: a second literal here is a second thing to keep in step, and the
# two describe the same page size.
CHUNK = ROW_PAGE


def sample_rows_with_pages(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[tuple[int, dict, list[str], int]]:
    """Sample ~n *distinct* rows via random pages of the /rows endpoint, keeping
    row indices, the names of any cells the server shortened to fit its
    response limit, and the offset of the page each row arrived in.

    A truncated cell only matters to a caller reading the text itself — a search
    cannot find a string in the part the server cut — so it travels alongside the
    row rather than in it.

    The page offset travels the same way, and for the same reason it cannot be
    reconstructed later: pages start at arbitrary offsets, so the ten rows of one
    draw straddle any fixed grid over the index, and a caller that guessed at
    `index // ROW_PAGE` would split correlated rows across two clusters — the
    anti-conservative direction for an interval. The rows that shared a request
    are the ones that were adjacent on disk, and only this function knows which
    those were.

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

    Deterministic in (n, seed). Three changes have moved which rows a given seed
    draws — deduplication, widening the page-start bound, and filling the gaps
    the rounds leave on a nearly-exhausted split — so runs are only comparable to
    each other when they were drawn by the same version. The last two only reach
    a split the request nearly covers, which none of the Dolci mixes are. Over
    the nine Dolci splits at n=300, 8,997 of 9,000 seeds draw exactly what they
    drew before the bound widened; the exceptions are seed 998 on Dolci-Think-RL-7B
    and seeds 71 and 764 on Dolci-Instruct-RL. Everything committed under
    docs/data/ is seed 0, which is unchanged on all nine, so those files still
    describe the rows they were drawn from. A re-run at a different seed is a
    different sample and does not join against them.
    """
    total = num_rows(dataset, config, split)
    rng = random.Random(seed)
    chunk = ROW_PAGE

    def draw(pages: int) -> list[int]:
        # Inclusive upper bound. `randrange` stops one short, which left the
        # last page start unreachable and with it the final rows of the split:
        # an 11-row split could only ever draw offset 0 and came back
        # permanently short of its own 11 rows.
        return sorted(rng.randrange(max(1, total - chunk + 1)) for _ in range(pages))

    # index -> (row, truncated cells, offset of the page it arrived in). First
    # write wins, so a row covered by two overlapping pages is attributed to the
    # one that actually fetched it rather than to whichever came later.
    seen: dict[int, tuple[dict, list[str], int]] = {}
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
                seen.setdefault(off + i, (r["row"], r.get("truncated_cells") or [], off))
        if len(seen) >= min(n, total):
            break
        shortfall = n - len(seen)
        offsets = draw((shortfall + chunk - 1) // chunk)

    if len(seen) < min(n, total):
        # Random offsets do not guarantee coverage. When the request is a large
        # fraction of the split, the rounds can end with a row undrawn simply
        # because no page happened to start near it — an 11-row split asked for
        # 11 rows came back with 10 — and downstream that reads as a smaller
        # sample rather than an unlucky one. Walk the uncovered pages in order.
        #
        # Only the near-full regime can get here, and there uniformity is moot:
        # the draw already wants essentially every row.
        #
        # Every page listed here is uncovered, so every request adds at least one
        # row and the walk below stops at the shortfall — a handful of requests,
        # however many gaps there are. Listing them costs one pass over the page
        # starts and no network, which is why an earlier attempt to cap the list
        # was wrong: at 76 rows with n=30 and seed 11 the rounds came back one
        # row short, the seventh gap exceeded the cap, and the whole fallback was
        # abandoned with 47 rows still unseen.
        gaps = []
        for off in range(0, total, chunk):
            if not all(i in seen for i in range(off, min(off + chunk, total))):
                gaps.append(off)
        # Walk the gaps in random order. Taking them by offset and stopping as
        # soon as the sample is full hands every remaining slot to the front of
        # the split: at 50 rows with n=20 and seed 13 the rounds covered rows
        # 9-27 and the fill took offset 0, so rows 28-49 could not be selected
        # at all. Shuffling costs nothing and is still deterministic in `seed`.
        rng.shuffle(gaps)
        for off in gaps:
            if len(seen) >= min(n, total):
                break
            j = _get(
                "rows", dataset=dataset, config=config, split=split, offset=off, length=chunk
            )
            for i, r in enumerate(j["rows"]):
                seen.setdefault(off + i, (r["row"], r.get("truncated_cells") or [], off))

    rows = [
        (index, row, truncated, page)
        for index, (row, truncated, page) in sorted(seen.items())
    ]
    rng.shuffle(rows)
    return rows[:n]


def sample_rows_with_truncation(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[tuple[int, dict, list[str]]]:
    """The same sample, without the page each row was drawn in."""
    return [
        (index, row, truncated)
        for index, row, truncated, _ in sample_rows_with_pages(dataset, n, seed, config, split)
    ]


def sample_rows_with_index(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[tuple[int, dict]]:
    """The same sample, without the truncation report."""
    return [
        (index, row)
        for index, row, _ in sample_rows_with_truncation(dataset, n, seed, config, split)
    ]


def sample_rows(
    dataset: str,
    n: int,
    seed: int = 0,
    config: str = "default",
    split: str = "train",
) -> list[dict]:
    """The same sample as sample_rows_with_index, without the indices."""
    return [row for _, row in sample_rows_with_index(dataset, n, seed, config, split)]


def search_count(
    dataset: str, query: str, config: str = "default", split: str = "train"
) -> tuple[int, bool]:
    """How many rows of the split contain `query`, via full-text search.

    Returns (matches, partial), on the same terms as `column_frequencies`: the
    server's full-text index covers only the first 5 GB of a split, and on a
    bigger one it sets `partial` and the count is over that prefix. Dolci Think
    SFT is 36 GB, so the flag is not hypothetical — treating a partial count as
    a whole-split one would understate it about sevenfold, and understate it for
    exactly the largest stages, which is worse than a wrong number: it is a
    ranking that puts the biggest mixes last for being big.

    Nothing is downloaded and nothing is sampled. The index covers the string
    columns, including strings nested inside a struct or a list of structs, so
    the assistant turns of a `messages` column are searched and not just the
    scalar metadata beside them. The first search against a cold dataset warms
    the index and can take minutes; `_get` waits it out with extra retries.
    `num_rows_total` is the whole match count regardless of the page size, so
    this asks for the smallest legal page (one row) and reads the total off it.

    Matching is by token, not by substring: the server stems each token
    (Porter) and ANDs the query's tokens together, so a multi-word query finds
    rows holding all of its words rather than the literal phrase, and finds
    "develops" for "developed". A phrase pulled from a transcript is exactly
    that shape, which is why `behavior` emits word windows rather than raw
    substrings — but a count is an upper bound on verbatim occurrences, and a
    hit is worth clicking through to `search` before it is believed.
    """
    j = _get(
        "search",
        server_error_retries=20,
        dataset=dataset,
        config=config,
        split=split,
        query=query,
        offset=0,
        length=1,
    )
    return j["num_rows_total"], bool(j.get("partial"))


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
