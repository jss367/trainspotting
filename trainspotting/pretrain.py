"""Sample documents from Dolma 3 pretraining shards without downloading them.

The post-training layers read the HuggingFace datasets-server, but that route
does not work for pretraining. The server indexes only the first ~5 GB of these
repos (`"partial": true`), and the shards are ordered by topic cluster, so
`/rows` walks one cluster at a time: offsets 0-100k of the 150B mix are adult
content, 300k onward are art. A sample drawn that way is not slightly skewed,
it is a tour of whichever clusters sort first.

So we go to the repo files instead. Dolma 3 ships as `.jsonl.zst` shards under
paths that name their own provenance:

    data/common_crawl-crime_and_law-0007/shard_00000112.jsonl.zst
    data/stack_edu-Python-0001/shard_00000009.jsonl.zst

We list every shard once, draw shards with probability proportional to size,
and read each pick's head with an HTTP range request. A zstd stream decodes
from the front, so ~96 KB over the wire yields whole documents out of a shard
that may be 400 MB — with one larger retry where a single document runs longer
than that read, which the long-context mixes are full of.

The bias this leaves is stated plainly in `SAMPLING_CAVEAT` and travels with
every result file: shards are drawn properly, documents within a shard are not
— we always see a shard's opening documents. With thousands of shards across
dozens of source/topic groups, the shard draw carries most of the variance, but
this is not a uniform sample over documents and nothing here should claim it is.
"""

import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import requests.exceptions
import zstandard

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"

# One range request per sampled shard. 96 KB decompresses to roughly 300 KB of
# JSONL, which is dozens of web pages or a handful of olmOCR papers — and only
# one document is kept, so reading more is wasted on most shards.
READ_BYTES = 96_000

# Except where a single document is longer than the whole read. The long-context
# mixes hold documents past 200k characters, so their shards decode to nothing
# at 96 KB; those get one retry at this multiple.
LONG_DOC_FACTOR = 8

# Shards are drawn in batches until the sample is full. This bounds the retries
# when a corpus simply cannot supply the request — a mix of 55 shards cannot
# yield 300 documents at one per shard, and the caller should get a short sample
# and an honest count rather than an endless loop.
MAX_BATCHES = 6

SAMPLING_CAVEAT = (
    "Shards are drawn with probability proportional to size and one document is "
    "taken uniformly from each, so draws are independent. What is not corrected "
    "for is position: a range request only reaches the front of a shard, so each "
    "document comes from its shard's first few hundred, never the tail."
)


CACHE_DIR = Path(__file__).resolve().parent.parent / ".shard-cache"


def _cache_path(dataset: str, revision: str) -> Path:
    return CACHE_DIR / f"{dataset.replace('/', '__')}@{revision}.json"


def _get(url: str, **kwargs) -> requests.Response:
    """GET with backoff, returning a response whose body is already in memory.

    The Hub rate-limits (429), occasionally 500s, and occasionally hangs up
    mid-body on a range request — a truncated transfer raises only when the
    content is read, so `.content` is touched here, inside the retry loop,
    rather than leaving a landmine for the caller.
    """
    last: Exception | None = None
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=90, **kwargs)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", 0)) or 5 * 2**attempt)
                continue
            if r.status_code >= 500:
                time.sleep(2 * 2**attempt)
                continue
            r.raise_for_status()
            r.content
            return r
        except (
            requests.ConnectionError,
            requests.Timeout,
            # Not re-exported at the top level of `requests`, unlike the others.
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last = e
            time.sleep(2 * 2**attempt)
    if last:
        raise last
    r.raise_for_status()


# Shard directories name their own provenance, but the three mixes spell it
# differently — the pretrain mix uses `common_crawl-crime_and_law-0007`, Dolmino
# prefixes a recipe bucket and a quality vigintile
# (`ingredient1-common_crawl-high-quality_19_crime_and_law`), and Longmino adds
# decontamination rounds (`common_crawl-high-quality_vigintile0019_subset-decon-2-crime_and_law`).
# Parsing is vocabulary-driven rather than positional so the same corpus lands
# under the same label in all three, which is what lets the site compare stages.

# Longest first: `olmocr_science_pdfs` must win before any shorter prefix.
SOURCES = [
    ("olmocr_science_pdfs", "olmocr_science_pdfs"),
    ("common_crawl", "common_crawl"),
    ("stack_edu", "stack_edu"),
    ("finemath", "finemath"),
    ("rpj-proofpile-arxiv", "arxiv"),
    ("dolma1_7-wiki", "wikipedia"),
    ("wiki_to_rcqa", "wikipedia"),
]

# Filtering bookkeeping: quality tiers, vigintile indices, decontamination
# rounds, fill-in-the-middle markers, token-budget suffixes. None of it is
# provenance, all of it is glued into the directory name.
# The digit rule is separator-aware rather than \b-based, because "_" is a word
# character: `high-quality_19_crime_and_law` must lose the 19, while
# `finemath-3plus` must keep its quality tier.
BOOKKEEPING = re.compile(
    r"high[-_]quality|vigintile\d*|subset|decon\w*|fim|length|denyagain|part\d+"
    r"|(?<![0-9a-z])(?:\d+e\d+|\d+)(?![0-9a-z])",
    re.I,
)

# Dolmino abbreviates the topic clusters the pretrain mix spells out.
TOPIC_ALIASES = {
    "art_design": "art_and_design",
    "crime_law": "crime_and_law",
    "education_jobs": "education_and_jobs",
    "finance_business": "finance_and_business",
    "food_dining": "food_and_dining",
    "hardware": "electronics_and_hardware",
    "history": "history_and_geography",
    "home_hobbies": "home_and_hobbies",
    "science_tech": "science_math_and_technology",
    "software_dev": "software_development",
    "sports_fitness": "sports_and_fitness",
    "travel_tourism": "travel_and_tourism",
    "fashion_beauty": "fashion_and_beauty",
    "social": "social_life",
    "adult": "adult_content",
    "travel": "travel_and_tourism",
}


def split_group(directory: str) -> tuple[str, str]:
    """Directory name -> (source corpus, topic cluster).

    Mixes that are not split by topic (`cranemath`, `megamatt`, `tulu-3-sft`)
    are their own source and report no topic.
    """
    name = re.sub(r"^ingredient\d+-", "", directory)  # Dolmino recipe bucket
    name = re.sub(r"-\d+$", "", name)  # shard bucket index

    for prefix, source in SOURCES:
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            rest = BOOKKEEPING.sub(" ", rest)
            rest = re.sub(r"[-_\s]+", "_", rest).strip("_")
            return source, TOPIC_ALIASES.get(rest, rest)
    return name, ""


def list_shards(dataset: str, revision: str = "main", cache: bool = True) -> list[dict]:
    """Every data shard in the repo, with its size and provenance labels.

    A paginated walk of the Hub tree API — 6,081 shards over seven pages for the
    150B mix, far more for the full 6T one. The listing only changes when Ai2
    republishes the dataset, so it is cached on disk and every sample of the same
    mix reuses it.
    """
    cached = _cache_path(dataset, revision)
    if cache and cached.exists():
        raw = json.loads(cached.read_text())
    else:
        raw, url = [], f"{HF_API}/{dataset}/tree/{revision}?recursive=1&limit=1000"
        while url:
            r = _get(url)
            for entry in r.json():
                path = entry.get("path", "")
                if entry.get("type") != "file" or not path.endswith(".jsonl.zst"):
                    continue
                if len(path.split("/")) < 2:
                    continue
                raw.append([path, entry.get("size") or 0])
            match = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = match.group(1) if match else None
        if not raw:
            raise RuntimeError(f"no .jsonl.zst shards found in {dataset}")
        if cache:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(raw, separators=(",", ":")))

    # Labels are derived on load, not cached, so editing split_group re-labels
    # existing caches instead of silently serving stale provenance.
    shards = []
    for path, size in raw:
        source, topic = split_group(path.split("/")[-2])
        shards.append({"path": path, "size": size, "source": source, "topic": topic})
    return shards


def group_sizes(shards: list[dict]) -> dict[str, dict]:
    """Compressed bytes and shard count per source/topic group, for the facts layer."""
    out: dict[str, dict] = {}
    for s in shards:
        key = f"{s['source']}-{s['topic']}" if s["topic"] else s["source"]
        g = out.setdefault(
            key, {"source": s["source"], "topic": s["topic"], "bytes": 0, "shards": 0}
        )
        g["bytes"] += s["size"]
        g["shards"] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["bytes"]))


def read_shard_head(
    dataset: str, path: str, revision: str = "main", read_bytes: int = READ_BYTES
) -> list[dict]:
    """Documents from the front of one shard, via a single range request.

    The last line of a truncated zstd stream is almost always a partial JSON
    object, and the decompressor itself raises once it runs out of frame, so
    both are swallowed: we keep whatever parsed cleanly.
    """
    url = HF_RESOLVE.format(dataset=dataset, revision=revision, path=path)
    r = _get(url, headers={"Range": f"bytes=0-{read_bytes - 1}"})
    buf = io.BytesIO(r.content)
    docs = []
    try:
        with zstandard.ZstdDecompressor().stream_reader(buf) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # truncated tail record
    except (zstandard.ZstdError, EOFError):
        pass  # stream ends mid-frame; the docs already parsed are still good
    return docs


def sample_documents(
    dataset: str,
    n: int,
    seed: int = 0,
    revision: str = "main",
    shards: list[dict] | None = None,
    read_bytes: int = READ_BYTES,
    docs_per_shard: int = 1,
    workers: int = 8,
    progress=None,
) -> list[dict]:
    """`n` documents drawn across size-weighted shards.

    Weighting by compressed size approximates weighting by token count, which
    is what makes the source mix come out right — `common_crawl-health` is 27x
    the bytes of `common_crawl-fashion_and_beauty` and should appear 27x as
    often.

    One document per shard by default, which costs a round trip per document but
    keeps the draws independent. Raising `docs_per_shard` is faster and returns
    correlated documents: neighbours in a shard share a topic cluster, so a
    proportion measured over them has roughly `docs_per_shard` times the variance
    a confidence interval computed on the document count would suggest.

    Each returned document carries its shard path, so any row on the site can
    link back to the exact file on the Hub.
    """
    shards = shards if shards is not None else list_shards(dataset, revision)
    rng = random.Random(seed)
    weights = [s["size"] for s in shards]

    def fetch(pick):
        i, shard = pick
        try:
            docs = read_shard_head(dataset, shard["path"], revision, read_bytes)
            # A head that decodes to nothing usable means the first document is
            # longer than the read. Common in the long-context mixes, where
            # single documents run past 200k characters, and silently dropping
            # those shards would bias the sample against exactly the documents
            # the stage exists to train on. Pay for one larger read there only.
            if not any((d.get("text") or "").strip() for d in docs):
                docs = read_shard_head(
                    dataset, shard["path"], revision, read_bytes * LONG_DOC_FACTOR
                )
            return i, shard, docs
        except requests.RequestException:
            # One unreachable shard out of hundreds should cost a document, not
            # the whole run — the next batch covers the shortfall.
            return i, shard, []

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    pick_no = 0

    # Draw in batches until n documents are in hand rather than betting the whole
    # sample on one over-draw. How many documents a shard head yields varies by
    # two orders of magnitude — hundreds for stack_edu, one for a long-context
    # shard whose first document is 200k characters — so a picks count derived
    # from docs_per_shard alone can fall far short of what was asked for.
    for _ in range(MAX_BATCHES):
        if len(out) >= n:
            break
        short = n - len(out)
        batch = rng.choices(
            shards, weights=weights, k=max(1, int(short / docs_per_shard * 1.3) + 1)
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, shard, docs in ex.map(
                fetch, ((pick_no + j, s) for j, s in enumerate(batch))
            ):
                if progress:
                    progress(len(out), n, shard["path"])
                # Seeded on the pick index as well as the shard, so the draw stays
                # deterministic regardless of thread completion order AND a shard
                # drawn twice — which byte-weighting makes common, a 21 GB mix like
                # lc_synth-rex_s2pdf spreads over only 55 shards — yields a different
                # document each time instead of the same one twice.
                random.Random(f"{seed}:{i}:{shard['path']}").shuffle(docs)
                kept = 0
                for doc in docs:
                    if kept >= docs_per_shard:
                        break
                    text = (doc.get("text") or "").strip()
                    if not text:
                        continue
                    # Belt and braces: independent draws can still collide by
                    # chance, and the same document twice is not a second
                    # observation.
                    key = (shard["path"], doc.get("id") or text[:200])
                    if key in seen:
                        continue
                    seen.add(key)
                    kept += 1
                    out.append(
                        {
                            "id": doc.get("id"),
                            "text": text,
                            "source": shard["source"],
                            "topic": shard["topic"],
                            "shard": shard["path"],
                            "metadata": _metadata(doc),
                        }
                    )
        pick_no += len(batch)

    rng.shuffle(out)
    return out[:n]


# Dolma 3 keeps its per-document provenance in a JSON string. These are the
# fields worth surfacing: which crawl it came from, what the quality classifier
# scored it, how confident the language ID was, and how many exact duplicates
# it had. Everything else is filter bookkeeping.
def _metadata(doc: dict) -> dict:
    raw = doc.get("metadata")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    if raw.get("cc_dump"):
        out["cc_dump"] = raw["cc_dump"]
    qc = raw.get("dolma2_qc")
    if isinstance(qc, dict) and "1" in qc:
        out["quality_score"] = round(float(qc["1"]), 4)
    lang = raw.get("lang")
    if isinstance(lang, dict) and lang:
        top = max(lang.items(), key=lambda kv: kv[1])
        out["lang"] = top[0]
        out["lang_score"] = round(float(top[1]), 4)
    if raw.get("original_word_count"):
        out["word_count"] = raw["original_word_count"]
    if raw.get("exact_duplicates") is not None:
        out["exact_duplicates"] = raw["exact_duplicates"]
    return out
