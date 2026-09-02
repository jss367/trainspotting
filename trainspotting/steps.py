"""Count a string in the batches a model actually saw, step by step.

Every other layer reads a corpus or a mix as a set: how much of it holds a
string, which source it concentrates in. Order is invisible to a set, and for
one model in the registry the order is public. EleutherAI trained every Pythia
size on the same tokens in the same sequence and released that sequence:
`EleutherAI/pile-deduped-pythia-preshuffled` is the deduplicated Pile as
GPT-NeoX tokenized and shuffled it, 143,000 steps of 1,024 sequences of 2,049
token ids, in the order the optimizer took them. Step s is bytes
[s·4,196,352, (s+1)·4,196,352) of the concatenated shards. That makes "when did
the model see this" a range request rather than a download: one step is 4.2 MB
and the 600 GB around it stays where it is.

The reason to want the order is what has been measured on these checkpoints.
Timaeus's influence-dynamics work finds that a training example's pull on a
behaviour is not constant over training: it peaks around developmental
transitions and can change sign. A rate over the corpus is therefore an
exposure and not an effect, and the quantity that carries across is how much of
that exposure had happened by the step a given checkpoint was saved at. This
layer measures a string's density along the run and turns it into that:
expected sequences holding the string, seen by each of the 154 published
checkpoints.

What it can and cannot see. The unit is a training sequence, not a document.
Documents are concatenated with no separator — there is no end-of-text token
anywhere in these batches, one document's last sentence runs straight into the
next one's title — and cut into 2,049-token sequences, so one sequence can hold
the ends of several documents, and a string that falls across a sequence
boundary is missed. Steps are sampled, one from each equal slice of the run, so
the per-step counts are a curve with an interval and not a census: a string the
sample never lands on gets an upper bound here and no first-seen step. Finding
the exact steps a rare string appears at would mean scanning the 600 GB, and
this layer does not.
"""

import array
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from . import hf, pretrain
from .stats import cluster_wilson, wilson

# Characters kept either side of a match in a stored example. Enough to read the
# sentence; not enough to republish the document.
SNIPPET_CHARS = 120

# Range requests in flight at once. Each is one step, 4.2 MB, from a CDN that
# serves them at a few seconds apiece; four keeps the decoder fed without
# turning a run into a burst the hub rate-limits.
WORKERS = 4

TOKENIZER_API = "https://huggingface.co/api/models/{repo}/revision/{revision}"
TOKENIZER_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json"

CAVEAT = (
    "The unit is a 2,049-token training sequence, not a document. Documents are "
    "concatenated with no separator and cut into sequences, so one sequence can "
    "hold the ends of several documents, and a string falling across a sequence "
    "boundary is missed. Steps are sampled, one from each equal slice of the run, "
    "and every sequence of a sampled step is read, so the interval is clustered by "
    "step. The exposure figures assume the sampled rate holds across the steps not "
    "read; the per-slice rates are the check on that."
)


def step_bytes(order: dict) -> int:
    """Bytes one training step occupies in the concatenated shards."""
    return order["sequences_per_step"] * order["sequence_tokens"] * order["dtype_bytes"]


def total_bytes(order: dict) -> int:
    return (order["shards"] - 1) * order["shard_bytes"] + order["last_shard_bytes"]


def check_layout(order: dict) -> None:
    """Refuse a layout whose arithmetic does not close.

    Every offset below is `step × step_bytes`, which is only an address if the
    shards hold exactly `steps` steps and nothing else. A registry entry with one
    constant wrong would otherwise read the right number of bytes from the wrong
    place and decode them into text nobody trained on.
    """
    expected = order["steps"] * step_bytes(order)
    if expected != total_bytes(order):
        raise ValueError(
            f"{order['dataset']}: {order['steps']:,} steps × {step_bytes(order):,} bytes = "
            f"{expected:,}, but the shards total {total_bytes(order):,}"
        )


def segments(order: dict, step: int) -> list[tuple[int, int, int]]:
    """(shard, first byte, last byte) for each shard a step touches.

    Shards are a 30 GB split of one stream, not aligned to steps, so a step near
    a boundary has its tail in the next shard. Inclusive byte ranges, as HTTP
    Range headers are written.
    """
    if not 0 <= step < order["steps"]:
        raise ValueError(f"step {step:,} is outside 0–{order['steps'] - 1:,}")
    start = step * step_bytes(order)
    end = start + step_bytes(order) - 1
    out = []
    pos = start
    while pos <= end:
        shard, offset = divmod(pos, order["shard_bytes"])
        last = min(end, (shard + 1) * order["shard_bytes"] - 1)
        out.append((shard, offset, last - shard * order["shard_bytes"]))
        pos = last + 1
    return out


def shard_path(order: dict, shard: int) -> str:
    return order["file"].format(shard=shard)


def fetch_step(order: dict, step: int, revision: str, get=pretrain._get) -> array.array:
    """The token ids of one step, as the optimizer saw them.

    Pinned to a revision so the bytes are the bytes: `main` moving under a run
    would put this step's tokens at a different offset.
    """
    buf = bytearray()
    for shard, first, last in segments(order, step):
        url = pretrain.HF_RESOLVE.format(
            dataset=order["dataset"], revision=revision, path=shard_path(order, shard)
        )
        r = get(url, headers={**hf.HEADERS, "Range": f"bytes={first}-{last}"})
        if len(r.content) != last - first + 1:
            raise RuntimeError(
                f"step {step:,}: asked shard {shard} for bytes {first}-{last} and got "
                f"{len(r.content):,} bytes back (HTTP {r.status_code})"
            )
        buf += r.content
    tokens = array.array("H")
    tokens.frombytes(bytes(buf))
    if sys.byteorder == "big":
        tokens.byteswap()
    return tokens


def sequences(tokens: array.array, order: dict) -> list[list[int]]:
    """A step's ids cut back into the sequences the batch was made of."""
    n = order["sequence_tokens"]
    return [tokens[i : i + n].tolist() for i in range(0, len(tokens), n)]


def draw_steps(total: int, n: int, seed: int, at=()) -> list[int]:
    """One step from each of `n` equal slices of the run, plus any asked for by name.

    Stratified rather than uniform so the sample spans the run: `--sample 8`
    drawn uniformly can land all eight in the first half, and the point of this
    layer is the whole axis. Within a slice the pick is uniform, so nothing
    about a particular step's position is special. Sorted, because that is the
    order training happened in and the order the output reads in.
    """
    if n > total:
        raise ValueError(f"asked for {n:,} steps of a {total:,}-step run")
    rng = random.Random(seed)
    picks = {rng.randrange(i * total // n, (i + 1) * total // n) for i in range(n)}
    for s in at:
        if not 0 <= s < total:
            raise ValueError(f"step {s:,} is outside 0–{total - 1:,}")
        picks.add(s)
    return sorted(picks)


def compile_pattern(pattern: str, regex: bool = False, case_sensitive: bool = False) -> re.Pattern:
    """Same contract as `grep`: a literal unless --regex, case-insensitive unless asked."""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern if regex else re.escape(pattern), flags)


def snippet(text: str, start: int, end: int) -> str:
    return text[max(0, start - SNIPPET_CHARS) : end + SNIPPET_CHARS]


def count_step(step: int, texts: list[str], rx: re.Pattern, examples: list, limit: int) -> dict:
    """Sequences of one step holding the pattern, and how many times over.

    Two counts because they answer different questions: `matched` is the share
    of training the string is in, `occurrences` is how often the objective saw
    it, and a sequence that is one document repeating a phrase forty times moves
    the second and not the first.
    """
    matched = occurrences = 0
    for i, text in enumerate(texts):
        hits = list(rx.finditer(text))
        if not hits:
            continue
        matched += 1
        occurrences += len(hits)
        if len(examples) < limit:
            m = hits[0]
            examples.append({"step": step, "sequence": i, "snippet": snippet(text, m.start(), m.end())})
    return {"step": step, "sequences": len(texts), "matched": matched, "occurrences": occurrences}


def scan(
    order,
    steps_,
    revision,
    rx,
    decode,
    examples_limit=20,
    progress=None,
    workers=WORKERS,
    priority_steps=(),
):
    """Fetch, decode and count each step, in training order.

    Fetches overlap; decoding and counting run in order on the main thread so
    `per_step` comes back sorted and the progress line reads as a walk through
    the run.
    """
    per_step, ordinary_examples, priority_examples = [], [], []
    priority_steps = set(priority_steps)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = pool.map(lambda s: fetch_step(order, s, revision), steps_)
        for i, (step, tokens) in enumerate(zip(steps_, fetched), 1):
            texts = decode(sequences(tokens, order))
            examples = priority_examples if step in priority_steps else ordinary_examples
            per_step.append(count_step(step, texts, rx, examples, examples_limit))
            if progress:
                progress(i, len(steps_), step)
    # An exact step is requested for inspection, so its evidence gets first use
    # of the same bounded result budget even when it sorts after sampled steps.
    # Each temporary list is itself capped, keeping memory bounded as well.
    return per_step, (priority_examples + ordinary_examples)[:examples_limit]


def _records(per_step: list[dict]) -> list[dict]:
    """One record per sequence, so `cluster_wilson` can group them by step."""
    return [
        {"match": 1 if j < c["matched"] else 0, "step": str(c["step"])}
        for c in per_step
        for j in range(c["sequences"])
    ]


def design_effect(per_step: list[dict]) -> float | None:
    """How much the clustering by step widens the interval, measured; None where
    it cannot be measured (one step, or every sequence agreeing)."""
    n = sum(c["sequences"] for c in per_step)
    k = sum(c["matched"] for c in per_step)
    if len(per_step) < 2 or k in (0, n):
        return None
    _, _, n_eff = cluster_wilson(_records(per_step), key="step")
    return n / n_eff


def summarize(per_step: list[dict], deff: float | None = None) -> dict:
    """The share of sequences holding the string, with an interval clustered by step.

    Every sequence of a sampled step is read, so the observations arrive in
    clusters of 1,024, and the interval is Wilson at the effective sample size
    n / design effect. The design effect is measured over the steps rather than
    assumed, and a caller with a better estimate — the whole run's, for one
    slice of it — passes it in.

    Where it cannot be measured, this layer uses 1 — every sequence counts —
    which is the opposite of what `stats.cluster_wilson` does for the shard
    sampler, and for a reason. A shard's documents share a topic, so a unanimous
    shard is one observation. A step is 1,024 sequences of a stream that was
    shuffled over the whole corpus before it was cut, so a step is 1,024 draws
    and not a cluster; the design effect measured wherever it can be is 1.0, and
    a unanimous zero over eight steps is 8,192 sequences without the string and
    not eight. Treating it as eight would put the upper bound on a string never
    seen at a third of the corpus.
    """
    n = sum(c["sequences"] for c in per_step)
    k = sum(c["matched"] for c in per_step)
    occurrences = sum(c["occurrences"] for c in per_step)
    if n == 0:
        return {
            "steps": 0, "sequences": 0, "matched": 0, "occurrences": 0,
            "rate": None, "lo": None, "hi": None, "design_effect": None, "n_effective": 0.0,
        }
    if deff is None:
        deff = design_effect(per_step) or 1.0
    n_eff = n / deff
    lo, hi = wilson(k / n * n_eff, n_eff)
    return {
        "steps": len(per_step), "sequences": n, "matched": k, "occurrences": occurrences,
        "rate": k / n, "lo": lo, "hi": hi, "design_effect": deff, "n_effective": n_eff,
    }


def by_slice(per_step: list[dict], total_steps: int, n: int, deff: float | None = None) -> list[dict]:
    """The same summary over each of `n` equal stretches of the run.

    This is the check on the assumption `exposure` makes. A shuffled order should
    show the same rate in every stretch; a stretch that does not — the second
    pass over a corpus smaller than the run, say — is visible here and nowhere
    else. The design effect is the whole run's: eight steps is too few to
    measure one, and the clustering is a property of the draw, not of the
    stretch.
    """
    out = []
    for i in range(n):
        lo_step, hi_step = i * total_steps // n, (i + 1) * total_steps // n
        inside = [c for c in per_step if lo_step <= c["step"] < hi_step]
        out.append({"from_step": lo_step, "to_step": hi_step - 1, **summarize(inside, deff=deff)})
    return out


def exposure(summary: dict, checkpoints: list[int], sequences_per_step: int) -> list[dict]:
    """Expected sequences holding the string that the model had seen by each checkpoint.

    Rate times sequences seen, so a straight line through the run, with the
    interval carried along. That is the honest shape for a sample: the order is
    shuffled, and nothing measured here says a particular checkpoint saw more
    than its share. What the figure adds over a corpus rate is the axis itself,
    which is the one the developmental work on these checkpoints is drawn on.
    A zero-match run still has an upper bound at every step.
    """
    out = []
    for step in checkpoints:
        seen = step * sequences_per_step
        rec = {"step": step, "sequences_seen": seen}
        for key in ("rate", "lo", "hi"):
            value = summary.get(key)
            rec["expected" if key == "rate" else key] = None if value is None else value * seen
        out.append(rec)
    return out


def second_pass_step(order: dict) -> int | None:
    """The step from which the run is re-reading documents, if the corpus is
    smaller than the token budget. Approximate: the corpus size is the paper's
    round figure, and it is divided by the new tokens a step adds, which is one
    fewer than a sequence holds because the last token is only a label."""
    corpus = order.get("corpus_tokens")
    if not corpus:
        return None
    per_step = order["sequences_per_step"] * (order["sequence_tokens"] - 1)
    step = math.ceil(corpus / per_step)
    return step if step < order["steps"] else None


def resolve_tokenizer_revision(repo: str, revision: str = "main") -> str:
    """The immutable model-repository commit holding ``tokenizer.json``."""
    url = TOKENIZER_API.format(repo=repo, revision=revision)
    return pretrain._get(url, headers=hf.HEADERS).json()["sha"]


def decoder(order: dict, revision: str):
    """Token ids -> text, with the tokenizer the batches were encoded with.

    The tokenizer file is fetched once into the shard cache. `tokenizers` is an
    optional extra because this is the only layer that ever holds a token id;
    everything else in the tool reads text.
    """
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError:
        sys.exit(
            "trainspotting steps needs tokenizers: pip install 'tokenizers>=0.15'\n"
            "(it is the only layer that decodes token ids, so it is an optional extra: "
            "pip install -e '.[steps]')"
        )
    repo = order["tokenizer"]
    path = pretrain.CACHE_DIR / (
        f"tokenizer__{repo.replace('/', '__')}@{revision}.json"
    )
    if not path.exists():
        r = pretrain._get(
            TOKENIZER_RESOLVE.format(repo=repo, revision=revision),
            headers=hf.HEADERS,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    tok = Tokenizer.from_file(str(path))
    return lambda seqs: tok.decode_batch(seqs, skip_special_tokens=False)
