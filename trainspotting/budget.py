"""Put every stage's match rate on one scale: tokens the model was fit to.

`ask` answers "what share of this stage matches the question" — a share of rows
for a post-training stage, a share of documents for a corpus. Those shares are
not comparable to each other and neither is a measure of how much training a
model got. Olmo 3 7B sees 5.93T pretraining tokens and about 4B tokens of SFT
target text; a 6% rate over Dolci-Instruct-DPO and a 1% rate over Dolma 3 are
three orders of magnitude apart in what they represent.

This layer multiplies the two together. For each stage it computes

    matching tokens  =  rate × stage size

and sums them, so a question asked across the pipeline gets one number and a
share of the whole budget.

## The unit

**Tokens the model was fit to** — loss-bearing tokens. That is the only unit
under which the stages mean the same thing, and it is not the same as the size
of the dataset:

| Stage | Fit to |
|---|---|
| pretrain / midtrain / long-context | every token; the stage size from the Olmo 3 paper is the answer |
| sft | the assistant turns, reasoning spans included — not the prompt it read |
| dpo | both completions; the preference loss is computed over the pair |
| rlvr | rollouts generated during training, which the dataset does not contain |

RLVR is the honest gap. The published mix holds prompts, verifiers and some
reference generations, not the text the policy was actually trained on, and how
many rollouts per prompt the run took is not in the data. What this reports for
an RL stage is a **floor**: one reference rollout per prompt. Real training
samples several, so the true figure is larger by whatever that factor was.

## The rate

Weighted by fit characters, not by rows. A stage's matching examples are not
average-length — in Dolci-Think-SFT a reasoning trace runs tens of thousands of
characters and a one-line prompt does not — so counting rows would answer "what
fraction of examples" when the question is "what fraction of training".

    char rate  =  Σ fit chars over matching examples  /  Σ fit chars over all judged

The interval is the count-based one (cluster-corrected for corpora, where the
`ask` run already stored it) rescaled by char-rate / count-rate. It carries the
sampling uncertainty in the *rate* and not the extra uncertainty in the length
ratio, so it is narrower than the truth by that much.

## What it does not do

It weighs tokens, not learning. A DPO preference token, an RL gradient step and
a pretraining cross-entropy token do not move a model equally, and post-training
is widely believed to be far higher-leverage per token than pretraining. Nothing
here corrects for that, so read the output as an exposure budget rather than as
an attribution of behaviour.
"""

import json

from . import extract, paths, registry
from .stats import cluster_wilson, wilson

# Characters per token, for turning a measured character count into the unit the
# registry's stage sizes are in. Roughly right for English prose and roughly
# wrong for code and CJK, both of which are in these mixes. Every figure derived
# through it is an estimate and is printed as one; the ratios between stages,
# which is what the comparison actually turns on, are much less sensitive to it
# than the absolute counts.
CHARS_PER_TOKEN = 4.0

# How each kind of example's fit text is described in the output. Keyed by
# registry.stage_kind.
FIT_TEXT = {
    "sft": "assistant turns (reasoning included)",
    "dpo": "both completions of the pair",
    "rlvr": "reference rollouts (one per prompt — a floor)",
    "chat": "nothing; a log is not a training example",
}


def _turn_chars(turn: dict) -> int:
    """True length of one stored turn, reasoning span included.

    `context` splits a `<think>` span out of the turn it precedes so the answer
    stays visible under truncation, but the model was fit to both, and in a
    think mix the reasoning is most of the length. Counting only the answer
    would understate Dolci-Think-SFT by about 20x.

    `chars` is the length before the 4,000-character display cut, so these are
    the real lengths and not what the record shows.
    """
    return turn.get("chars", 0) + (turn.get("reasoning") or {}).get("chars", 0)


def fit_chars(rec: dict) -> int | None:
    """Characters of `rec` the model was trained to produce, or None if the
    dataset does not hold them.

    None is not zero. An RL row has a target — the rollouts the policy sampled —
    and simply does not ship it; scoring that as zero would report an RL stage as
    weightless, which is the opposite of what it is.
    """
    kind = rec.get("kind")
    if kind == "sft":
        return sum(_turn_chars(t) for t in rec.get("turns", []) if t.get("role") == "assistant")
    if kind == "dpo":
        return sum(
            _turn_chars(t)
            for side in ("chosen", "rejected")
            for t in (rec.get(side) or {}).get("turns", [])
            if t.get("role") == "assistant"
        )
    if kind == "rlvr":
        sample = (rec.get("rollouts") or {}).get("sample")
        return sample["chars"] if sample else None
    return None  # chat: a log, fit to nothing


# --- reading previous runs back ------------------------------------------


def load(name: str) -> dict | None:
    """A committed result file by name, parsed, or None if this checkout has none.

    Every reader of a previous run goes through here so they all fall back from
    `results/` to the committed `docs/data/` copy the same way.
    """
    path = paths.find(name)
    return json.loads(path.read_text()) if path else None


def load_context(target_name: str, stage: str) -> dict:
    """Context records for a stage, keyed both ways a result record can address one.

    Newer result records carry the row's absolute index; ones committed before
    that join on the first 400 characters of the prompt. Both keys go in the same
    map so a caller does not have to know which vintage it is holding — see
    `context.build`.
    """
    data = load(f"{target_name}.{stage}.context.json")
    if not data:
        return {}
    out = {}
    for rec in data["records"]:
        if rec.get("row") is not None:
            out[("row", rec["row"])] = rec
        out.setdefault(("key", rec["key"]), rec)
    return out


def context_for(rec: dict, by_key: dict) -> dict | None:
    """The context record a result record addresses, by row if it has one."""
    if rec.get("row") is not None and ("row", rec["row"]) in by_key:
        return by_key[("row", rec["row"])]
    return by_key.get(("key", rec["prompt"][: 400]))


def stage_rows(target_name: str, stage: str) -> int | None:
    """Exact row count of a post-training stage, from the committed `sources` run.

    Offline on purpose: a budget is a rollup of runs that already happened, and
    it should not need the network to add them up. A target with no `sources`
    run yet gets None and the stage reports its size as unknown rather than
    guessing one.
    """
    data = load(f"{target_name}.sources.json") or {}
    return (data.get(stage) or {}).get("total")


# --- the estimate ---------------------------------------------------------


def _rate(matched_chars: float, total_chars: float) -> float:
    return matched_chars / total_chars if total_chars else 0.0


def _post_training_stage(target_name: str, stage: dict, slug: str) -> dict:
    name = stage["stage"]
    kind = registry.stage_kind(stage)
    out = {
        "stage": name,
        "name": stage["name"],
        "kind": kind,
        "family": "post-training",
        "dataset": stage["hf_dataset"],
        "fit_text": FIT_TEXT.get(kind, "unknown"),
        "notes": [],
    }
    ask = load(f"{target_name}.{name}.ask-{slug}.json")
    if not ask:
        out["measured"] = False
        return out
    records = ask["records"]
    by_key = load_context(target_name, name)
    joined = [(r, context_for(r, by_key)) for r in records]
    unjoined = sum(1 for _, c in joined if c is None)
    fit = [(r, fit_chars(c)) for r, c in joined if c is not None]
    weighable = [(r, f) for r, f in fit if f is not None]

    k, n = sum(bool(r["match"]) for r in records), len(records)
    lo, hi = wilson(k, n)
    total_chars = sum(f for _, f in weighable)
    matched_chars = sum(f for r, f in weighable if r["match"])
    char_rate = _rate(matched_chars, total_chars)
    count_rate = k / n if n else 0.0

    out.update(
        {
            "measured": True,
            "question": ask["question"],
            "n": n,
            "matched": k,
            "count_rate": count_rate,
            "count_ci": [lo, hi],
            "weighed": len(weighable),
            "fit_chars_total": total_chars,
            "fit_chars_matched": matched_chars,
            "char_rate": char_rate,
        }
    )
    if unjoined:
        out["notes"].append(
            f"{unjoined} of {n} judged records have no stored training example, so the"
            " rate below is weighed over the rest"
        )
    if len(weighable) < len(fit):
        # RL is where this bites: a row with no stored reference generation has
        # no length to weigh by, and for Dolci-Instruct-RL that is most of them.
        out["notes"].append(
            f"{len(fit) - len(weighable)} of {len(fit)} examples store no text the model"
            " was fit to, so they carry no weight in the rate"
        )

    rows = stage_rows(target_name, name)
    if rows is None or not weighable:
        out["notes"].append(
            f"stage size unknown — run `trainspotting sources {target_name} --json`"
            if rows is None
            else "stage size unknown — no example stores text the model was fit to"
        )
        return out
    mean_fit = total_chars / len(weighable)
    out["rows"] = rows
    out["mean_fit_chars"] = mean_fit
    out["size_tokens"] = rows * mean_fit / CHARS_PER_TOKEN
    out["size_basis"] = (
        f"{rows:,} rows x {mean_fit:,.0f} mean chars of {out['fit_text']} / {CHARS_PER_TOKEN:g}"
    )
    # An RL stage's real target is the rollouts the policy sampled during
    # training, and the mix ships reference generations instead. One per prompt
    # is the smallest defensible reading of it.
    out["size_is_floor"] = kind == "rlvr"
    _apply_rate(out)
    return out


def _pretrain_stage(target_name: str, stage: dict, slug: str) -> dict:
    name = stage["stage"]
    out = {
        "stage": name,
        "name": stage["name"],
        "kind": "corpus",
        "family": "pretrain",
        "dataset": stage.get("sample_dataset") or stage.get("hf"),
        "fit_text": "every token",
        "size_tokens": stage["tokens"],
        "size_basis": "Olmo 3 paper / release blog",
        "size_is_floor": False,
        "notes": [],
    }
    ask = load(f"{target_name}.{name}.ask-{slug}.json")
    if not ask:
        out["measured"] = False
        return out
    records = ask["records"]
    n = len(records)
    k = sum(bool(r["match"]) for r in records)
    # The corpus interval is cluster-corrected and the `ask` run already stored
    # it. Recompute only for a run written before that field existed, and say so
    # rather than silently reporting the narrow one.
    if ask.get("ci"):
        lo, hi = ask["ci"]
    else:
        lo, hi, _ = cluster_wilson(records)
        out["notes"].append("interval recomputed here; the ask run stored none")

    lengths = _doc_lengths(target_name, name, records)
    total_chars = sum(lengths)
    matched_chars = sum(c for r, c in zip(records, lengths) if r["match"])
    out.update(
        {
            "measured": True,
            "question": ask["question"],
            "n": n,
            "matched": k,
            "count_rate": k / n if n else 0.0,
            "count_ci": [lo, hi],
            "weighed": sum(1 for c in lengths if c),
            "fit_chars_total": total_chars,
            "fit_chars_matched": matched_chars,
            "char_rate": _rate(matched_chars, total_chars),
        }
    )
    if not total_chars:
        out["notes"].append(
            "no document lengths available, so the rate is weighed by document count"
        )
        out["char_rate"] = out["count_rate"]
    judged = ask.get("judged_chars", extract.MAX_DOCUMENT_CHARS)
    long_docs = sum(1 for c in lengths if c > judged)
    if long_docs:
        # The judgment read an excerpt spanning the document; the weight is the
        # whole thing. That is the right weight — the model trained on all of it —
        # but the two are not the same text and the gap is widest exactly where
        # the weight is largest.
        out["notes"].append(
            f"{long_docs} of {n} documents are longer than the {judged:,} characters"
            " judged, and are weighed by their full length"
        )
    _apply_rate(out)
    return out


def _doc_lengths(target_name: str, stage: str, records: list[dict]) -> list[int]:
    """Full length of each judged document, in the order the ask run scored them.

    `ask --pretrain` records a document's length directly. A run from before
    that gets it by joining back to the sample the documents came from, on the
    excerpt text the run stored — which is exactly what was judged, so it is a
    key rather than a prefix of one.
    """
    if all("chars" in r for r in records):
        return [r["chars"] for r in records]
    docs = load(f"{target_name}.{stage}.docs.json")
    by_text = {d["text"]: d["chars"] for d in (docs or {}).get("records", [])}
    return [r.get("chars") or by_text.get(r["prompt"], 0) for r in records]


# Below this many matching examples, the difference between the row rate and
# the length-weighted rate is one or two examples' lengths rather than a
# measured property of the matching content. The rate is still the best estimate
# available; the point is that the reweighting is not independently supported.
FEW_MATCHES = 10


def _apply_rate(out: dict) -> None:
    """Multiply the measured rate by the stage size, and carry the interval."""
    size = out.get("size_tokens")
    if not out.get("measured") or size is None:
        return
    if 0 < out["matched"] < FEW_MATCHES:
        out["notes"].append(
            f"weighing by length rests on {out['matched']} matching example(s), so the"
            f" gap between the row rate ({out['count_rate'] * 100:.1f}%) and the length"
            f" rate ({out['char_rate'] * 100:.1f}%) is those examples' lengths, not a"
            " measured property of matching content"
        )
    out["matching_tokens"] = out["char_rate"] * size
    # The interval is over the count rate; rescale it by however much weighing
    # by length moved the point estimate, and treat that factor as known. It is
    # not — matching examples being longer is itself measured on 300 draws — so
    # this is narrower than the truth.
    count_rate = out["count_rate"]
    ratio = out["char_rate"] / count_rate if count_rate else 1.0
    lo, hi = out["count_ci"]
    out["matching_tokens_ci"] = [lo * ratio * size, hi * ratio * size]


def estimate(target_name: str, slug: str) -> dict:
    """The whole pipeline's budget for one `ask` question.

    Reads only committed runs, so it costs nothing and answers for exactly the
    samples someone can go and read.
    """
    target = registry.resolve(target_name)
    stages = []
    for stage in registry.pretrain_stages(target):
        stages.append(_pretrain_stage(target_name, stage, slug))
    for stage in registry.post_training_stages(target):
        stages.append(_post_training_stage(target_name, stage, slug))

    question = next((s["question"] for s in stages if s.get("question")), None)
    # A question reworded between runs is two different measurements summed into
    # one total, and nothing else in the output would show it.
    variants = sorted({s["question"] for s in stages if s.get("question")})
    return {
        "target": target_name,
        "is_model": target["is_model"],
        "slug": slug,
        "question": question,
        "question_variants": variants if len(variants) > 1 else [],
        "chars_per_token": CHARS_PER_TOKEN,
        "stages": stages,
        "totals": totals(stages),
    }


def totals(stages: list[dict]) -> dict:
    """Summed sizes and matching tokens, by family and overall.

    A stage whose size is unknown is summed into neither, and counted in
    `unsized` so the share it reports is visibly a share of what was measured.
    """
    out = {}
    for family in ("pretrain", "post-training", "all"):
        rows = [s for s in stages if family == "all" or s["family"] == family]
        sized = [s for s in rows if s.get("size_tokens") is not None]
        priced = [s for s in sized if s.get("matching_tokens") is not None]
        size = sum(s["size_tokens"] for s in sized)
        matching = sum(s["matching_tokens"] for s in priced)
        out[family] = {
            "size_tokens": size,
            "matching_tokens": matching,
            "share": matching / size if size else 0.0,
            "matching_tokens_ci": [
                sum(s["matching_tokens_ci"][i] for s in priced) for i in (0, 1)
            ],
            "stages": len(rows),
            "measured": sum(1 for s in rows if s.get("measured")),
            "unsized": [s["stage"] for s in rows if s.get("size_tokens") is None],
            # A stage counted at its floor makes the total a floor too.
            "floor": [s["stage"] for s in priced if s.get("size_is_floor")],
        }
    return out
