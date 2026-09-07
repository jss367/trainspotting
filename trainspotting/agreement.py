"""Is the classifier right, and does it say the same thing twice?

Every helpful/honest/harmless share on the site is a Claude label over a sample
of prompts. Nothing else in the repo checks that instrument: no human has labeled
the same prompts, and no run has been repeated to see how many labels move. This
module holds the arithmetic for both checks, so a share can be printed next to
the two numbers that say how far to trust it.

    gold     a stratified, blind draw of already-labeled prompts for a person
             to label by hand under the same rubric the classifier used
    score    how the classifier's labels agree with those human labels —
             accuracy with its interval, Cohen's kappa, and where the two
             disagree, label by label
    compare  the same arithmetic between two classifier runs over the same
             rows, which is what "stability" means here: the classifier
             samples at temperature and thinks before it answers, so two
             runs on identical prompts are not guaranteed to agree

Kappa rather than accuracy alone, because most prompts in these mixes are
`capability` or `helpfulness`: a labeler who wrote `capability` on everything
would score high on agreement and has measured nothing. Kappa discounts the
agreement that label frequencies alone would produce.

The draw is stratified by the classifier's own label so the rare labels are in
the set at all. `honesty` is a few percent of a stage; a uniform draw of fifty
prompts would hold one or two, and a precision figure for it would be a coin
flip. Stratifying makes the per-label figures readable and the overall accuracy
*not* a population estimate — it is over-weighted toward rare labels by design,
and `score` says so in the record it writes.
"""

import random

from . import classify
from .stats import wilson

LABELS = classify.LABELS


def draw_gold(records: list[dict], per_label: int = 8, seed: int = 0) -> list[dict]:
    """A blind, stratified draw of prompts for hand labeling.

    `records` is the `records` list of a committed labels file. Rows the
    verifier settled (`by == "verifier"`) are not the classifier's judgment and
    are left out; rows it never labeled are left out too, there being nothing
    to agree with. Up to `per_label` rows per label, all of them for a label
    with fewer, shuffled so the order carries no hint. The classifier's label is
    deliberately not on the item: a labeler who can see it is checking it, not
    labeling.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in records:
        if r.get("label") is None or r.get("by") == "verifier":
            continue
        by_label.setdefault(r["label"], []).append(r)
    drawn = []
    # Walk the labels in taxonomy order rather than dict order, so the same
    # seed draws the same set whatever order the labels happened to appear in.
    for label in LABELS:
        pool = sorted(by_label.get(label, []), key=lambda r: r["row"])
        rng.shuffle(pool)
        drawn.extend(pool[:per_label])
    rng.shuffle(drawn)
    return [{"row": r["row"], "prompt": r["prompt"], "human_label": None} for r in drawn]


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over (a, b) label pairs. None when undefined.

    Undefined when there are no pairs, or when the expected agreement is 1 —
    both raters used a single label throughout — in which case observed
    agreement is 1 too and 0/0 says nothing. That is the honest answer for a
    two-item gold set where everything was `capability`.
    """
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}
    for a, b in pairs:
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1
    expected = sum(counts_a[k] * counts_b.get(k, 0) for k in counts_a) / (n * n)
    if expected >= 1:
        return None
    return (observed - expected) / (1 - expected)


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """{first: {second: count}} over the labels that actually occur."""
    out: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        out.setdefault(a, {})
        out[a][b] = out[a].get(b, 0) + 1
    return out


def _per_label(pairs: list[tuple[str, str]], reference: str, other: str) -> dict[str, dict]:
    """Precision and recall of `other` against `reference`, per label.

    For a gold set the reference is the person; precision then reads "of the
    prompts the classifier called honesty, how many the person did too", and
    recall "of the prompts the person called honesty, how many the classifier
    found". Both are None where the denominator is empty rather than 0 or 1.
    """
    out = {}
    for label in LABELS:
        ref_n = sum(a == label for a, _ in pairs)
        oth_n = sum(b == label for _, b in pairs)
        both = sum(a == label and b == label for a, b in pairs)
        if not ref_n and not oth_n:
            continue
        out[label] = {
            reference: ref_n,
            other: oth_n,
            "precision": both / oth_n if oth_n else None,
            "recall": both / ref_n if ref_n else None,
        }
    return out


def _summary(pairs: list[tuple[str, str]], first: str, second: str) -> dict:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    lo, hi = wilson(agree, n) if n else (0.0, 0.0)
    return {
        "n": n,
        "agree": agree,
        "accuracy": agree / n if n else None,
        "accuracy_ci": [lo, hi],
        "kappa": kappa(pairs),
        "confusion": _confusion(pairs),
        "per_label": _per_label(pairs, first, second),
    }


def score(gold_items: list[dict], records: list[dict]) -> dict:
    """The classifier against a hand-labeled gold set, joined on row index.

    An item with no `human_label` is a prompt nobody has labeled yet; it is
    counted, not scored, so a half-finished gold set reports what it has. An
    item whose row is not in `records` means the labels file was regenerated
    since the draw (a different seed, a republished dataset) and the two no
    longer describe the same prompts; those are counted too and the caller
    should treat a non-zero count as a reason to redraw, not a rounding error.
    A human label outside the taxonomy is a typo in the gold file and is
    reported by value so it can be fixed rather than silently dropped.
    """
    by_row = {r["row"]: r.get("label") for r in records}
    pairs = []
    unlabeled = 0
    missing = 0
    invalid: dict[str, int] = {}
    for item in gold_items:
        human = item.get("human_label")
        if human is None or human == "":
            unlabeled += 1
            continue
        if human not in LABELS:
            invalid[human] = invalid.get(human, 0) + 1
            continue
        if item["row"] not in by_row or by_row[item["row"]] is None:
            missing += 1
            continue
        pairs.append((human, by_row[item["row"]]))
    out = _summary(pairs, "human", "classifier")
    out.update(
        {
            "gold_items": len(gold_items),
            "unlabeled_gold": unlabeled,
            "missing_rows": missing,
            "invalid_labels": invalid,
            # The draw was stratified by label, so the accuracy above is over a
            # set that over-represents rare labels relative to the stage. It
            # is the right number for "how often is a label right"; it is not
            # the share of the stage that is correctly labeled.
            "stratified": True,
        }
    )
    return out


def compare(first: list[dict], second: list[dict]) -> dict:
    """Two classifier runs over the same draw, joined on row index.

    Rows one run labeled and the other did not are counted as `unpaired`
    rather than as disagreements: a refusal on the second pass is a fact about
    coverage, and the agreement figure should be about the labels both runs
    committed to. Verifier-settled rows are fixed by construction and are
    left out, so they cannot inflate the agreement.
    """
    a = {r["row"]: r["label"] for r in first if r.get("label") and r.get("by") != "verifier"}
    b = {r["row"]: r["label"] for r in second if r.get("label") and r.get("by") != "verifier"}
    rows = sorted(set(a) & set(b))
    pairs = [(a[i], b[i]) for i in rows]
    out = _summary(pairs, "first", "second")
    out["unpaired"] = len(set(a) ^ set(b))
    # What a rerun does to the headline: the share of each label, both times.
    # A kappa of 0.8 can hide a share that moved by a third if the flips all
    # land on one rare label, and the share is what the site prints.
    out["shares"] = {
        label: {
            "first": sum(v == label for v in a.values()) / len(a) if a else None,
            "second": sum(v == label for v in b.values()) / len(b) if b else None,
        }
        for label in LABELS
        if any(v == label for v in a.values()) or any(v == label for v in b.values())
    }
    return out
