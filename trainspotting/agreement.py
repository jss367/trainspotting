"""Does the classifier say the same thing twice?

Every helpful/honest/harmless share on the site is a Claude label over a sample
of prompts, and the classifier samples at temperature and thinks before it
answers, so two runs on identical prompts are not guaranteed to agree. This
module compares two such runs over the same draw — agreement with its interval,
Cohen's kappa, where the labels moved, and what that did to each label's share —
so a share can be printed next to the number that says how stable it is.

Kappa rather than agreement alone, because most prompts in these mixes are
`capability` or `helpfulness`: two runs that both wrote `capability` on
everything would agree perfectly and have measured nothing. Kappa discounts the
agreement that label frequencies alone would produce.
"""


from . import classify
from .context import KEY_CHARS
from .stats import wilson

LABELS = classify.LABELS


def _join_key(rec: dict):
    """What identifies a labeled prompt across files.

    A labels run written since result records carried a row index joins on it.
    The runs committed before that have only the prompt text, and join the way
    the site joins them: on its first `KEY_CHARS` characters. Two prompts that
    share an opening collapse under it, which a curated mix mostly gets away
    with; the row index exists because a chat log does not.
    """
    if rec.get("row") is not None:
        return ("row", rec["row"])
    return ("key", (rec.get("prompt") or "")[:KEY_CHARS])


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over (a, b) label pairs. None when undefined.

    Undefined when there are no pairs, or when the expected agreement is 1 —
    both raters used a single label throughout — in which case observed
    agreement is 1 too and 0/0 says nothing. That is the honest answer for a
    two-prompt stage where both runs wrote `capability`.
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

    With the first run as reference, precision reads "of the prompts the second
    run called honesty, how many the first did too", and recall "of the prompts
    the first run called honesty, how many the second kept". Both are None where
    the denominator is empty rather than 0 or 1.
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


def compare(first: list[dict], second: list[dict]) -> dict:
    """Two classifier runs over the same draw, joined on row index.

    Rows one run labeled and the other did not are counted as `unpaired`
    rather than as disagreements: a refusal on the second pass is a fact about
    coverage, and the agreement figure should be about the labels both runs
    committed to. Verifier-settled rows are fixed by construction and are
    left out of agreement statistics, so they cannot inflate agreement.
    Headline shares include every labeled row, including verifier-settled rows.
    """
    a = {_join_key(r): r["label"] for r in first if r.get("label") and r.get("by") != "verifier"}
    b = {_join_key(r): r["label"] for r in second if r.get("label") and r.get("by") != "verifier"}
    rows = sorted(set(a) & set(b), key=str)
    pairs = [(a[i], b[i]) for i in rows]
    out = _summary(pairs, "first", "second")
    out["unpaired"] = len(set(a) ^ set(b))
    # What a rerun does to the headline: the share of each label, both times.
    # A kappa of 0.8 can hide a share that moved by a third if the flips all
    # land on one rare label, and the share is what the site prints.
    all_a = [r["label"] for r in first if r.get("label")]
    all_b = [r["label"] for r in second if r.get("label")]
    out["shares"] = {
        label: {
            "first": sum(v == label for v in all_a) / len(all_a) if all_a else None,
            "second": sum(v == label for v in all_b) / len(all_b) if all_b else None,
        }
        for label in LABELS
        if label in all_a or label in all_b
    }
    return out
