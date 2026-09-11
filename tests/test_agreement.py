"""The classifier against a second run of itself.

Each test pins a way the arithmetic could quietly flatter the instrument — a
kappa that rewards two runs that wrote one word everywhere, a rerun whose
refusals read as agreement.
"""

from trainspotting import agreement
from trainspotting.classify import LABELS


def _records(labels, start=0):
    return [
        {"row": start + i, "prompt": f"p{start + i}", "label": lab}
        for i, lab in enumerate(labels)
    ]


class TestKappa:
    def test_perfect(self):
        assert agreement.kappa([("a", "a"), ("b", "b")]) == 1.0

    def test_one_label_everywhere_is_undefined_not_perfect(self):
        """Both raters said `capability` on every item. Agreement is 100% and
        it means nothing; kappa has to say so rather than return 1."""
        assert agreement.kappa([("capability", "capability")] * 10) is None

    def test_empty_is_undefined(self):
        assert agreement.kappa([]) is None

    def test_chance_agreement_is_discounted(self):
        # 3 of 4 agree, but the labels are so skewed that chance alone gives
        # most of it. Accuracy 0.75, kappa well under.
        pairs = [("a", "a"), ("a", "a"), ("a", "a"), ("a", "b")]
        assert agreement.kappa(pairs) == 0.0


class TestCompare:
    def test_refusal_on_the_rerun_is_unpaired_not_disagreement(self):
        first = _records(["honesty", "capability", "helpfulness"])
        second = _records(["honesty", "capability", None])
        c = agreement.compare(first, second)
        assert c["n"] == 2 and c["agree"] == 2
        assert c["unpaired"] == 1

    def test_verifier_rows_do_not_inflate_agreement(self):
        first = _records(["instruction_following"] * 5 + ["honesty"])
        second = _records(["instruction_following"] * 5 + ["helpfulness"])
        for r in first[:5] + second[:5]:
            r["by"] = "verifier"
        c = agreement.compare(first, second)
        assert c["n"] == 1 and c["agree"] == 0
        assert c["shares"]["instruction_following"] == {"first": 5 / 6, "second": 5 / 6}
        assert c["shares"]["honesty"] == {"first": 1 / 6, "second": 0}
        assert c["shares"]["helpfulness"] == {"first": 0, "second": 1 / 6}

    def test_all_verifier_run_has_headline_shares_but_no_agreement(self):
        records = _records(["instruction_following"] * 3)
        for record in records:
            record["by"] = "verifier"
        c = agreement.compare(records, records)
        assert c["n"] == 0 and c["accuracy"] is None and c["kappa"] is None
        assert c["shares"] == {"instruction_following": {"first": 1, "second": 1}}

    def test_headline_counts_all_records_despite_legacy_join_key_collisions(self):
        first = [{"prompt": "same", "label": label} for label in ["honesty", "capability"]]
        second = [{"prompt": "same", "label": "honesty"}]
        c = agreement.compare(first, second)
        assert c["shares"]["honesty"] == {"first": 0.5, "second": 1}
        assert c["shares"]["capability"] == {"first": 0.5, "second": 0}


    def test_shares_show_what_a_rerun_does_to_the_headline(self):
        first = _records(["honesty", "honesty", "capability", "capability"])
        second = _records(["honesty", "capability", "capability", "capability"])
        c = agreement.compare(first, second)
        assert c["shares"]["honesty"] == {"first": 0.5, "second": 0.25}
        assert c["confusion"] == {"honesty": {"honesty": 1, "capability": 1}, "capability": {"capability": 2}}


def test_old_shape_records_join_on_the_prompt_prefix():
    """The committed Olmo labels predate the row index: records carry only the
    prompt and its label, and a replicate has to find its way back without a row."""
    recs = [{"prompt": "alpha " * 100, "label": "honesty"}, {"prompt": "beta", "label": "capability"}]
    rep = [{"prompt": "beta", "label": "helpfulness"}, {"prompt": "alpha " * 100, "label": "honesty"}]
    c = agreement.compare(recs, rep)
    assert c["n"] == 2 and c["agree"] == 1 and c["unpaired"] == 0


def test_taxonomy_is_the_classifiers():
    """Shares are reported per label in taxonomy order, the classifier's own."""
    assert agreement.LABELS is LABELS
