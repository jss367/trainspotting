"""The two checks on the classifier: against a person, and against itself.

Each test pins a way the arithmetic could quietly flatter the instrument — a
gold draw that never holds a rare label, a kappa that rewards a labeler who
wrote one word everywhere, a rerun whose refusals read as agreement.
"""

import pytest

from trainspotting import agreement
from trainspotting.classify import LABELS


def _records(labels, start=0):
    return [
        {"row": start + i, "prompt": f"p{start + i}", "label": lab}
        for i, lab in enumerate(labels)
    ]


class TestDrawGold:
    def test_rare_labels_are_in_the_draw(self):
        """One honesty prompt among a hundred capability ones has to come out,
        or the per-label figures are undefined for exactly the labels the
        site leads with."""
        recs = _records(["capability"] * 100 + ["honesty"])
        drawn = agreement.draw_gold(recs, per_label=5)
        rows = {d["row"] for d in drawn}
        assert 100 in rows
        assert len(drawn) == 6

    def test_blind(self):
        drawn = agreement.draw_gold(_records(["honesty", "capability"]), per_label=5)
        for d in drawn:
            assert set(d) == {"row", "prompt", "human_label"}
            assert d["human_label"] is None

    def test_verifier_and_unlabeled_rows_are_left_out(self):
        recs = _records(["honesty", None, "capability"])
        recs[0]["by"] = "verifier"
        drawn = agreement.draw_gold(recs, per_label=5)
        assert [d["row"] for d in drawn] == [2]

    def test_deterministic_in_seed_and_not_in_record_order(self):
        recs = _records(["capability"] * 20 + ["helpfulness"] * 20)
        a = agreement.draw_gold(recs, per_label=3, seed=1)
        b = agreement.draw_gold(list(reversed(recs)), per_label=3, seed=1)
        assert a == b
        assert a != agreement.draw_gold(recs, per_label=3, seed=2)


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


class TestScore:
    def test_joins_on_row_and_reports_what_it_skipped(self):
        recs = _records(["honesty", "capability", "helpfulness", "tool_use"])
        gold = [
            {"row": 0, "human_label": "honesty"},
            {"row": 1, "human_label": "helpfulness"},   # disagreement
            {"row": 2, "human_label": None},            # not yet labeled
            {"row": 3, "human_label": "toolz"},         # typo
            {"row": 99, "human_label": "other"},        # row no longer in the run
        ]
        s = agreement.score(gold, recs)
        assert s["n"] == 2 and s["agree"] == 1
        assert s["unlabeled_gold"] == 1
        assert s["invalid_labels"] == {"toolz": 1}
        assert s["missing_rows"] == 1
        assert s["confusion"] == {"honesty": {"honesty": 1}, "helpfulness": {"capability": 1}}
        assert s["per_label"]["helpfulness"] == {
            "human": 1, "classifier": 0, "precision": None, "recall": 0.0
        }
        assert s["per_label"]["capability"]["precision"] == 0.0
        assert s["stratified"] is True
        lo, hi = s["accuracy_ci"]
        assert lo < 0.5 < hi

    def test_empty_gold_scores_nothing_without_crashing(self):
        s = agreement.score([], _records(["honesty"]))
        assert s["n"] == 0 and s["accuracy"] is None and s["kappa"] is None


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

    def test_shares_show_what_a_rerun_does_to_the_headline(self):
        first = _records(["honesty", "honesty", "capability", "capability"])
        second = _records(["honesty", "capability", "capability", "capability"])
        c = agreement.compare(first, second)
        assert c["shares"]["honesty"] == {"first": 0.5, "second": 0.25}
        assert c["confusion"] == {"honesty": {"honesty": 1, "capability": 1}, "capability": {"capability": 2}}


def test_old_shape_records_join_on_the_prompt_prefix():
    """The committed Olmo labels predate the row index: records carry only the
    prompt and its label, and the gold file drawn from them has to find its way
    back without a row."""
    recs = [{"prompt": "alpha " * 100, "label": "honesty"}, {"prompt": "beta", "label": "capability"}]
    drawn = agreement.draw_gold(recs, per_label=5)
    assert all(d["row"] is None for d in drawn)
    for d in drawn:
        d["human_label"] = "honesty" if d["prompt"].startswith("alpha") else "helpfulness"
    s = agreement.score(drawn, recs)
    assert s["n"] == 2 and s["agree"] == 1 and s["missing_rows"] == 0
    assert agreement.compare(recs, recs)["n"] == 2


def test_taxonomy_is_the_classifiers():
    """A gold file is labeled under the same seven words the classifier used;
    a label the module accepted that the classifier could not emit would score
    a disagreement nobody could act on."""
    assert agreement.LABELS is LABELS


@pytest.mark.parametrize("label", LABELS)
def test_every_label_is_a_valid_human_label(label):
    s = agreement.score([{"row": 0, "human_label": label}], _records([label]))
    assert s["n"] == 1 and s["invalid_labels"] == {}
