"""Turning a match rate into a share of the training budget.

The arithmetic here is the whole point of the layer: a rate over rows and a
rate over tokens are different numbers, and the second is the one that answers
"how much training did the model get". These pin the definition of a fit token
per kind, the length weighting, and what happens to a stage nobody can size.
"""

import pytest

from trainspotting import budget


def turn(role, chars, reasoning=None):
    t = {"role": role, "text": "x", "chars": chars}
    if reasoning is not None:
        t["reasoning"] = {"text": "r", "chars": reasoning}
    return t


SFT = {"kind": "sft", "turns": [turn("user", 100), turn("assistant", 200, reasoning=1000)]}
DPO = {
    "kind": "dpo",
    "chosen": {"turns": [turn("user", 100), turn("assistant", 50)]},
    "rejected": {"turns": [turn("user", 100), turn("assistant", 70)]},
}
RLVR = {"kind": "rlvr", "prompt_full": {"chars": 100}, "rollouts": {"sample": {"chars": 400}}}


def test_sft_is_fit_to_its_assistant_turns_reasoning_included():
    # The reasoning span is most of a think example's length and the model was
    # fit to all of it. Counting only the visible answer understates the stage
    # by the ratio between them.
    assert budget.fit_chars(SFT) == 1200


def test_dpo_is_fit_to_both_completions_and_not_to_the_prompt():
    # The shared history is stored on both sides, but only the completions are
    # fit to, so the prefix never enters the count in the first place.
    assert budget.fit_chars(DPO) == 120


def test_an_rl_row_without_a_stored_generation_has_no_length_rather_than_zero():
    # Zero would report the stage as weightless, which is the opposite of true:
    # the target exists, the published mix just does not ship it.
    assert budget.fit_chars(RLVR) == 400
    assert budget.fit_chars({"kind": "rlvr", "rollouts": {}}) is None


def test_a_chat_log_is_fit_to_nothing():
    assert budget.fit_chars({"kind": "chat", "turns": [turn("user", 10)]}) is None


def test_the_rate_is_weighed_by_length_not_by_row():
    # Two examples, one matching. By row that is 50%; by length the matching one
    # is a tenth of the text, so it is 9%.
    long_miss = {"kind": "sft", "turns": [turn("assistant", 1000)]}
    short_hit = {"kind": "sft", "turns": [turn("assistant", 100)]}
    total = budget.fit_chars(long_miss) + budget.fit_chars(short_hit)
    assert budget._rate(budget.fit_chars(short_hit), total) == pytest.approx(100 / 1100)


def test_the_token_interval_is_the_count_interval_rescaled_by_the_length_ratio():
    out = {
        "measured": True,
        "matched": 50,
        "n": 100,
        "count_rate": 0.5,
        "count_ci": [0.4, 0.6],
        "rate": 0.25,  # matching examples are half the average length
        "size_tokens": 1_000,
        "notes": [],
    }
    budget._apply_rate(out)
    assert out["matching_tokens"] == pytest.approx(250)
    # The ratio rate/count is 0.5, so the interval halves with the point estimate.
    assert out["matching_tokens_ci"] == pytest.approx([200, 300])


def test_a_zero_rate_keeps_the_upper_end_of_its_interval():
    # char_rate is 0 here only because nothing matched, so there is no length
    # ratio to rescale by — and dropping the upper end would report "none found"
    # as "none present", which a 300-draw sample cannot support.
    out = {
        "measured": True,
        "matched": 0,
        "n": 300,
        "count_rate": 0.0,
        "count_ci": [0.0, 0.0126],
        "rate": 0.0,
        "size_tokens": 1_000_000,
        "notes": [],
    }
    budget._apply_rate(out)
    assert out["matching_tokens"] == 0
    assert out["matching_tokens_ci"][1] == pytest.approx(12_600)


def test_weighing_by_length_on_a_handful_of_matches_says_so():
    out = {
        "measured": True,
        "matched": 3,
        "n": 300,
        "count_rate": 0.01,
        "count_ci": [0.003, 0.029],
        "rate": 0.001,
        "size_tokens": 1_000,
        "notes": [],
    }
    budget._apply_rate(out)
    assert any("3 matching example" in n for n in out["notes"])


def test_a_stage_nobody_can_size_is_left_out_of_the_total_and_named():
    stages = [
        {"stage": "pretrain", "family": "pretrain", "measured": True, "size_tokens": 1_000,
         "matching_tokens": 10, "matching_tokens_ci": [5, 20]},
        {"stage": "rlvr", "family": "post-training", "measured": True, "size_tokens": None},
    ]
    totals = budget.totals(stages)
    assert totals["all"]["size_tokens"] == 1_000
    assert totals["all"]["matching_tokens"] == 10
    assert totals["all"]["unsized"] == ["rlvr"]
    # The share is over what could be sized, and `unsized` is what says so.
    assert totals["all"]["share"] == pytest.approx(0.01)


def test_a_floor_sized_stage_makes_the_total_a_floor():
    stages = [
        {"stage": "rlvr", "family": "post-training", "measured": True, "size_tokens": 100,
         "matching_tokens": 5, "matching_tokens_ci": [1, 9], "size_is_floor": True},
    ]
    assert budget.totals(stages)["post-training"]["floor"] == ["rlvr"]


def test_a_result_record_joins_to_its_example_by_row_before_prefix():
    # Two rows opening with the same 400 characters is routine in a chat log and
    # happens in these mixes too; the row index is the key that tells them apart.
    same = "a" * 400
    by_key = {
        ("row", 7): {"kind": "sft", "row": 7},
        ("key", same): {"kind": "sft", "row": 3},
    }
    assert budget.context_for({"row": 7, "prompt": same}, by_key)["row"] == 7
    # A record from before result files carried a row still resolves.
    assert budget.context_for({"prompt": same}, by_key)["row"] == 3


def test_a_multi_turn_pair_branches_where_it_branches_not_by_role():
    # Both sides store the whole conversation, so an assistant turn before the
    # branch is shared history: counting it once per side charges the stage
    # twice for text neither completion was preferred for.
    shared_user, shared_asst = turn("user", 100), turn("assistant", 900)
    rec = {
        "kind": "dpo",
        "chosen": {"turns": [shared_user, shared_asst, turn("user", 10), turn("assistant", 50)]},
        "rejected": {"turns": [shared_user, shared_asst, turn("user", 10), turn("assistant", 70)]},
    }
    assert budget.fit_chars(rec) == 120


def test_an_identical_pair_still_has_two_completions():
    # Every turn agrees, so a plain prefix scan would call the whole thing
    # shared and leave the pair with no completions at all. The last turn is a
    # candidate answer by definition.
    same = {"turns": [turn("user", 100), turn("assistant", 50)]}
    assert budget.fit_chars({"kind": "dpo", "chosen": same, "rejected": same}) == 100


def test_a_pair_that_never_agrees_is_all_completion():
    rec = {
        "kind": "dpo",
        "chosen": {"turns": [turn("user", 100), turn("assistant", 50)]},
        "rejected": {"turns": [turn("user", 111), turn("assistant", 70)]},
    }
    # Branch at turn 0, so no shared prefix — and the user turns are still not
    # fit to, because the role filter applies past the branch as well.
    assert budget.fit_chars(rec) == 120


def test_one_slug_over_two_wordings_gets_no_total(capsys):
    """A slug is not a question.

    `--slug` takes any string and a generated one is truncated to 60
    characters, so stages sharing a slug can have been scored against different
    words. Summing them produces a number no single question ever measured, and
    the ask cards above already split such a collision apart.
    """
    from trainspotting.cli import _warn_mixed_questions

    est = {"slug": "s", "instruments": 2,
           "question_variants": ["wording one?", "wording two?"], "classifiers": []}
    assert _warn_mixed_questions(est) is True
    out = capsys.readouterr().out
    # On stdout, not stderr: the table is what gets piped into a document, and
    # the caveat has to travel with it.
    assert "no total is shown" in out.lower()
    assert "wording one?" in out and "wording two?" in out

    assert _warn_mixed_questions({"slug": "s", "instruments": 1}) is False
    assert capsys.readouterr().out == ""


def test_one_wording_judged_by_two_classifiers_also_gets_no_total(capsys):
    """The instrument is what produced the number, not just what it asked.

    The same words put to two different judges are two measurements, and the
    question text alone cannot show it — which is why the site already buckets
    ask results by question *and* classifier.
    """
    from trainspotting.cli import _warn_mixed_questions

    est = {"slug": "s", "instruments": 2, "question_variants": ["one wording?"],
           "classifiers": ["claude-opus-5", "claude-sonnet-5"]}
    assert _warn_mixed_questions(est) is True
    out = capsys.readouterr().out
    assert "2 different classifiers" in out
    assert "claude-opus-5" in out and "claude-sonnet-5" in out
    # Not described as a wording collision, because it is not one.
    assert "different wordings" not in out


def test_a_corpus_rate_is_not_weighed_by_length_a_second_time():
    """Shard selection already does the token weighting.

    `pretrain.sample_documents` draws shards with probability proportional to
    compressed size and takes one document from each, so every sampled document
    stands for the same byte mass and the document rate *is* the byte-weighted
    rate. Weighing by each document's own length on top would let a stratum of
    200k-character Longmino PDFs count 100x a stratum of 2k-character web pages
    that holds exactly as many bytes.
    """
    out = {
        "measured": True,
        "matched": 10,
        "n": 100,
        "count_rate": 0.10,
        "count_ci": [0.055, 0.175],
        # The matching documents happen to be ten times the average length.
        "rate": 0.10,
        "char_rate": 0.53,
        "size_tokens": 1_000_000,
        "notes": [],
    }
    budget._apply_rate(out)
    assert out["matching_tokens"] == pytest.approx(100_000)
    # And the interval is the plain count interval — the rescaling factor is 1.
    assert out["matching_tokens_ci"] == pytest.approx([55_000, 175_000])
    # The length note is for stages that actually reweigh; it would be a
    # non-sequitur here.
    assert out["notes"] == []


def test_a_partial_total_names_what_was_actually_measured():
    """The share's denominator is the whole pipeline, not the measured part.

    With the corpora unasked those differ by three orders of magnitude, so the
    share is a lower bound and `measured_size_tokens` is what stops it reading
    as "0.0006% of the training we looked at".
    """
    stages = [
        {"stage": "pretrain", "family": "pretrain", "measured": False, "size_tokens": 5_930_000_000_000},
        {"stage": "sft", "family": "post-training", "measured": True, "size_tokens": 1_000_000,
         "matching_tokens": 10_000, "matching_tokens_ci": [5_000, 20_000]},
    ]
    t = budget.totals(stages)["all"]
    assert t["size_tokens"] == 5_930_001_000_000
    assert t["measured_size_tokens"] == 1_000_000
    assert t["measured"] == 1 and t["stages"] == 2
    # Divided by the whole pipeline — a lower bound, not 1% of what was read.
    assert t["share"] == pytest.approx(10_000 / 5_930_001_000_000)
