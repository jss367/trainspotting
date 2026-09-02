"""Cutting a probe out of a benchmark item, and fetching the items to cut."""

import re

import pytest

from trainspotting import benchmarks, hf


@pytest.mark.parametrize("name", sorted(benchmarks.BENCHMARKS))
def test_every_benchmark_names_its_fields(name):
    spec = benchmarks.resolve(name)
    for key in ("name", "hf_dataset", "config", "split", "question", "note"):
        assert spec[key], f"{name}: missing {key}"
    assert "answer" in spec, f"{name}: say whether there is an answer field, even if None"
    assert "choices" in spec, f"{name}: say whether there is a choices field, even if None"
    assert benchmarks.parts(spec)[0] == "question"
    assert ("answer" in benchmarks.parts(spec)) == bool(spec["answer"])
    assert ("choices" in benchmarks.parts(spec)) == bool(spec["choices"])


def test_parts_are_question_then_choices_then_answer():
    assert benchmarks.parts(benchmarks.resolve("gsm8k")) == ["question", "answer"]
    assert benchmarks.parts(benchmarks.resolve("mmlu")) == ["question", "choices"]
    assert benchmarks.parts(benchmarks.resolve("arc-challenge")) == ["question", "choices"]
    assert benchmarks.parts(benchmarks.resolve("ifeval")) == ["question"]


def test_unknown_benchmark_names_the_known_ones():
    with pytest.raises(KeyError, match="gsm8k"):
        benchmarks.resolve("gsm9k")


GSM = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
    "does she make every day at the farmers' market?"
)


def test_probe_is_a_window_of_words_from_the_middle():
    p = benchmarks.probe(GSM)
    assert p["words"] == 13
    assert p["literal"] in GSM
    words = GSM.split()
    start = (len(words) - 13) // 2
    assert p["literal"].split() == words[start : start + 13]
    assert not p["literal"].startswith("Janet")


def test_probe_regex_matches_a_rewrapped_copy_and_a_recased_one():
    p = benchmarks.probe(GSM)
    rx = re.compile(p["regex"], re.IGNORECASE)
    assert rx.search(GSM)
    rewrapped = GSM.replace(" ", "\n   ", 12)
    assert rx.search(rewrapped)
    assert rx.search(GSM.upper())
    # And not a text sharing only most of the window.
    assert not rx.search(GSM.replace("farmers'", "grocers'"))


def test_probe_regex_escapes_what_needs_escaping():
    p = benchmarks.probe("a b c d e f g $2.50 (x) [y] h+i j*k")
    rx = re.compile(p["regex"])
    assert rx.search("a b c d e f g $2.50 (x) [y] h+i j*k")
    assert not rx.search("a b c d e f g $2x50 (x) [y] h+i j*k")


def test_probe_literal_keeps_the_original_whitespace():
    text = "one two\n\nthree   four five six seven eight nine ten"
    # Eight of ten words is the window that spans both oddities, and the
    # literal must be that span exactly — not a normalized copy of it.
    p = benchmarks.probe(text, words=8)
    assert p["literal"] == "two\n\nthree   four five six seven eight nine"
    # While the regex does not care how the copy was wrapped.
    assert re.compile(p["regex"]).search("two three four five six seven eight nine")


def test_short_items_are_not_probed():
    assert benchmarks.probe("What is the capital of France?") is None
    assert benchmarks.probe("a b c d e f g h", min_words=8)["words"] == 8
    assert benchmarks.probe("a b c d e f g", min_words=8) is None


def test_a_window_below_the_minimum_is_refused_not_cut():
    """`words=1` would admit every item of eight-plus words and cut one common
    word from each, which matches everywhere; the floor is on the window."""
    text = "a b c d e f g h i j k l m"
    with pytest.raises(ValueError, match="at least 8 words"):
        benchmarks.probe(text, words=1)
    with pytest.raises(ValueError):
        benchmarks.probe(text, words=7, min_words=8)
    assert benchmarks.probe(text, words=8, min_words=8)["words"] == 8


def test_item_text_applies_the_cleaning_rule():
    spec = benchmarks.resolve("gsm8k")
    row = {"question": "q", "answer": "16 - 3 - 4 = <<16-3-4=9>>9 eggs\n#### 9"}
    assert benchmarks.item_text(spec, row, "answer") == "16 - 3 - 4 = 9 eggs\n#### 9"
    assert benchmarks.item_text(spec, row, "question") == "q"
    assert benchmarks.item_text(spec, {"question": None}, "question") is None


OPTIONS = ["the mitochondrion", "the ribosome", "the nucleus", "the cell wall"]


def test_choices_come_back_joined_by_newlines_in_order_without_prefixes():
    mmlu = benchmarks.resolve("mmlu")
    row = {"question": "Which organelle makes ATP?", "choices": OPTIONS, "answer": 0}
    assert benchmarks.item_text(mmlu, row, "choices") == "\n".join(OPTIONS)
    # MMLU-Pro keeps them under `options`.
    pro = benchmarks.resolve("mmlu-pro")
    assert benchmarks.item_text(pro, {"options": OPTIONS}, "choices") == "\n".join(OPTIONS)
    # ARC nests them in a struct with the labels alongside.
    arc = benchmarks.resolve("arc-challenge")
    row = {"choices": {"text": OPTIONS, "label": ["A", "B", "C", "D"]}}
    assert benchmarks.item_text(arc, row, "choices") == "\n".join(OPTIONS)
    assert benchmarks.column(arc, "choices") == "choices"
    assert benchmarks.column(arc, "question") == "question"


def test_choices_that_are_missing_or_not_strings_are_not_text():
    mmlu = benchmarks.resolve("mmlu")
    assert benchmarks.item_text(mmlu, {"question": "q"}, "choices") is None
    assert benchmarks.item_text(mmlu, {"choices": []}, "choices") is None
    assert benchmarks.item_text(mmlu, {"choices": [1, 2]}, "choices") is None
    arc = benchmarks.resolve("arc-challenge")
    assert benchmarks.item_text(arc, {"choices": "flat"}, "choices") is None
    assert benchmarks.item_text(arc, {"choices": {"label": ["A"]}}, "choices") is None


def test_a_short_stem_still_yields_a_choices_probe():
    """The stem is under the floor and would go unprobed on its own; the
    options carry the item's text, and the probe is cut across them."""
    mmlu = benchmarks.resolve("mmlu")
    row = {
        "question": "Which of the following is true?",
        "choices": [
            "Light travels faster than sound in air",
            "Sound travels faster than light in air",
            "Both travel at the same speed in air",
            "Neither travels through air at all",
        ],
    }
    assert benchmarks.probe(benchmarks.item_text(mmlu, row, "question")) is None
    p = benchmarks.probe(benchmarks.item_text(mmlu, row, "choices"))
    assert p is not None and p["words"] == 13
    assert "\n" in p["literal"]
    rx = re.compile(p["regex"], re.IGNORECASE)
    assert rx.search(" ".join(row["choices"]))
    # The stated miss: a copy with letter prefixes between the options.
    assert rx.search("\n".join(f"{c}. {o}" for c, o in zip("ABCD", row["choices"]))) is None


def test_pick_indices_is_seeded_and_covers_a_small_set_whole():
    assert benchmarks.pick_indices(10, 200, 0) == list(range(10))
    a = benchmarks.pick_indices(1319, 200, 0)
    assert a == benchmarks.pick_indices(1319, 200, 0)
    assert a != benchmarks.pick_indices(1319, 200, 1)
    assert a == sorted(a) and len(set(a)) == 200


def test_fetch_items_reads_each_page_once_and_keeps_truncation(monkeypatch):
    calls = []

    def fake_rows_page(dataset, offset, length, config, split):
        calls.append(offset)
        return {
            "rows": [
                {"row_idx": i, "row": {"question": f"q{i}"},
                 "truncated_cells": ["question"] if i == 105 else []}
                for i in range(offset, offset + length)
            ]
        }

    monkeypatch.setattr(hf, "rows_page", fake_rows_page)
    spec = benchmarks.resolve("mmlu")
    items = benchmarks.fetch_items(spec, [3, 7, 105, 199])
    assert calls == [0, 100]
    assert [it["index"] for it in items] == [3, 7, 105, 199]
    assert items[2]["truncated"] == ["question"]
    assert items[0]["row"] == {"question": "q3"}
