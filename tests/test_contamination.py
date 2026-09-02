"""Many probes in one scan: the chunking, the attribution, and the query itself.

The query runs for real against a small Parquet file written here, the same way
`tests/test_grep.py` checks `grep.scan`, because the failure this guards against
is SQL that parses and returns a plausible wrong number — a hit on the wrong
side, a row counted twice, a second probe in the same row lost.
"""

import pytest

from trainspotting import benchmarks, contamination, grep


def _probe(i, item, part, text):
    return {"id": i, "item": item, "part": part, **benchmarks.probe(text, words=5, min_words=3)}


Q0 = "Janet sells duck eggs at the farmers market for two dollars each"
A0 = "She has sixteen minus seven equals nine eggs to sell"
Q1 = "A robe takes two bolts of blue fiber and half that white"
Q2 = "Nothing in the mix says anything like this sentence does"

PROBES = [
    _probe(0, 0, "question", Q0),
    _probe(1, 0, "answer", A0),
    _probe(2, 1, "question", Q1),
    _probe(3, 2, "question", Q2),
]


def test_chunks_respect_a_character_budget_and_keep_order():
    probes = [{"id": i, "regex": "x" * 40} for i in range(10)]
    out = contamination.chunks(probes, budget=100)
    assert [len(c) for c in out] == [2, 2, 2, 2, 2]
    assert [p["id"] for c in out for p in c] == list(range(10))
    # A probe longer than the budget still gets a chunk rather than being dropped.
    assert contamination.chunks([{"id": 0, "regex": "y" * 500}], budget=100) == [[{"id": 0, "regex": "y" * 500}]]


def _find(probes, text, **kw):
    return sorted(p["id"] for p in contamination.find(
        text, contamination.compile_alternations(probes, **kw),
        contamination.compile_probes(probes, **kw),
    ))


def test_find_hands_a_match_to_its_own_probe():
    assert _find(PROBES, "Question: " + PROBES[0]["literal"] + " ok") == [0]
    assert _find(PROBES, PROBES[0]["literal"].upper()) == [0]
    assert _find(PROBES, PROBES[0]["literal"].upper(), case_sensitive=True) == []
    assert _find(PROBES, "not a probe") == []


def test_two_items_sharing_a_window_are_both_hit():
    twins = [_probe(0, 0, "question", Q0), _probe(1, 9, "question", Q0)]
    assert _find(twins, twins[0]["literal"]) == [0, 1]


# Two near-duplicate items — the second is the first shifted by one word, as a
# templated benchmark produces — so their windows overlap in a string holding
# both. A finder that resumes after each match sees only the first.
SHIFT_A = "one two three four five six seven"
SHIFT_B = "two three four five six seven eight"
SHIFTED = [_probe(0, 0, "question", SHIFT_A), _probe(1, 1, "question", SHIFT_B)]


def test_overlapping_windows_are_both_found():
    assert SHIFTED[0]["literal"] == "two three four five six"
    assert SHIFTED[1]["literal"] == "three four five six seven"
    assert _find(SHIFTED, "one two three four five six seven eight") == [0, 1]
    # And in either order of the alternation's branches.
    assert _find(SHIFTED[::-1], "one two three four five six seven eight") == [0, 1]


@pytest.fixture
def con():
    pytest.importorskip("duckdb")
    return grep.connect()


@pytest.fixture
def dpo_parquet(con, tmp_path):
    """Four rows shaped like a Dolci DPO mix. Row 1 holds item 0's question in
    the prompt, its answer in the chosen completion, and a rewrapped copy of
    item 1's question in the rejected one. Row 2 holds item 0's question in the
    prompt only. Rows 3 and 4 hold nothing."""
    path = tmp_path / "dpo.parquet"
    q0_rewrapped = Q0.replace(" ", "\n  ")
    q1_recased = Q1.upper()
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ('Question: {Q0}',
             [{{'role': 'user', 'content': 'Question: {Q0}'}},
              {{'role': 'assistant', 'content': 'Sure. {A0}. #### 9'}}],
             [{{'role': 'user', 'content': 'Question: {Q0}'}},
              {{'role': 'assistant', 'content': 'Unrelated, but: {q1_recased}'}}],
             'gsm8k-copies'),
            ('{q0_rewrapped}',
             [{{'role': 'user', 'content': '{q0_rewrapped}'}},
              {{'role': 'assistant', 'content': 'nine'}}],
             [{{'role': 'user', 'content': '{q0_rewrapped}'}},
              {{'role': 'assistant', 'content': 'ten'}}],
             'wildchat'),
            ('Say hi',
             [{{'role': 'user', 'content': 'Say hi'}}, {{'role': 'assistant', 'content': 'hi'}}],
             [{{'role': 'user', 'content': 'Say hi'}}, {{'role': 'assistant', 'content': 'HI'}}],
             'wildchat'),
            ('Say bye',
             [{{'role': 'user', 'content': 'Say bye'}}, {{'role': 'assistant', 'content': 'bye'}}],
             [{{'role': 'user', 'content': 'Say bye'}}, {{'role': 'assistant', 'content': 'BYE'}}],
             'wildchat')
          ) AS t(prompt, chosen, rejected, dataset_source)
        ) TO '{path}' (FORMAT parquet)
        """
    )
    return path


def _run(con, path, probes, **kw):
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    source, _ = grep.source_expr(schema, ["dataset_source"])
    return contamination.scan(con, grep.read_parquet_sql([str(path)]), exprs, source, probes, **kw)


def _hits(result):
    return {(h["probe"], h["group"]): h["rows"] for h in result["probe_hits"]}


def test_scan_counts_rows_per_probe_per_side(con, dpo_parquet):
    result = _run(con, dpo_parquet, PROBES)
    assert result["matched"] == 2
    assert result["by_source"] == {"gsm8k-copies": 1, "wildchat": 1}
    assert _hits(result) == {
        (0, "prompt"): 2,   # question 0 in both rows' prompts, one of them rewrapped
        (1, "chosen"): 1,   # answer 0 in a chosen completion
        (2, "rejected"): 1, # question 1, recased, in a rejected completion
    }
    assert result["by_group"] == {"prompt": 2, "chosen": 1, "rejected": 1}


def test_items_roll_up_by_the_claim_each_side_supports(con, dpo_parquet):
    result = _run(con, dpo_parquet, PROBES)
    items = contamination.items_hit(PROBES, result["probe_hits"])
    assert items["any"] == [0, 1]
    assert items["question_read"] == [0]
    assert items["answer_produced"] == [0]
    assert items["question_produced"] == []
    # Item 1 is only in a rejected completion: seen, but neither read nor produced.
    assert 1 not in items["question_read"] and 1 not in items["answer_produced"]
    assert 2 not in items["any"]


def test_case_sensitive_loses_the_recased_copy(con, dpo_parquet):
    result = _run(con, dpo_parquet, PROBES, case_sensitive=True)
    assert (2, "rejected") not in _hits(result)
    assert _hits(result)[(0, "prompt")] == 2


def test_examples_carry_the_item_and_a_snippet_holding_the_probe(con, dpo_parquet):
    result = _run(con, dpo_parquet, PROBES, examples=5)
    assert result["examples"]
    for ex in result["examples"]:
        assert ex["item"] in (0, 1)
        assert ex["group"] in ("prompt", "chosen", "rejected")
        assert ex["source"] in ("gsm8k-copies", "wildchat")
        words = [w for w in ex["snippet"].replace("…", " ").split() if w]
        assert len(words) >= 5
    assert _run(con, dpo_parquet, PROBES, examples=0)["examples"] == []


def test_many_chunks_find_the_same_rows_as_one(con, dpo_parquet, monkeypatch):
    one = _run(con, dpo_parquet, PROBES)
    monkeypatch.setattr(contamination, "CHUNK_CHARS", 1)
    many = _run(con, dpo_parquet, PROBES)
    assert len(contamination.chunks(PROBES)) == len(PROBES)
    assert _hits(one) == _hits(many) and one["matched"] == many["matched"]


def test_scan_credits_both_of_two_overlapping_probes_in_one_row(con, tmp_path):
    path = tmp_path / "prompts.parquet"
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('Solve: one two three four five six seven eight', 'templated'),
            ('Say hi', 'wildchat')
        ) AS t(prompt, dataset_source)) TO '{path}' (FORMAT parquet)
        """
    )
    result = _run(con, path, SHIFTED)
    assert result["matched"] == 1
    assert _hits(result) == {(0, "prompt"): 1, (1, "prompt"): 1}


def test_no_hit_is_an_empty_result_not_an_error(con, dpo_parquet):
    result = _run(con, dpo_parquet, [PROBES[3]])
    assert result["matched"] == 0 and result["probe_hits"] == [] and result["examples"] == []


def test_corpus_rollup_keeps_unanswered_probes_out_of_every_denominator():
    """Item 0: question hit, answer errored — present, a hit is a hit. Item 1:
    errored, no hit — unresolved, on neither side of the rate. Item 2: counted,
    zero — settled absent."""
    counts = [
        {"probe": 0, "occurrences": 3, "approx": False},
        {"probe": 1, "occurrences": None, "approx": False, "error": "unreachable"},
        {"probe": 2, "occurrences": None, "approx": False, "error": "unreachable"},
        {"probe": 3, "occurrences": 0, "approx": False},
    ]
    out = contamination.corpus_items(PROBES, counts)
    assert out["items"]["any"] == [0]
    assert out["items_probed"] == [0, 2]
    assert out["items_unresolved"] == [1]
    assert out["errors"] == [1, 2]
    # The side keys stay empty: a corpus document is neither a prompt nor a response.
    assert out["items"]["question_read"] == [] and out["items"]["answer_produced"] == []


def test_corpus_rollup_with_every_probe_counted_is_the_plain_rollup():
    counts = [{"probe": p["id"], "occurrences": 1 if p["item"] == 1 else 0, "approx": False}
              for p in PROBES]
    out = contamination.corpus_items(PROBES, counts)
    assert out["items"]["any"] == [1]
    assert out["items_probed"] == [0, 1, 2] and out["items_unresolved"] == [] and out["errors"] == []


def test_summary_says_how_many_probes_were_not_counted():
    counts = [
        {"probe": 0, "occurrences": 3, "approx": False},
        {"probe": 1, "occurrences": None, "approx": False, "error": "unreachable"},
        {"probe": 2, "occurrences": None, "approx": False, "error": "unreachable"},
        {"probe": 3, "occurrences": 0, "approx": False},
    ]
    corpus = {"index": "v4_piletrain_llama", "counts": counts,
              **contamination.corpus_items(PROBES, counts)}
    text = "\n".join(contamination.summary([], corpus, []))
    # Over the two settled items, not three; the unanswered probes named, not summed.
    assert "corpus items seen 1/2 = 50.0%" in text
    assert "2 probe(s) could not be counted — not zeros; 1 item(s) left unresolved" in text
    assert "occurrences over all probes: 3" in text
    # And no such line when everything was counted.
    clean = {"index": "v4_piletrain_llama", "items_probed": [0], "items": {"any": []},
             "counts": [{"occurrences": 0, "approx": False}]}
    assert "could not be counted" not in "\n".join(contamination.summary([], clean, []))


def test_summary_calls_a_side_that_field_left_out_unsearched():
    def run(fields):
        return {
            "stage": "dpo", "items_probed": [0, 1], "has_answer_probes": True,
            "fields": fields, "available_fields": ["prompt", "chosen", "rejected"],
            "items": {"any": [], "question_read": [], "question_produced": [],
                      "answer_produced": [], "answer_rejected": []},
            "matched": 0, "total_rows": 10, "partial": False,
        }

    prompts_only = "\n".join(contamination.summary([run(["prompt"])], None, []))
    assert "question in a prompt: 0" in prompts_only
    assert "answer in produced text: not searched — not a zero" in prompts_only
    produced_only = "\n".join(contamination.summary([run(["chosen"])], None, []))
    assert "question in a prompt: not searched — not a zero" in produced_only
    assert "answer in produced text: 0" in produced_only
    # Reading the rejected side too changes nothing: it is not a produced side.
    everything = "\n".join(contamination.summary([run(["prompt", "chosen", "rejected"])], None, []))
    assert "not searched" not in everything


def test_corpus_side_defaults_to_a_pretraining_only_index():
    """The corpus side stands in for the crawl. An index holding Dolmino and
    Tulu 3 as well would return another model's post-training as corpus: the
    same GSM8K probes count 200/200 items in the full OLMo 2 index and 9/200 in
    the pretraining-only one."""
    from trainspotting import infinigram, registry

    index = registry.infinigram_index(registry.resolve("olmo-3-7b-instruct"))
    assert index == "v4_olmo-mix-1124_llama"
    assert "pretraining only" in infinigram.INDEXES[index]
    assert registry.infinigram_index(registry.resolve("pythia-12b-deduped")) == "v4_piletrain_llama"
    # A dataset has no pretraining behind it, so nothing to count in.
    assert registry.infinigram_index(registry.resolve("wildchat-1m")) is None


def test_corpus_only_leaves_every_stage_unscanned():
    from trainspotting import cli

    stages = [{"stage": "sft"}, {"stage": "dpo"}, {"stage": "rlvr"}]
    # A --stage selection scans one and names the rest.
    assert cli._contam_unscanned(stages, [stages[1]], corpus_only=False) == ["sft", "rlvr"]
    # No selection scans them all and names none.
    assert cli._contam_unscanned(stages, stages, corpus_only=False) == []
    # --corpus-only reads nothing, so the selection does not matter: every
    # stage is unscanned, and the summary has to say so.
    assert cli._contam_unscanned(stages, stages, corpus_only=True) == ["sft", "dpo", "rlvr"]
    assert cli._contam_unscanned(stages, [stages[1]], corpus_only=True) == ["sft", "dpo", "rlvr"]


def test_summary_names_unscanned_stages_and_the_corpus_caveat():
    run = {
        "stage": "sft", "items_probed": [0, 1, 2], "has_answer_probes": True,
        "items": {"any": [0], "question_read": [0], "question_produced": [],
                  "answer_produced": [], "answer_rejected": []},
        "matched": 4, "total_rows": 1000, "partial": False,
    }
    corpus = {
        "index": "v4_piletrain_llama", "items_probed": [0, 1, 2],
        "items": {"any": [0, 1]}, "counts": [{"occurrences": 3, "approx": False}],
        "caveat": "deduplicated elsewhere",
    }
    lines = contamination.summary([run], corpus, ["dpo", "rlvr"])
    text = "\n".join(lines)
    assert "sft    items seen 1/3 = 33.3%" in text
    assert "dpo    not scanned — not a zero" in text and "rlvr   not scanned" in text
    assert "corpus items seen 2/3" in text and "deduplicated elsewhere" in text
    assert "answer in produced text: 0" in text
