"""Exact string search over a whole mix: the schema mapping, and the SQL it builds.

Two halves. The saved Parquet schemas under fixtures/schemas/ pin the mapping
from column names to the part of the example they hold, so an upstream rename
fails here instead of quietly shrinking a count. The rest builds a small Parquet
file locally and runs the real query against it, which needs duckdb but no
network — the failure mode being guarded against is SQL that parses and returns
a plausible wrong number.
"""

import json
import re
from pathlib import Path

import pytest

from trainspotting import grep, registry

SCHEMAS = Path(__file__).resolve().parent / "fixtures" / "schemas"

STAGES = [
    (model_name, stage)
    for model_name, model in registry.MODELS.items()
    for stage in registry.post_training_stages(model)
]
STAGE_IDS = [f"{m}.{s['stage']}" for m, s in STAGES]


def schema_fixture(model: str, stage: str) -> dict:
    path = SCHEMAS / f"{model}.{stage}.json"
    assert path.exists(), (
        f"no saved schema for {model}/{stage} — run scripts/capture_parquet_schemas.py"
    )
    return json.loads(path.read_text())


# --- the schema mapping, offline -------------------------------------------


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_saved_schema_maps_to_the_same_fields(model_name, stage):
    saved = schema_fixture(model_name, stage["stage"])
    assert saved["dataset"] == stage["hf_dataset"]
    exprs, leaves, unsearched = grep.text_fields(saved["schema"])
    assert {g: len(e) for g, e in exprs.items()} == saved["groups"]
    assert [list(x) for x in leaves] == saved["leaves"]
    assert unsearched == saved["unsearched"]


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_every_text_column_is_classified(model_name, stage):
    """A text column in neither table is the finding: it would be searched by
    nobody and reported to nobody until someone reads the warning."""
    saved = schema_fixture(model_name, stage["stage"])
    _, _, unsearched = grep.text_fields(saved["schema"])
    assert unsearched == [], (
        f"{stage['hf_dataset']} has text columns this layer does not place: "
        f"{unsearched} — add them to PLAIN_TEXT/LIST_TEXT or METADATA"
    )


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_every_stage_has_a_prompt_to_search(model_name, stage):
    saved = schema_fixture(model_name, stage["stage"])
    exprs, _, _ = grep.text_fields(saved["schema"])
    assert exprs.get("prompt"), f"{stage['hf_dataset']}: nothing maps to the prompt"


@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_source_column_resolves(model_name, stage):
    saved = schema_fixture(model_name, stage["stage"])
    _, name = grep.source_expr(saved["schema"], stage["source_columns"])
    assert name == saved["source_column"]


def test_narrowing_fields_narrows_what_is_paid_for():
    """`--field prompt` should not pull the response columns; the leaves are what
    the byte cost is computed over, so this is the difference between 1 GB and 36."""
    schema = schema_fixture("olmo-3-7b-think", "rlvr")["schema"]
    all_exprs, all_leaves, _ = grep.text_fields(schema)
    exprs, leaves, _ = grep.text_fields(schema, ["prompt"])
    assert list(exprs) == ["prompt"]
    assert set(leaves) < set(all_leaves)
    assert ("outputs", None) in all_leaves and ("outputs", None) not in leaves


def test_reserved_word_columns_are_quoted():
    """Dolci RL mixes have a column called `constraint`, which DuckDB reserves."""
    exprs, _, _ = grep.text_fields({"constraint": "VARCHAR"})
    assert exprs["reference"] == ['list_value("constraint")']


def test_literal_pattern_escapes_like_wildcards():
    test = grep._element_test("100%_done", regex=False, case_sensitive=False)
    assert "ILIKE" in test
    assert "100\\%\\_done" in test


def test_case_sensitive_and_regex_pick_different_operators():
    assert "LIKE" in grep._element_test("x", regex=False, case_sensitive=True)
    assert "ILIKE" in grep._element_test("x", regex=False, case_sensitive=False)
    assert grep._element_test("x+", regex=True, case_sensitive=True) == (
        "regexp_matches(t, 'x+')"
    )
    assert ", 'i'" in grep._element_test("x+", regex=True, case_sensitive=False)


def test_quoting_survives_a_pattern_with_a_quote():
    assert "''" in grep._element_test("it's", regex=False, case_sensitive=False)


# --- snippets ---------------------------------------------------------------


def test_snippet_centres_on_the_match():
    text = "a" * 5000 + "ChatGPT" + "b" * 5000
    out = grep.snippet(text, "ChatGPT", regex=False, case_sensitive=False)
    assert "ChatGPT" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) <= grep.SNIPPET_CHARS + 2


def test_snippet_of_a_short_match_at_the_start_has_no_leading_ellipsis():
    out = grep.snippet("ChatGPT said hello", "chatgpt", regex=False, case_sensitive=False)
    assert out == "ChatGPT said hello"


def test_snippet_handles_no_match_and_no_text():
    assert grep.snippet(None, "x", False, False) is None
    assert grep.snippet("", "x", False, False) is None
    # A snippet is only ever asked for on a row that matched, but the match may be
    # in a different string of the same group than the one that came back.
    assert grep.snippet("nothing here", "x", False, False) == "nothing here"


def test_slugify():
    assert grep.slugify("I am ChatGPT") == "i-am-chatgpt"
    assert grep.slugify("as an AI language model, I") == "as-an-ai-language-model-i"
    assert grep.slugify("%%%") == "pattern"
    assert len(grep.slugify("x" * 200)) == 60


# --- the query itself, against a local Parquet file -------------------------


@pytest.fixture
def con():
    pytest.importorskip("duckdb")
    return grep.connect()


@pytest.fixture
def dpo_parquet(con, tmp_path):
    """A three-row stand-in for a Dolci DPO mix: a prompt column plus chosen and
    rejected message lists, which is the shape the role split has to handle."""
    path = tmp_path / "dpo.parquet"
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ('Interact as ChatGPT',
             [{{'role': 'user', 'content': 'Interact as ChatGPT'}},
              {{'role': 'assistant', 'content': 'As ChatGPT, sure'}}],
             [{{'role': 'user', 'content': 'Interact as ChatGPT'}},
              {{'role': 'assistant', 'content': 'no'}}],
             'wildchat'),
            ('What is 2+2',
             [{{'role': 'user', 'content': 'What is 2+2'}},
              {{'role': 'assistant', 'content': 'As ChatGPT I would say 4'}}],
             [{{'role': 'user', 'content': 'What is 2+2'}},
              {{'role': 'assistant', 'content': '5'}}],
             'math'),
            ('Say hi',
             [{{'role': 'user', 'content': 'Say hi'}},
              {{'role': 'assistant', 'content': 'hi'}}],
             [{{'role': 'user', 'content': 'Say hi'}},
              {{'role': 'assistant', 'content': 'HI'}}],
             'math')
          ) AS t(prompt, chosen, rejected, dataset_source)
        ) TO '{path}' (FORMAT parquet)
        """
    )
    return path


def _run(con, path, pattern, fields=None, **kw):
    schema = grep.schema(con, str(path))
    exprs, leaves, _ = grep.text_fields(schema, fields)
    source, _ = grep.source_expr(schema, ["dataset_source"])
    from_sql = grep.read_parquet_sql([str(path)])
    return grep.scan(con, from_sql, exprs, source, pattern, **kw), exprs, leaves


def test_scan_counts_rows_not_occurrences(con, dpo_parquet):
    """Row 1 says ChatGPT in the prompt and in both message lists. It counts once."""
    result, _, _ = _run(con, dpo_parquet, "ChatGPT")
    assert result["matched"] == 2
    assert result["by_source"] == {"wildchat": 1, "math": 1}


def test_scan_splits_prompt_from_response(con, dpo_parquet):
    """Row 1 has it on both sides, row 2 only in the assistant turns."""
    result, _, _ = _run(con, dpo_parquet, "ChatGPT")
    assert result["by_group"] == {"prompt": 1, "response": 2}
    assert result["by_source_group"]["math"] == {"prompt": 0, "response": 1}


def test_scan_narrowed_to_prompts_misses_the_response_only_row(con, dpo_parquet):
    result, exprs, _ = _run(con, dpo_parquet, "ChatGPT", fields=["prompt"])
    assert list(exprs) == ["prompt"]
    assert result["matched"] == 1
    assert result["by_source"] == {"wildchat": 1}


def test_scan_is_case_insensitive_by_default(con, dpo_parquet):
    assert _run(con, dpo_parquet, "chatgpt")[0]["matched"] == 2
    assert _run(con, dpo_parquet, "chatgpt", case_sensitive=True)[0]["matched"] == 0


def test_scan_regex(con, dpo_parquet):
    result, _, _ = _run(con, dpo_parquet, r"(As|Interact as) ChatGPT", regex=True)
    assert result["matched"] == 2
    assert _run(con, dpo_parquet, r"^hi$", regex=True)[0]["matched"] == 1


def test_scan_examples_carry_a_readable_snippet(con, dpo_parquet):
    result, _, _ = _run(con, dpo_parquet, "ChatGPT", examples=1)
    assert len(result["examples"]) == 1
    example = result["examples"][0]
    assert "ChatGPT" in example["snippet"]
    assert example["groups"] and example["source"] in ("wildchat", "math")


def test_examples_cap_does_not_change_the_count(con, dpo_parquet):
    assert _run(con, dpo_parquet, "ChatGPT", examples=0)[0]["matched"] == 2


def test_no_match_is_an_empty_result_not_an_error(con, dpo_parquet):
    result, _, _ = _run(con, dpo_parquet, "Gemini")
    assert result["matched"] == 0
    assert result["by_source"] == {} and result["examples"] == []
    assert result["by_group"] == {"prompt": 0, "response": 0}


def test_row_count_and_byte_cost_come_from_the_footers(con, dpo_parquet):
    _, _, leaves = _run(con, dpo_parquet, "ChatGPT")
    urls = [str(dpo_parquet)]
    assert grep.total_rows(con, urls) == 3
    projected = grep.byte_cost(con, urls, leaves)
    everything = grep.byte_cost(con, urls, [(c, None) for c in grep.schema(con, str(dpo_parquet))])
    assert 0 < projected <= everything


def test_byte_cost_of_a_narrowed_scan_is_smaller(con, tmp_path):
    """Only where a column belongs to one group. A message list holds both sides
    of the conversation in one column chunk, so `--field prompt` on an SFT or DPO
    mix costs exactly what searching all of it costs; an RL mix, where `outputs`
    is response-only, is where narrowing actually saves the bytes."""
    path = tmp_path / "rl.parquet"
    con.execute(
        f"""COPY (SELECT 'p' AS prompt, ['{'x' * 20000}'] AS outputs)
            TO '{path}' (FORMAT parquet)"""
    )
    urls = [str(path)]
    schema = grep.schema(con, str(path))
    _, all_leaves, _ = grep.text_fields(schema)
    _, prompt_leaves, _ = grep.text_fields(schema, ["prompt"])
    assert ("outputs", None) in all_leaves and ("outputs", None) not in prompt_leaves
    assert grep.byte_cost(con, urls, prompt_leaves) < grep.byte_cost(con, urls, all_leaves)


def test_source_totals_gives_the_denominator(con, dpo_parquet):
    totals = grep.source_totals(
        con, grep.read_parquet_sql([str(dpo_parquet)]), '"dataset_source"'
    )
    assert totals == {"wildchat": 1, "math": 2}


def test_a_mix_with_no_source_column_still_scans(con, tmp_path):
    """Dolci-Instruct-DPO carries no source column; the scan groups everything."""
    path = tmp_path / "nosource.parquet"
    con.execute(
        f"COPY (SELECT 'I am ChatGPT' AS prompt) TO '{path}' (FORMAT parquet)"
    )
    schema = grep.schema(con, str(path))
    source, name = grep.source_expr(schema, ["dataset_source"])
    assert (source, name) == (None, None)
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(path)]), exprs, source, "ChatGPT"
    )
    assert result["matched"] == 1
    assert result["by_source"] == {"(no source column)": 1}


def test_null_text_does_not_match(con, tmp_path):
    """Most RL columns are null for most mixes — `constraint` is set only for the
    instruction-following rows — and a null must not count as a match or crash."""
    path = tmp_path / "nulls.parquet"
    con.execute(
        f"""COPY (SELECT * FROM (VALUES
              ('has ChatGPT', NULL), (NULL, 'ChatGPT here'), (NULL, NULL)
            ) AS t(prompt, "constraint")) TO '{path}' (FORMAT parquet)"""
    )
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(con, grep.read_parquet_sql([str(path)]), exprs, None, "ChatGPT")
    assert result["matched"] == 2
    assert result["by_group"] == {"prompt": 1, "reference": 1}


# --- live ------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.parametrize(("model_name", "stage"), STAGES, ids=STAGE_IDS)
def test_live_schema_still_matches_the_saved_one(model_name, stage):
    """The check that catches an upstream re-release: footers only, no rows."""
    pytest.importorskip("duckdb")
    saved = schema_fixture(model_name, stage["stage"])
    con = grep.connect()
    listing = grep.parquet_listing(stage["hf_dataset"])
    live = grep.schema(con, listing["urls"][0])
    assert live == saved["schema"], (
        f"{stage['hf_dataset']} schema moved — re-run "
        "scripts/capture_parquet_schemas.py and read the diff"
    )
    assert listing["partial"] == saved["partial"]


def test_snippet_survives_a_regex_re_cannot_compile():
    """The count comes from DuckDB's RE2, which accepts syntax `re` rejects. The
    snippet loses its centring rather than taking the scan down with it."""
    out = grep.snippet("a ChatGPT b", r"(?P<x>a)(?P<x>b)", regex=True, case_sensitive=False)
    assert out == "a ChatGPT b"


# --- review round 1: role chunks, tool turns, matches past the SQL window ----


def test_role_chunks_are_part_of_the_cost():
    """A role-filtered expression reads `role` as well as `content`. Leaving the
    role leaf out understated `bytes_read` and the --max-gb guard."""
    schema = schema_fixture("olmo-3-7b-think", "dpo")["schema"]
    _, leaves, _ = grep.text_fields(schema)
    assert ("chosen", "role") in leaves
    assert ("rejected", "role") in leaves


def test_tool_turns_count_as_model_input():
    """A tool result is handed back to the model, not emitted by it, so it belongs
    on the prompt side. Only assistant output is a response."""
    typ = 'STRUCT("content" VARCHAR, "role" VARCHAR)[]'
    exprs, _, _ = grep.text_fields({"messages": typ})
    assert "'assistant', 'model'" in exprs["response"][0]
    assert exprs["prompt"][0].startswith("list_transform(list_filter(\"messages\", m -> NOT (")


@pytest.fixture
def tool_parquet(con, tmp_path):
    path = tmp_path / "tool.parquet"
    con.execute(
        f"""COPY (SELECT [
              {{'role': 'user', 'content': 'look it up'}},
              {{'role': 'assistant', 'content': 'calling the tool'}},
              {{'role': 'tool', 'content': 'the tool said ChatGPT'}}
            ] AS messages) TO '{path}' (FORMAT parquet)"""
    )
    return path


def test_tool_result_lands_in_prompt_not_response(con, tool_parquet):
    schema = grep.schema(con, str(tool_parquet))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(tool_parquet)]), exprs, None, "ChatGPT"
    )
    assert result["by_group"] == {"prompt": 1, "response": 0}


def test_only_prompt_field_still_finds_the_tool_result(con, tool_parquet):
    schema = grep.schema(con, str(tool_parquet))
    exprs, _, _ = grep.text_fields(schema, ["prompt"])
    result = grep.scan(
        con, grep.read_parquet_sql([str(tool_parquet)]), exprs, None, "ChatGPT"
    )
    assert result["matched"] == 1


def test_snippet_finds_a_match_past_the_sql_window(con, tmp_path):
    """A Dolci response can run past 100k characters. Cutting the first N before
    searching would show the opening of a response instead of the match."""
    path = tmp_path / "long.parquet"
    filler = "a" * (grep.SNIPPET_SOURCE_CHARS * 3)
    con.execute(
        f"COPY (SELECT '{filler} ChatGPT appears late {filler}' AS prompt)"
        f" TO '{path}' (FORMAT parquet)"
    )
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(path)]), exprs, None, "ChatGPT", examples=1
    )
    assert result["matched"] == 1
    snip = result["examples"][0]["snippet"]
    assert "ChatGPT appears late" in snip
    assert snip.startswith("…")


def test_snippet_finds_a_late_regex_match_too(con, tmp_path):
    path = tmp_path / "longre.parquet"
    filler = "b" * (grep.SNIPPET_SOURCE_CHARS * 2)
    con.execute(
        f"COPY (SELECT '{filler} I am ChatGPT {filler}' AS prompt)"
        f" TO '{path}' (FORMAT parquet)"
    )
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(path)]), exprs, None,
        r"(I am|I'm) ChatGPT", regex=True, examples=1,
    )
    assert "I am ChatGPT" in result["examples"][0]["snippet"]


def test_window_start_drives_the_leading_ellipsis():
    """Text the SQL window elided is as elided as text this function drops."""
    assert grep.snippet("ChatGPT here", "ChatGPT", False, False, 1) == "ChatGPT here"
    assert grep.snippet("ChatGPT here", "ChatGPT", False, False, 4001).startswith("…")


# --- review round 2: null and empty source values ---------------------------


def test_null_and_empty_source_values_share_one_denominator(con, tmp_path):
    """A column holding both NULL and '' used to produce two SQL groups whose
    labels collided, so one denominator overwrote the other while the matches
    summed under the shared key."""
    path = tmp_path / "mixedsrc.parquet"
    con.execute(
        f"""COPY (SELECT * FROM (VALUES
              ('ChatGPT a', 'wildchat'),
              ('ChatGPT b', NULL),
              ('ChatGPT c', ''),
              ('nothing',   NULL),
              ('nothing',   '')
            ) AS t(prompt, dataset_source)) TO '{path}' (FORMAT parquet)"""
    )
    urls = [str(path)]
    schema = grep.schema(con, str(path))
    source, _ = grep.source_expr(schema, ["dataset_source"])
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(con, grep.read_parquet_sql(urls), exprs, source, "ChatGPT")
    totals = grep.source_totals(con, grep.read_parquet_sql(urls), source)

    assert result["by_source"] == {grep.NO_SOURCE_VALUE: 2, "wildchat": 1}
    # Four rows carry a null-or-empty source, not the two of whichever group
    # happened to come back last.
    assert totals == {grep.NO_SOURCE_VALUE: 4, "wildchat": 1}
    # Which is what makes the printed rate meaningful: 2 of 4, not 2 of 2.
    assert result["by_source"][grep.NO_SOURCE_VALUE] < totals[grep.NO_SOURCE_VALUE]


def test_every_matched_label_has_a_denominator(con, tmp_path):
    """The invariant behind the percentages: scan and source_totals label cells
    the same way, so no key in one is missing from the other."""
    path = tmp_path / "labels.parquet"
    con.execute(
        f"""COPY (SELECT * FROM (VALUES
              ('ChatGPT', 'a'), ('ChatGPT', NULL), ('ChatGPT', '')
            ) AS t(prompt, dataset_source)) TO '{path}' (FORMAT parquet)"""
    )
    urls = [str(path)]
    schema = grep.schema(con, str(path))
    source, _ = grep.source_expr(schema, ["dataset_source"])
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(con, grep.read_parquet_sql(urls), exprs, source, "ChatGPT")
    totals = grep.source_totals(con, grep.read_parquet_sql(urls), source)
    assert set(result["by_source"]) <= set(totals)
    assert all(result["by_source"][k] <= totals[k] for k in result["by_source"])


def test_an_empty_source_value_is_not_a_missing_source_column():
    """Two different facts about a mix, so two different labels."""
    assert grep.source_label(None, has_column=False) == grep.NO_SOURCE_COLUMN
    assert grep.source_label(None, has_column=True) == grep.NO_SOURCE_VALUE
    assert grep.source_label("", has_column=True) == grep.NO_SOURCE_VALUE
    assert grep.source_label("wildchat", has_column=True) == "wildchat"


def test_regex_window_locates_the_match_not_an_earlier_lookalike(con, tmp_path):
    """`\\bChatGPT` against a string holding `xChatGPT` at the front and a real
    match past the window: searching for the extracted text finds the copy inside
    `xChatGPT` and cuts the window around a non-match."""
    path = tmp_path / "lookalike.parquet"
    filler = "a" * (grep.SNIPPET_SOURCE_CHARS * 3)
    con.execute(
        f"COPY (SELECT 'xChatGPT {filler} the real ChatGPT match {filler}' AS prompt)"
        f" TO '{path}' (FORMAT parquet)"
    )
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(path)]), exprs, None,
        r"\bChatGPT", regex=True, examples=1,
    )
    assert result["matched"] == 1
    assert "the real ChatGPT match" in result["examples"][0]["snippet"]


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        (r"(As|Interact as) ChatGPT", "zzz Interact as ChatGPT", 5),
        (r"^hi", "hi there", 1),
        (r"^hi", "x hi", 0),          # anchored and unmatched, not "position 1"
        (r"a+b", "zzz aaab", 5),
        (r"\bChatGPT", "xChatGPTonly", 0),
        ("ChatGPT", "line1\nline2 ChatGPT", 13),   # the prefix has to cross newlines
    ],
)
def test_regex_offset_sql(con, pattern, text, expected):
    """The offset expression itself, over the cases that broke the old one:
    alternation, the user's own groups, anchors, and a no-match that must not
    read as a match at the start of the string."""
    sql = grep._offset_sql("s", pattern, regex=True, case_sensitive=False)
    got = con.execute(f"SELECT {sql} FROM (SELECT ? AS s)", [text]).fetchone()[0]
    assert got == expected


# --- the committed artifacts have to substantiate their own counts -----------

RESULTS = Path(__file__).resolve().parent.parent / "results"
COMMITTED_RUNS = sorted(RESULTS.glob("*.grep-*.json"))


@pytest.mark.parametrize("path", COMMITTED_RUNS, ids=[p.name for p in COMMITTED_RUNS])
def test_committed_snippets_contain_their_pattern(path):
    """Every saved snippet has to contain the pattern the run counted.

    `scan` takes each snippet from the first *matching* string in a group, so the
    match is in there by construction — which makes this an invariant rather than
    a style preference, and the one that says the artifact is evidence. A result
    file whose examples do not show the pattern cannot substantiate its own
    number, and that is exactly what a run committed before the late-match window
    fix looked like: the snippet was the head of a long response with nothing to
    do with the query.

    Re-run the scan rather than editing a result file by hand — the point is that
    the artifact came out of the code.
    """
    run = json.loads(path.read_text())
    flags = 0 if run["case_sensitive"] else re.IGNORECASE
    expr = run["pattern"] if run["regex"] else re.escape(run["pattern"])
    missing = [
        i for i, e in enumerate(run["examples"])
        if not (e.get("snippet") and re.search(expr, e["snippet"], flags))
    ]
    assert not missing, (
        f"{path.name}: examples {missing} show no {run['pattern']!r} match — "
        f"re-run `trainspotting grep` for this stage and pattern"
    )


# --- examples have to depend on the match set, not on arrival order ----------


def test_example_selection_is_independent_of_arrival_order():
    """DuckDB reads shards in parallel, so the same match set arrives in a
    different order each run. Keeping the first N rewrote every example on a
    re-run; hashing keeps the choice a function of the matches alone."""
    records = [{"source": "s", "groups": ["prompt"], "snippet": f"match {i}"} for i in range(50)]

    def pick(order):
        kept, by_snippet = [], {}
        for i in order:
            grep._pick(kept, by_snippet, 5, records[i]["snippet"], records[i])
        return [by_snippet[t]["snippet"] for _, t in sorted(kept, key=lambda kv: -kv[0])]

    forward = pick(range(50))
    assert len(forward) == 5
    assert pick(reversed(range(50))) == forward
    assert pick([*range(25, 50), *range(25)]) == forward


def test_example_selection_spreads_rather_than_taking_a_prefix():
    """The old rule took whatever the first-read shard held — 12 consecutive
    unit-test assertions out of 134 matches. A digest-ordered sample does not
    collapse onto one run of neighbours."""
    kept, by_snippet = [], {}
    for i in range(200):
        rec = {"source": "s", "groups": ["prompt"], "snippet": f"match {i}"}
        grep._pick(kept, by_snippet, 10, rec["snippet"], rec)
    chosen = sorted(int(by_snippet[t]["snippet"].split()[1]) for _, t in kept)
    assert chosen != list(range(10))
    assert max(chosen) - min(chosen) > 20


def test_duplicate_snippets_are_kept_once():
    kept, by_snippet = [], {}
    for _ in range(10):
        grep._pick(kept, by_snippet, 5, "same text", {"source": "s", "groups": [], "snippet": "same text"})
    assert len(kept) == 1


def test_zero_examples_keeps_none(con, dpo_parquet):
    result, _, _ = _run(con, dpo_parquet, "ChatGPT", examples=0)
    assert result["matched"] == 2 and result["examples"] == []


@pytest.mark.parametrize("path", COMMITTED_RUNS, ids=[p.name for p in COMMITTED_RUNS])
def test_committed_runs_record_a_usable_pattern(path):
    """A run file has to say what was actually searched, in a form that can be
    re-run. This exists because a shell-quoting slip in a regeneration script
    turned `I'm` into `I.m` across three files: same counts, so nothing else
    noticed, but the artifact then documented a pattern nobody had chosen.
    """
    run = json.loads(path.read_text())
    assert run["pattern"] and run["pattern"].strip() == run["pattern"]
    assert run["fields"] and set(run["fields"]) <= set(grep.GROUPS)
    if run["regex"]:
        re.compile(run["pattern"])
        # `.` where a literal was meant is the shape of that slip: an unescaped
        # dot inside a word, next to characters, matching more than intended.
        assert not re.search(r"[A-Za-z]\.[A-Za-z]", run["pattern"]), (
            f"{path.name}: {run['pattern']!r} has an unescaped dot between letters"
            " — likely a quoting slip; write \\. or a character class if meant"
        )


# --- review round 4 ---------------------------------------------------------


def test_snippet_uses_the_sql_offset_when_re_cannot_compile():
    """RE2 accepts patterns `re` rejects. The old fallback searched in Python,
    failed, and cropped from offset zero — 4,000 characters short of the match
    the SQL window had been built around."""
    text = "a" * 4000 + "ChatGPT" + "b" * 4000
    bad_for_python = r"(?P<x>Chat)(?P<x>GPT)"
    blind = grep.snippet(text, bad_for_python, regex=True, case_sensitive=False)
    assert "ChatGPT" not in blind          # what the fallback can do on its own
    told = grep.snippet(
        text, bad_for_python, regex=True, case_sensitive=False,
        window_start=1, match_at=4001, match_len=7,
    )
    assert "ChatGPT" in told


def test_sql_offset_and_python_search_agree_on_ordinary_patterns():
    """Which is why switching the scan to the SQL offset moves no committed
    snippet: for a pattern both engines understand they find the same match."""
    text = "x" * 100 + "ChatGPT" + "y" * 100
    assert grep.snippet(text, "ChatGPT", False, False) == grep.snippet(
        text, "ChatGPT", False, False, 1, 101, 7
    )


def test_window_sql_reports_where_the_match_sits(con, tmp_path):
    """End to end: a match past the window cut still lands inside the snippet,
    with a pattern Python's `re` refuses to compile."""
    path = tmp_path / "re2only.parquet"
    filler = "z" * (grep.SNIPPET_SOURCE_CHARS * 2)
    con.execute(
        f"COPY (SELECT '{filler} ChatGPT late {filler}' AS prompt) TO '{path}' (FORMAT parquet)"
    )
    schema = grep.schema(con, str(path))
    exprs, _, _ = grep.text_fields(schema)
    result = grep.scan(
        con, grep.read_parquet_sql([str(path)]), exprs, None,
        r"(?P<a>Chat)(?P<a>GPT)", regex=True, examples=1,
    )
    assert result["matched"] == 1
    assert "ChatGPT late" in result["examples"][0]["snippet"]


def test_negative_example_limit_is_a_usage_error_not_an_indexerror():
    """`--examples -1` used to be accepted, run the whole scan, then die on the
    first matching row with an empty heap and no result written."""
    from trainspotting import cli
    with pytest.raises(Exception):
        cli._nonnegative_int("-1")
    assert cli._nonnegative_int("0") == 0
    assert cli._nonnegative_int("20") == 20


def test_pick_refuses_a_nonpositive_limit():
    kept, by_snippet = [], {}
    grep._pick(kept, by_snippet, -1, "text", {"snippet": "text"})
    grep._pick(kept, by_snippet, 0, "text", {"snippet": "text"})
    assert kept == [] and by_snippet == {}


def test_source_orderings_are_total_not_arrival_ordered():
    """Same names and numbers in a different order is still a diff. DuckDB
    aggregates in parallel, so nothing about arrival order is stable — ties break
    on the name to make the sort total rather than merely mostly-defined."""
    assert list(grep._by_count({"b": 1, "a": 3, "c": 1})) == ["a", "b", "c"]
    assert list(grep._by_count({"c": 1, "a": 3, "b": 1})) == ["a", "b", "c"]


@pytest.mark.parametrize("path", COMMITTED_RUNS, ids=[p.name for p in COMMITTED_RUNS])
def test_committed_runs_are_in_canonical_order(path):
    """The whole file has to be a function of the scan, not of which thread
    finished first — otherwise `git diff` after a regeneration says nothing."""
    run = json.loads(path.read_text())
    for field in ("by_source", "rows_by_source"):
        got = run.get(field) or {}
        assert list(got) == list(grep._by_count(got)), (
            f"{path.name}: {field} is not biggest-first-then-name — re-run the scan"
        )
    assert list(run["by_source_group"]) == [
        s for s in run["by_source"] if s in run["by_source_group"]
    ]


def test_locating_a_match_keeps_the_pattern_own_newline_semantics(con):
    """`.` not crossing a newline is part of what the user's pattern means. The
    locator used to pass `s` for the whole composite pattern, so `a.b` matched
    `a\\nb` while locating and not while matching: the offset landed on text the
    predicate had rejected, windowing out the hit that made the row count."""
    text = "a\nb" + "z" * 20000 + "a-b"
    sql = grep._offset_sql("s", "a.b", regex=True, case_sensitive=False)
    got = con.execute(f"SELECT {sql} FROM (SELECT ? AS s)", [text]).fetchone()[0]
    assert got == text.index("a-b") + 1
    # and the row does match, on the later occurrence
    test = grep._element_test("a.b", regex=True, case_sensitive=False)
    assert con.execute(f"SELECT {test} FROM (SELECT ? AS t)", [text]).fetchone()[0]


def test_dotall_stays_available_to_a_pattern_that_asks_for_it(con):
    """Scoping `s` to the prefix does not take it away: a user who writes `(?s)`
    gets it, in the locator and the predicate alike."""
    text = "a\nb"
    for expr, col in ((grep._offset_sql("s", "(?s)a.b", True, False), "s"),
                      (grep._element_test("(?s)a.b", True, False), "t")):
        got = con.execute(f"SELECT {expr} FROM (SELECT ? AS {col})", [text]).fetchone()[0]
        assert got in (1, True), got
