"""The `roles` layer: which turn roles a mix's rows carry, and its system prompts.

The column choice is checked against the saved Parquet schemas `test_grep`
pins, so an upstream re-release that drops a mix's roles fails here rather than
reporting "not recorded" for a stage that used to be counted. The scan itself
runs against small local Parquet files shaped like the Dolci mixes.
"""

import json
from pathlib import Path

import pytest

from trainspotting import grep, roles

SCHEMAS = Path(__file__).resolve().parent / "fixtures" / "schemas"


def _schema(name):
    return json.loads((SCHEMAS / f"{name}.json").read_text())["schema"]


# --- which columns hold turns ------------------------------------------------


def test_sft_turns_are_in_messages():
    assert roles.role_columns(_schema("olmo-3-7b-instruct.sft")) == ["messages"]
    assert roles.role_columns(_schema("olmo-3-7b-think.sft")) == ["messages"]


def test_a_pair_reads_both_sides():
    assert roles.role_columns(_schema("olmo-3-7b-think.dpo")) == ["chosen", "rejected"]
    assert roles.role_columns(_schema("olmo-3-7b-instruct.dpo")) == ["chosen", "rejected"]


def test_an_rl_conversation_is_read_from_source_prompt():
    assert roles.role_columns(_schema("olmo-3-7b-instruct.rlvr")) == ["source_prompt"]


def test_a_flat_prompt_string_has_no_roles_to_count():
    # Think RL stores the prompt as one VARCHAR. Reporting zero system turns for
    # it would be a claim the data cannot support either way.
    assert roles.role_columns(_schema("olmo-3-7b-think.rlvr")) == []


def test_plan_charges_role_text_and_both_source_reads():
    leaves = roles.plan_leaves(["messages"], "source_dataset")
    assert leaves == [
        ("messages", "role"), ("messages", "content"),
        ("source_dataset", None), ("source_dataset", None),
    ]
    assert roles.plan_leaves(["messages"], None) == [("messages", "role"), ("messages", "content")]


# --- the scan, against local Parquet -----------------------------------------


@pytest.fixture
def con():
    pytest.importorskip("duckdb")
    return grep.connect()


TOOLS = "You are a helpful function-calling AI assistant. " + "x" * 300


@pytest.fixture
def sft_parquet(con, tmp_path):
    """Six rows: two tool-use system prompts that share an opening and differ
    after it, one identity prompt with its role in upper case, and three rows
    with no system turn — one of them with no turns at all."""
    path = tmp_path / "sft.parquet"
    con.execute(
        f"""
        COPY (
          SELECT * FROM (VALUES
            ([{{'role': 'system', 'content': '{TOOLS}<functions>a</functions>'}},
              {{'role': 'user', 'content': 'call a'}},
              {{'role': 'assistant', 'content': 'ok'}}], 'tools'),
            ([{{'role': 'system', 'content': '{TOOLS}<functions>b</functions>'}},
              {{'role': 'user', 'content': 'call b'}},
              {{'role': 'assistant', 'content': 'ok'}},
              {{'role': 'tool', 'content': '42'}}], 'tools'),
            ([{{'role': 'System', 'content': 'You are Olmo.'}},
              {{'role': 'user', 'content': 'who are you'}},
              {{'role': 'assistant', 'content': 'Olmo'}}], 'hardcoded'),
            ([{{'role': 'user', 'content': 'hi'}},
              {{'role': 'assistant', 'content': 'hello'}}], 'chat'),
            ([{{'role': 'user', 'content': 'and you'}},
              {{'role': 'assistant', 'content': 'me'}}], 'chat'),
            ([]::STRUCT(role VARCHAR, content VARCHAR)[], 'chat')
          ) AS t(messages, source_dataset)
        ) TO '{path}' (FORMAT parquet)
        """
    )
    return path


def _scan(con, path, columns, source_col="source_dataset"):
    schema = grep.schema(con, str(path))
    source, _ = grep.source_expr(schema, [source_col])
    return roles.scan(con, grep.read_parquet_sql([str(path)]), columns, source)


def test_rows_are_counted_per_role(con, sft_parquet):
    r = _scan(con, sft_parquet, ["messages"])
    assert r["total_rows"] == 6
    assert r["rows_with_turns"] == 5
    # Lowercased, so `System` and `system` are one role.
    assert r["rows_by_role"] == {"assistant": 5, "user": 5, "system": 3, "tool": 1}


def test_system_prompts_group_by_their_opening(con, sft_parquet):
    system = _scan(con, sft_parquet, ["messages"])["system"]
    assert system["rows"] == 3
    assert system["by_source"] == {"tools": 2, "hardcoded": 1}
    assert system["distinct_texts"] == 3
    assert system["groups"] == 2
    top, rest = system["prompts"]
    # The two tool-use rows differ only past the opening, so they are one group
    # of two distinct texts — the shape of the real Instruct SFT tool prompt.
    assert top["rows"] == 2 and top["distinct_texts"] == 2
    assert top["preview"] == TOOLS[:roles.PREVIEW_CHARS]
    assert top["truncated"] is True
    assert top["sources"] == {"tools": 2}
    assert rest["preview"] == "You are Olmo."
    assert rest["truncated"] is False
    assert rest["chars"] == {"min": 13, "max": 13}


def test_a_mix_with_no_system_turn_reports_zero(con, tmp_path):
    path = tmp_path / "think.parquet"
    con.execute(
        f"""COPY (SELECT [{{'role': 'user', 'content': 'q'}},
                          {{'role': 'assistant', 'content': 'a'}}] AS messages,
                         'x' AS dataset_source)
            TO '{path}' (FORMAT parquet)"""
    )
    system = _scan(con, path, ["messages"])["system"]
    assert system == {"rows": 0, "by_source": {}, "distinct_texts": 0, "groups": 0, "prompts": []}


def test_a_pair_counts_its_shared_system_turn_once(con, tmp_path):
    """A DPO pair repeats the conversation in both sides. One row, one system
    prompt — not two."""
    path = tmp_path / "dpo.parquet"
    turns = "{{'role': 'system', 'content': 'Be terse.'}}, {{'role': 'user', 'content': 'q'}}"
    con.execute(
        f"""COPY (SELECT [{turns.format()}, {{'role': 'assistant', 'content': 'a'}}] AS chosen,
                         [{turns.format()}, {{'role': 'assistant', 'content': 'b'}}] AS rejected)
            TO '{path}' (FORMAT parquet)"""
    )
    r = _scan(con, path, ["chosen", "rejected"], source_col="none")
    assert r["rows_by_role"] == {"assistant": 1, "system": 1, "user": 1}
    assert r["system"]["rows"] == 1
    assert r["system"]["by_source"] == {grep.NO_SOURCE_COLUMN: 1}
    assert r["system"]["prompts"][0]["preview"] == "Be terse."
