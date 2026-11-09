"""Who speaks in a training example: the turn roles a mix's rows carry, and
what its system prompts say.

A system prompt is the one instruction a deployed model is meant to weigh above
the conversation, and how far it does is a property of what it was trained on.
A mix in which every system turn is the same tool-use boilerplate teaches
something different from one where the system turn sets a persona, a style or
a rule the assistant then keeps — and neither shows up in a composition table,
because no source column says which rows have a system turn at all. This layer
counts them, over every row, from the Parquet conversion `grep` already scans.

Counted, not sampled, for the reason `grep` gives: a system prompt carried by
0.1% of a mix is still tens of thousands of rows a model fit to, and a 300-row
draw expects to miss it.

What it cannot see is a turn the data never stored. Several mixes keep the
prompt as one flat string (`prompt` VARCHAR) with no roles at all, and the
released OLMo 3 chat templates add a default system turn whenever a conversation
has none — so a row with no system turn was most likely trained under that
default text, not under no system prompt. A stage with no role-bearing column
reports exactly that rather than a zero.
"""

from . import grep

# The opening of a system prompt that rows are grouped by. Grouping by the full
# text does not group anything: the Instruct SFT tool-use prompt is one fixed
# preamble followed by that row's function signatures, so 227,579 rows would be
# 227,579 groups of one. The opening is where a prompt says what it is; the
# group then reports how many distinct full texts share it.
PREVIEW_CHARS = 200

# System-prompt openings kept per stage, biggest first. The rest are counted in
# `system.groups` and summed into `system.rows`, not dropped.
TOP_PROMPTS = 20

NO_ROLE = "(no role)"


def role_columns(schema_: dict[str, str]) -> list[str]:
    """The message-list columns a row's turns are stored in, in schema order.

    The same names `grep` reads as conversations, kept to the ones whose turns
    carry both a role and text: a list of turns without a role cannot say who
    spoke, and one without text has no system prompt to read.
    """
    return [
        col for col, typ in schema_.items()
        if col in grep.MESSAGE_LISTS and typ.startswith("STRUCT")
        and grep._text_subfield(typ, "role") and grep._text_subfield(typ, "content")
    ]


def plan_leaves(columns: list[str], source_column: str | None) -> list[tuple[str, str | None]]:
    """Every column chunk the scan reads, for `grep.byte_cost`.

    The source label column is read twice, once by the scan and once by the
    denominators query, and is charged twice for the reason `grep.plan_leaves`
    gives.
    """
    leaves = [(col, sub) for col in columns for sub in ("role", "content")]
    if source_column:
        leaves += [(source_column, None)] * grep.QUERIES_READING_SOURCE
    return leaves


def _role(expr: str) -> str:
    return f"lower(trim(coalesce({expr}, '')))"


def scan(con, from_sql: str, columns: list[str], source: str | None) -> dict:
    """Rows per turn role, and the system prompts grouped by their opening.

    One pass over the remote shards into a local table holding a few hundred
    bytes a row — the roles present, the source, and the first system turn's
    opening, length and hash — and every count after it reads that table. The
    message text is the expensive part and it is read once.

    A DPO pair stores the shared conversation in both `chosen` and `rejected`,
    so a row is counted once per role whichever column the turn is in, and its
    system prompt is the first one found in column order. A row whose two sides
    disagree on the system prompt would be counted under `chosen`'s; the
    `distinct_texts` figure is over the same choice.
    """
    turns = grep._flatten([f"coalesce({grep._ident(c)}, [])" for c in columns])
    system_text = "coalesce(" + ", ".join(
        f"list_filter({grep._ident(c)}, m -> {_role('m.role')} = 'system')[1].content"
        for c in columns
    ) + ")"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE roles_scan AS
        SELECT {source or 'NULL'} AS src,
               list_distinct(list_transform({turns}, m -> {_role('m.role')})) AS roles,
               left(s, {PREVIEW_CHARS}) AS preview,
               length(s) AS chars,
               hash(s) AS digest
        FROM (SELECT *, {system_text} AS s FROM {from_sql})
    """)

    total, with_turns = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE len(roles) > 0) FROM roles_scan"
    ).fetchone()
    rows_by_role = {
        role or NO_ROLE: n
        for role, n in con.execute(
            "SELECT r, count(*) FROM (SELECT unnest(roles) AS r FROM roles_scan) GROUP BY 1"
        ).fetchall()
    }

    has_system = "list_contains(roles, 'system')"
    system_rows = con.execute(
        f"SELECT count(*) FROM roles_scan WHERE {has_system}"
    ).fetchone()[0]
    by_source: dict[str, int] = {}
    for value, n in con.execute(
        f"SELECT src, count(*) FROM roles_scan WHERE {has_system} GROUP BY 1"
    ).fetchall():
        key = grep.source_label(value, has_column=source is not None)
        by_source[key] = by_source.get(key, 0) + n

    groups = con.execute(f"""
        SELECT preview, count(*) AS n, count(DISTINCT digest) AS texts,
               min(chars), max(chars)
        FROM roles_scan WHERE {has_system}
        GROUP BY preview
    """).fetchall()
    # Biggest first, then by text, so a re-run writes the same file: DuckDB
    # returns groups in no fixed order. `grep._by_count` does the same for the
    # source tables.
    groups.sort(key=lambda g: (-g[1], g[0] or ""))
    prompts = []
    for preview, n, texts, lo, hi in groups[:TOP_PROMPTS]:
        srcs = con.execute(
            f"SELECT src, count(*) FROM roles_scan WHERE {has_system} "
            "AND preview IS NOT DISTINCT FROM ? GROUP BY 1",
            [preview],
        ).fetchall()
        sources: dict[str, int] = {}
        for value, k in srcs:
            key = grep.source_label(value, has_column=source is not None)
            sources[key] = sources.get(key, 0) + k
        prompts.append({
            # A system turn with no text still counts as a system turn; its
            # opening is the empty string rather than null so the page can
            # print it.
            "preview": preview or "",
            "truncated": (hi or 0) > PREVIEW_CHARS,
            "rows": n,
            "distinct_texts": texts,
            "chars": {"min": lo, "max": hi},
            "sources": grep._by_count(sources),
        })

    distinct = con.execute(
        f"SELECT count(DISTINCT digest) FROM roles_scan WHERE {has_system}"
    ).fetchone()[0]
    con.execute("DROP TABLE roles_scan")
    return {
        "total_rows": total,
        "rows_with_turns": with_turns,
        "rows_by_role": grep._by_count(rows_by_role),
        "system": {
            "rows": system_rows,
            "by_source": grep._by_count(by_source),
            "distinct_texts": distinct,
            "groups": len(groups),
            "prompts": prompts,
        },
    }
