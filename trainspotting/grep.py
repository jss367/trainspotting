"""Exact string search over a whole post-training mix, without downloading it.

The sampling layers answer the unconditional question — "what is in here", with
a rate and a confidence interval. This one answers the conditional one: "how
many rows contain this string", exactly, over every row of the mix. That is the
question you have once you already know the string, and a 300-row sample cannot
answer it: a pattern present in 0.1% of a mix is expected to miss a sample of
that size entirely.

The route is the datasets-server's own Parquet conversion of each repo (the
`refs/convert/parquet` branch), scanned in place by DuckDB over HTTP. Parquet is
columnar, so a query that projects three text columns pays for those columns and
skips the rest — searching prompts in a mix whose bulk is tokenised `input_ids`
costs a small fraction of the repo. Nothing is written to disk.

The two cheaper routes do not work on these repos. `/search` (full-text) and
`/filter` (SQL `LIKE`) both need a server-side index that is not built for any
Dolci mix: every one of them answers "the dataset index is loading" or 502s,
indefinitely. And `/statistics`, which the `sources` layer uses, counts whole
values of a low-cardinality column — it cannot look inside a prompt.

Rows are matched, not occurrences: a row whose response says "ChatGPT" four
times counts once. That is the unit the training run sees.
"""

import hashlib
import heapq
from collections import Counter
import os
import re
import sys
import urllib.parse

import requests

BASE = "https://datasets-server.huggingface.co"
HUB = "https://huggingface.co"
PARQUET_BRANCH = "refs/convert/parquet"

# Which part of the training example a column belongs to. The first three are
# the cut the `context` layer draws: what the model is asked, what it is fit to
# (or pushed between), and what scores it.
#
# `chosen` and `rejected` are separate for the same reason, and it is the
# sharpest case: the same string in each teaches opposite things. "I am ChatGPT"
# in a chosen completion trains the model toward saying it; in the rejected one
# it trains the model away. Adding the two into one `response` count is worse
# than not counting — it let a phrase that only ever appears in rejected text
# rank DPO as where the model learned to say it.
#
# `rollout` exists to stay *out* of the produce side too. An RL row's
# `outputs` are reference-model generations kept to compute a passrate for
# difficulty filtering — the context view says so in as many words, and an RL row
# stores no response at all. Counting them as response made a hit in text the
# objective never pushes the model to emit into produce-side evidence, which is
# what `influence` ranks a stage's origin on: a phrase appearing only in
# rollouts could rank RLVR as where the model learned to say it. They stay
# searchable, in their own group, outside `influence.PRODUCE`.
GROUPS = ("prompt", "response", "chosen", "rejected", "reference", "rollout")

# Turn roles whose content the model is fit to *produce*. Everything else in a
# message list is text it conditions on, which includes the tool turns: a tool
# or function result is handed back to the model, not emitted by it, so counting
# it as a response would credit the model with text it only read. An unset or
# unrecognised role lands on the prompt side for the same reason — under-counting
# responses is the honest direction when the question is what a model was
# trained to say.
RESPONSE_ROLES = ("assistant", "model")

# Columns holding training text, by the group they belong to. A message-list
# column is split by turn role and so lands in two groups at once; it is listed
# here under the subfields to read, and `_message_exprs` does the split.
PLAIN_TEXT = {
    "prompt": "prompt",
    "solution": "reference",  # RL: a reference implementation, not a target
    "constraint": "reference",  # RL: the constraint list a checker verifies
}
LIST_TEXT = {
    "ground_truth": "reference",
    "outputs": "rollout",  # reference generations, not anything the model is fit to
}
STRUCT_TEXT = {
    ("reward_model", "ground_truth"): "reference",
}
MESSAGE_LISTS = ("messages", "chosen", "rejected", "source_prompt", "conversation")
# Extra subfields some message lists carry, and which side of the turn they are
# on: a tool schema is given to the model, a call is emitted by it.
MESSAGE_EXTRAS = {"functions": "prompt", "function_calls": "response"}

# Text columns that are provenance, labels, or identifiers rather than training
# text. Named so that a column in neither table gets reported as unsearched
# instead of being silently skipped.
METADATA = {
    "id", "custom_id", "prompt_id", "conversation_hash", "key", "dataset",
    "dataset_source", "source_dataset", "source", "original_dataset",
    "data_source", "model", "chosen_model", "rejected_model", "predicted_label",
    "preference_type", "setting_key", "setting_name", "domain", "ability",
    "topic", "characters", "constraint_type", "difficulty_explanation",
    "extra_info", "difficulty",
    # WildChat-1M, the first standalone dataset target: a chat log carries the
    # request's own metadata beside the turns — where it came from, what
    # language it was in, and what two moderation models made of it. None of it
    # is text the model was fit to.
    "language", "hashed_ip", "header", "openai_moderation",
    "detoxify_moderation", "turn", "timestamp", "toxic", "redacted",
    "state", "country",
}


def _ident(name: str) -> str:
    """Quote a column name. Dolci uses `constraint`, which is a reserved word."""
    return '"' + name.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def parquet_listing(dataset: str, config: str = "default", split: str = "train") -> dict:
    """Every Parquet shard of one mix, pinned to a revision.

    The conversion lives on a branch that moves, so the file list is resolved to
    the commit it was read at and the URLs address that SHA — a result file then
    names a specific tree rather than a moving `main`, the same way the
    pretraining sampler pins its shard listing.

    A repo the server only converted part of comes back under a `partial-`
    prefixed split, which makes any count a lower bound; that travels in the
    return value rather than being silently dropped.
    """
    headers = {"Authorization": f"Bearer {hf_token()}"} if hf_token() else {}
    r = requests.get(
        f"{BASE}/parquet", params={"dataset": dataset}, headers=headers, timeout=60
    )
    r.raise_for_status()
    files = [
        f for f in r.json()["parquet_files"]
        if f["config"] == config and f["split"] in (split, f"partial-{split}")
    ]
    if not files:
        raise RuntimeError(f"no {config}/{split} parquet files for {dataset}")

    branch = urllib.parse.quote(PARQUET_BRANCH, safe="")
    rev = requests.get(
        f"{HUB}/api/datasets/{dataset}/revision/{branch}", headers=headers, timeout=60
    )
    rev.raise_for_status()
    sha = rev.json()["sha"]

    urls = [
        f["url"].replace(urllib.parse.quote(PARQUET_BRANCH, safe=""), sha)
        .replace(PARQUET_BRANCH, sha)
        for f in files
    ]
    return {
        "urls": urls,
        "revision": sha,
        "partial": any(f["split"].startswith("partial-") for f in files),
        "compressed_bytes": sum(f["size"] for f in files),
    }


def connect():
    """A DuckDB connection that can read from the hub over HTTP.

    A scan is thousands of range requests against one host, so the hub's rate
    limiter is part of the job rather than an anomaly: DuckDB's three default
    retries run out mid-scan and surface as a bare HTTP 429 after several minutes
    of reading. Retry further and back off. A token, if there is one in the
    environment, raises the limit that is being hit in the first place.
    """
    try:
        import duckdb
    except ModuleNotFoundError:
        sys.exit(
            "trainspotting grep needs duckdb: pip install 'duckdb>=1.0'\n"
            "(it is the only layer that needs it, so it is an optional extra: "
            "pip install -e '.[grep]')"
        )
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET http_retries=10; SET http_retry_wait_ms=1000; SET http_retry_backoff=2;")
    token = hf_token()
    if token:
        con.execute(
            "CREATE OR REPLACE SECRET hf_hub (TYPE HTTP, EXTRA_HTTP_HEADERS "
            f"MAP{{'Authorization': 'Bearer {token}'}})"
        )
    return con


def schema(con, url: str) -> dict[str, str]:
    """Column name -> DuckDB type, read from one shard's footer."""
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_lit(url)})").fetchall()
    return {name: str(typ) for name, typ, *_ in rows}


PAIR = ("chosen", "rejected")


def _pair_exprs(typ: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """Expressions for a DPO pair, split where the two completions branch.

    A multi-turn pair shares a conversation prefix — assistant turns included —
    and differs only from the point one answer diverges from the other. Those
    shared turns are the conversation the pair is judged *in*, not either
    candidate answer. Attributing them by role put a string from the shared
    history into both `chosen` and `rejected`, so a phrase neither completion
    contains counted twice as text the objective pushes the model toward.

    `search.py` has drawn the line here from the start (`_shared_turns`); this
    is the same rule expressed over columns instead of parsed records, which is
    the third finding to come out of those two maps being written twice.
    """
    c, r = _ident("chosen"), _ident("rejected")
    roles = ", ".join(_lit(x) for x in RESPONSE_ROLES)
    emitted = f"lower(coalesce(m.role, '')) IN ({roles})"
    both = ("content", "role")
    # 1-based index of the first turn the two completions disagree on, or one
    # past the shorter list when one is a prefix of the other.
    same = f"list_transform(list_zip({c}, {r}), z -> (z[1] IS NOT DISTINCT FROM z[2]))"
    b = f"coalesce(list_position({same}, false), least(len({c}), len({r})) + 1)"

    def tail(col):
        return f"list_slice({col}, {b}, len({col}))"

    def content(expr, keep):
        return f"list_transform(list_filter({expr}, m -> {keep}), m -> m.content)"

    out = [
        # The shared prefix in full, whatever role each turn carries.
        ("prompt", f"list_transform(list_slice({c}, 1, {b} - 1), m -> m.content)", both),
        ("chosen", content(tail(c), emitted), both),
        ("rejected", content(tail(r), emitted), both),
    ]
    # A user turn after the branch is still something the model reads, not
    # something either completion claims.
    for col in (c, r):
        out.append(("prompt", content(tail(col), f"NOT ({emitted})"), both))
    for sub, group in MESSAGE_EXTRAS.items():
        if f'"{sub}"' in typ or f" {sub} " in typ:
            for col in (c, r):
                out.append((group, f"list_transform({col}, m -> m.{_ident(sub)})", (sub,)))
    return out


def _message_exprs(col: str, typ: str) -> list[tuple[str, str, tuple[str, ...]]]:
    """(group, expression, subfields read) for each searchable part of a message list.

    The expression evaluates to VARCHAR[] so every group matches the same way.
    Roles are lowercased and null-guarded because `source_prompt` in the RL
    mixes carries rows with neither.

    A role-filtered expression reads `role` as well as `content`, and both
    subfields travel back so the byte cost covers the whole transfer. Role
    chunks are small next to message text, but a cost the scan does not actually
    pay for is the wrong number to put in front of a `--max-gb` decision.
    """
    q = _ident(col)
    roles = ", ".join(_lit(r) for r in RESPONSE_ROLES)
    out = []
    if "content" in typ:
        emitted = f"lower(coalesce(m.role, '')) IN ({roles})"
        both = ("content", "role")
        # A DPO pair's two completions are two different claims about the same
        # prompt, so the produced side takes the column's own name. Everything
        # else produced is `response`. The input side is shared — a pair's user
        # turns are the same turns — so it stays `prompt` either way.
        produced = col if col in ("chosen", "rejected") else "response"
        out.append((produced, f"list_transform(list_filter({q}, m -> {emitted}), m -> m.content)", both))
        out.append(("prompt", f"list_transform(list_filter({q}, m -> NOT ({emitted})), m -> m.content)", both))
    for sub, group in MESSAGE_EXTRAS.items():
        if f'"{sub}"' in typ or f" {sub} " in typ:
            out.append((group, f"list_transform({q}, m -> m.{_ident(sub)})", (sub,)))
    return out


def text_fields(
    schema_: dict[str, str], fields: list[str] | None = None
) -> tuple[dict[str, list[str]], list[tuple[str, str | None]], list[str]]:
    """Split a mix's schema into what to search, what to pay for, and what is left.

    Returns the VARCHAR[]-valued expressions per group, the (column, subfield)
    leaves those expressions read (which is what the byte cost is computed over),
    and any text column recognised as neither training text nor metadata — a new
    column in an upstream re-release shows up there rather than being dropped.

    `fields` narrows to a subset of the groups, and narrows the leaves with them,
    so asking for prompts alone does not pay for the responses.
    """
    want = set(fields or GROUPS)
    exprs: dict[str, list[str]] = {g: [] for g in GROUPS}
    leaves: list[tuple[str, str | None]] = []
    unsearched: list[str] = []

    def add(group, expr, col, subs):
        if group not in want:
            return
        exprs[group].append(expr)
        for sub in subs if isinstance(subs, tuple) else (subs,):
            if (col, sub) not in leaves:
                leaves.append((col, sub))

    pair = all(
        c in schema_ and schema_[c].startswith("STRUCT") for c in PAIR
    )
    for col, typ in schema_.items():
        q = _ident(col)
        if pair and col in PAIR:
            # The two are one mapping, emitted once when `chosen` comes round.
            # Both columns are read by it, so both contribute leaves.
            if col == "chosen":
                for group, expr, subs in _pair_exprs(typ):
                    add(group, expr, "chosen", subs)
            if any(g in want for g in ("prompt", *PAIR)):
                for sub in ("content", "role"):
                    if (col, sub) not in leaves:
                        leaves.append((col, sub))
        elif col in MESSAGE_LISTS and typ.startswith("STRUCT"):
            for group, expr, subs in _message_exprs(col, typ):
                add(group, expr, col, subs)
        elif col in PLAIN_TEXT and typ == "VARCHAR":
            add(PLAIN_TEXT[col], f"list_value({q})", col, (None,))
        elif col in LIST_TEXT and typ == "VARCHAR[]":
            add(LIST_TEXT[col], q, col, (None,))
        elif any(c == col for c, _ in STRUCT_TEXT) and typ.startswith("STRUCT"):
            for (c, sub), group in STRUCT_TEXT.items():
                if c == col and sub in typ:
                    add(group, f"list_value({q}.{_ident(sub)})", col, (sub,))
        elif "VARCHAR" in typ and col not in METADATA:
            unsearched.append(col)

    return {g: e for g, e in exprs.items() if e}, leaves, unsearched


# The scan and the denominators query. Both read the source label column once.
QUERIES_READING_SOURCE = 2


def plan_leaves(
    schema_: dict[str, str], fields: list[str] | None, source_column: str | None
) -> list[tuple[str, str | None]]:
    """Every column read for one scan, with the multiplicity it is read at.

    Multiplicity counts *queries that touch a column*, not roles it plays: the
    scan reads a column once whether it is filtered, projected or both, and the
    denominators query reads the source label column once more. So the source
    leaf is clamped to two rather than appended to whatever `text_fields`
    already returned — `--by prompt` on an RL stage would otherwise be charged
    three copies of a column read twice.

    This lives here because it was computed in two places and they diverged: the
    CLI learned to clamp and `scripts/recompute_grep_bytes.py` kept appending,
    so the maintenance script would have rewritten `bytes_read` with the
    inflated figure the CLI had just stopped producing.
    """
    _, leaves, _ = text_fields(schema_, fields)
    if source_column:
        already = sum(1 for leaf in leaves if leaf == (source_column, None))
        leaves = [*leaves, *[(source_column, None)] * max(0, QUERIES_READING_SOURCE - already)]
    return leaves


def source_expr(schema_: dict[str, str], columns: list[str]) -> tuple[str | None, str | None]:
    """The column to break the counts down by, as SQL, and its name.

    Takes the registry's own `source_columns` so a `scan` breakdown lines up with
    what `sources` reports. `dataset` is a one-element list in the RL mixes and a
    plain string in Think-DPO-32B, so the type decides how it is read. Some mixes
    (Dolci-Instruct-DPO) carry no source column at all.
    """
    for col in columns:
        typ = schema_.get(col)
        if typ == "VARCHAR":
            return _ident(col), col
        if typ == "VARCHAR[]":
            return f"{_ident(col)}[1]", col
    return None, None


def byte_cost(con, urls: list[str], leaves: list[tuple[str, str | None]]) -> int:
    """Compressed bytes the scan will pull, from the shard footers.

    Exact rather than estimated: Parquet records the compressed size of every
    column chunk, and reading footers costs a range request per shard. Worth
    knowing before the fact — the Think SFT mixes are tens of gigabytes of
    message text, while the DPO and RL mixes are one or two.
    """
    if not leaves:
        return 0
    rows = con.execute(
        f"SELECT path_in_schema, total_compressed_size FROM parquet_metadata({urls!r})"
    ).fetchall()
    # A leaf listed twice is read twice, and costs twice — the source label
    # column is, once by the scan and once by the denominators query. The break
    # stays so that a chunk matched by two different leaf specs is not double
    # counted; multiplicity comes from the count, not from the iteration.
    wanted = Counter(leaves)
    total = 0
    for path, size in rows:
        parts = [p.strip() for p in path.split(",")]
        for (col, sub), times in wanted.items():
            if parts[0] == col and (sub is None or parts[-1] == sub):
                total += size * times
                break
    return total


def total_rows(con, urls: list[str]) -> int:
    """Row count from the footers, so the denominator costs no data reads."""
    return con.execute(
        f"SELECT sum(num_rows) FROM parquet_file_metadata({urls!r})"
    ).fetchone()[0]


def _by_count(counts: dict[str, int]) -> dict[str, int]:
    """Biggest first, then by name — the order a reader wants, and a fixed one.

    DuckDB aggregates in parallel and returns groups in no particular order, and
    arrival order is not an order either. Left alone, every re-run reshuffled the
    keys of `rows_by_source` and `by_source_group`: same names, same numbers,
    a diff on every regeneration. Ties break on the name so the sort is total.
    """
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


NO_SOURCE_COLUMN = "(no source column)"
NO_SOURCE_VALUE = "(no source value)"


def source_label(value: str | None, has_column: bool) -> str:
    """The one label for a source cell, so a numerator and its denominator agree.

    Three states collapse to two labels and both distinctions matter. A mix with
    no source column at all (Dolci-Instruct-DPO) is a different thing from a row
    whose source column is null or empty, and null and the empty string are the
    same thing for this purpose.

    Normalising in one place is the point. The counts and the denominators used to
    each do their own, so a column holding both NULL and `''` produced two SQL
    groups whose labels collided: the matches summed under one key while the
    denominator kept only whichever group came back last, and the printed
    percentage and stored `rows_by_source` were both wrong for those rows.
    """
    if not has_column:
        return NO_SOURCE_COLUMN
    return value if value else NO_SOURCE_VALUE


def source_totals(con, from_sql: str, source: str) -> dict[str, int]:
    """Rows per source over the whole mix, which is the denominator that means
    something: "434 of the 17,596 WildChat rows" says what "0.3% of the mix" does
    not. A source label is a few bytes a row and compresses to almost nothing, so
    this is a rounding error next to the text scan.

    Only called when the mix has a source column, so every row is labelled as
    one, and groups that normalise together are added rather than overwritten.
    """
    rows = con.execute(
        f"SELECT {source} AS src, count(*) FROM {from_sql} GROUP BY 1"
    ).fetchall()
    out: dict[str, int] = {}
    for value, n in rows:
        key = source_label(value, has_column=True)
        out[key] = out.get(key, 0) + n
    return _by_count(out)


def _element_test(pattern: str, regex: bool, case_sensitive: bool) -> str:
    """SQL predicate on the string bound to `t`."""
    if regex:
        opts = "" if case_sensitive else ", 'i'"
        return f"regexp_matches(t, {_lit(pattern)}{opts})"
    op = "LIKE" if case_sensitive else "ILIKE"
    # A literal search should treat % and _ as themselves.
    esc = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"t {op} {_lit('%' + esc + '%')} ESCAPE '\\'"


def _flatten(exprs: list[str]) -> str:
    """One VARCHAR[] out of several. `flatten` takes any arity; `list_concat` does not."""
    if len(exprs) == 1:
        return exprs[0]
    return f"flatten(list_value({', '.join(exprs)}))"


def _matching(exprs: list[str], test: str) -> str:
    return f"list_filter({_flatten(exprs)}, t -> t IS NOT NULL AND {test})"


def _match_sql(exprs: list[str], test: str) -> str:
    return f"len({_matching(exprs, test)}) > 0"


# The first matching string is windowed to this in SQL and to SNIPPET_CHARS here,
# for the same reason the context layer cuts its fields: a Dolci response can run
# past 100k characters and no snippet needs that. The SQL window is centred on
# the match rather than taken from the head, because a match past the cut would
# otherwise be truncated away and the "snippet" would be the opening of a
# response that has nothing to do with the pattern.
SNIPPET_SOURCE_CHARS = 8000
SNIPPET_CHARS = 240


def _offset_sql(col: str, pattern: str, regex: bool, case_sensitive: bool) -> str:
    """1-based position of the first match inside `col`, or 0 if there is none.

    A regex has no `position` of its own. Searching for the text `regexp_extract`
    returned finds the wrong place whenever an earlier identical substring fails
    the pattern: `\bChatGPT` against a string holding `xChatGPT` at the front and
    a real match 20,000 characters later extracts "ChatGPT" and then locates the
    copy inside `xChatGPT`, so the window gets cut around a non-match.

    So measure the match's own offset instead, by capturing everything before it
    with a lazy prefix group. The user's pattern is parenthesised so an
    alternation binds as a unit and its own capture groups shift out of the way
    of group 1, and `s` makes `.` cross newlines, which these prompts are full of.
    `regexp_extract` also returns the empty string on no match, which is
    indistinguishable from a match at position 1, so `regexp_matches` decides
    that separately.
    """
    if regex:
        opts = _regex_opts(case_sensitive)
        tail = f", {_lit(opts)}" if opts else ""
        # `(?s:...)` scopes dot-all to the prefix. Passing `s` as an option would
        # apply it to the user's pattern too, so `.` would cross newlines while
        # locating a match but not while finding one — and the locator would
        # settle on text the predicate never matched, windowing out the real hit.
        prefix = f"'^(?s:(.*?))(' || {_lit(pattern)} || ')'"
        located = f"len(regexp_extract({col}, {prefix}, 1{tail})) + 1"
        matched = f"regexp_matches({col}, {_lit(pattern)}{tail})"
        return f"CASE WHEN NOT {matched} THEN 0 ELSE {located} END"
    if case_sensitive:
        return f"position({_lit(pattern)} IN {col})"
    return f"position(lower({_lit(pattern)}) IN lower({col}))"


def _regex_opts(case_sensitive: bool) -> str:
    """The flags the matching predicate uses, and therefore the only flags any
    expression locating that match may use. Dot-all is deliberately absent: `.`
    not crossing a newline is part of what the user's pattern means."""
    return "" if case_sensitive else "i"


def _match_len_sql(col: str, pattern: str, regex: bool, case_sensitive: bool) -> str:
    """Length of the first match, for centring the snippet on it."""
    if not regex:
        return str(len(pattern))
    opts = _regex_opts(case_sensitive)
    tail = f", {_lit(opts)}" if opts else ""
    return f"len(regexp_extract({col}, {_lit(pattern)}, 0{tail}))"


def window_chars(pattern: str, regex: bool) -> int:
    """How much of the matching string to pull back for the snippet.

    A literal's length is chosen by the caller, so the window grows to hold it:
    searching for an 8,000-character boilerplate paragraph and getting back a
    snippet that cannot contain it would fail the one invariant these artifacts
    carry. A regex match's length is chosen by the *data* and is unbounded — a
    `(?s).*` matches a whole 100k-character response — so that stays capped, and
    the ceiling is documented rather than removed. A snippet is evidence that a
    row matched, and every record links to the untruncated row on the hub; the
    `context` layer makes the same trade at 4,000 characters.
    """
    if regex:
        return SNIPPET_SOURCE_CHARS
    return max(SNIPPET_SOURCE_CHARS, len(pattern) + 2 * SNIPPET_CHARS)


def _window_sql(
    exprs: list[str], test: str, pattern: str, regex: bool, case_sensitive: bool
) -> tuple[str, str, str, str]:
    """The first matching string, windowed on its match, and where the match is.

    Four scalars per group: the window, where it began in the original string,
    where the match sits inside it, and how long the match is. The last two exist
    so the final crop uses the offset SQL already computed rather than searching
    for the match a second time in Python — RE2 accepts patterns `re` cannot
    compile, and re-deriving what is already known is what put the window in the
    wrong place in the first place.

    Written as correlated scalar subqueries over a one-row derived table so the
    list filtering happens once and every expression can refer to the string by
    name. Costs nothing over the wire; the column is already read by then.
    """
    first = f"{_matching(exprs, test)}[1]"
    chars = window_chars(pattern, regex)
    # Context before the match, which is what centres the *match* in the window
    # rather than centring the window on the match's start. Half the window is
    # right only while the match is short: a 10,000-character literal in a 10,480
    # window offset by half would have its last 4,760 characters cut off the end,
    # so the widening bought nothing. A literal's length is known here; a regex
    # match's is not, so that keeps the halved lead.
    half = chars // 2 if regex else max(0, (chars - len(pattern)) // 2)
    offset = _offset_sql("s", pattern, regex, case_sensitive)
    start = f"greatest(1, {offset} - {half})"
    scalar = f"FROM (SELECT {first} AS s)"

    def q(expr):
        return f"(SELECT CASE WHEN s IS NULL THEN NULL ELSE {expr} END {scalar})"

    return (
        q(f"substr(s, {start}, {chars})"),
        q(start),
        # Where the match lands inside the window. Zero when the pattern did not
        # match this string at all, which `snippet` reads as "no offset known".
        q(f"CASE WHEN {offset} = 0 THEN 0 ELSE {offset} - {start} + 1 END"),
        q(_match_len_sql("s", pattern, regex, case_sensitive)),
    )


def snippet(
    text: str | None,
    pattern: str,
    regex: bool,
    case_sensitive: bool,
    window_start: int = 1,
    match_at: int | None = None,
    match_len: int | None = None,
    max_chars: int = SNIPPET_SOURCE_CHARS,
) -> str | None:
    """A window of `text` centred on the first match, so a count can be read.

    `text` is already a match-containing window cut by `_window_sql`, and
    `window_start` is where that window began in the original string — which is
    what the leading ellipsis has to know about, since text elided by SQL is just
    as elided as text this function drops.

    `match_at` (1-based, within `text`) and `match_len` come from the same SQL
    that cut the window, and are what the scan passes. Searching again in Python
    is only the fallback for a direct call, and cannot be the primary route: RE2
    accepts patterns `re` rejects, and on those the search failed, the crop fell
    back to offset zero, and the saved snippet was 4,000 characters short of the
    match the window had been built around.
    """
    if not text:
        return None
    start, length = (match_at - 1 if match_at else None), match_len
    if start is None:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            m = re.search(pattern if regex else re.escape(pattern), text, flags)
        except re.error:
            m = None
        start, length = (m.start(), len(m.group(0))) if m else (0, 0)
    length = length or 0
    # Context each side, and the width needed to hold the match itself. A match
    # longer than SNIPPET_CHARS used to make the context negative, so the crop
    # started *inside* the match and returned its middle 240 characters — which
    # for a literal over 240 characters does not contain the literal, and past
    # 8,000 walked off the end of the window and returned two ellipses. A pattern
    # that long is a real thing to search for; the wrapper-template one here is
    # already 32 characters and a boilerplate paragraph would be hundreds.
    context = max(0, (SNIPPET_CHARS - length) // 2)
    width = max(SNIPPET_CHARS, min(length + 2 * context, max_chars))
    lead = max(0, min(start - context, max(0, len(text) - 1)))
    out = text[lead:lead + width]
    cut_before = lead > 0 or window_start > 1
    return ("…" if cut_before else "") + out + ("…" if lead + width < len(text) else "")


def _pick(kept: list, by_snippet: dict, limit: int, text: str | None, record: dict) -> None:
    """Keep the `limit` records whose snippet hashes smallest.

    Which examples get saved has to depend on the match set and nothing else.
    Keeping the first N that arrive does not: DuckDB reads shards in parallel, so
    re-running a scan returns the same counts in a different order and rewrites
    every example — the artifact stops being byte-reproducible, and "regenerate
    it" turns into unreadable churn.

    Hashing also samples better than arrival order did. The first N matches all
    came off whichever shard was read first, so 20 examples out of 134 in the
    Think RL mix were 12 consecutive unit-test assertions from one source. A
    digest-ordered N spreads across the mix instead.
    """
    if text is None or limit <= 0:
        return
    digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    # Negated so heapq's min-heap roots at the largest hash, the one to evict.
    neg = -int.from_bytes(digest, "big")
    if text in by_snippet:
        return
    if len(kept) < limit:
        by_snippet[text] = record
        heapq.heappush(kept, (neg, text))
    elif neg > kept[0][0]:
        _, evicted = heapq.heapreplace(kept, (neg, text))
        by_snippet.pop(evicted, None)
        by_snippet[text] = record


def scan(
    con,
    from_sql: str,
    exprs: dict[str, list[str]],
    source: str | None,
    pattern: str,
    regex: bool = False,
    case_sensitive: bool = False,
    examples: int = 20,
) -> dict:
    """Count matching rows over every shard, grouped by source and by group.

    One pass: the query returns only the rows that matched, with a flag per
    group and a snippet, and the tallying happens here. Running a separate
    aggregate query would double the bytes pulled, because the filtering is local
    — DuckDB has to read a column to search it either way.
    """
    test = _element_test(pattern, regex, case_sensitive)
    groups = list(exprs)

    select = [f"{source or 'NULL'} AS src"]
    for g in groups:
        select.append(f"({_match_sql(exprs[g], test)}) AS hit_{g}")
    for g in groups:
        # list_filter keeps order, so [1] is the first matching string in the group.
        window, at, mo, ml = _window_sql(exprs[g], test, pattern, regex, case_sensitive)
        select += [f"{window} AS snip_{g}", f"{at} AS at_{g}",
                   f"{mo} AS mo_{g}", f"{ml} AS ml_{g}"]

    any_match = " OR ".join(f"({_match_sql(exprs[g], test)})" for g in groups)
    con.execute(f"SELECT {', '.join(select)} FROM {from_sql} WHERE {any_match}")
    matched = 0
    by_source: dict[str, int] = {}
    by_group = {g: 0 for g in groups}
    by_source_group: dict[str, dict[str, int]] = {}
    # The kept examples, as a bounded max-heap on -hash: the N snippets with the
    # smallest digest survive, whatever order the rows arrive in. See `_pick`.
    kept: list[tuple[int, str]] = []
    by_snippet: dict[str, dict] = {}
    while True:
        batch = con.fetchmany(5000)
        if not batch:
            break
        for row in batch:
            src = source_label(row[0], has_column=source is not None)
            flags = {g: bool(row[1 + i]) for i, g in enumerate(groups)}
            base = 1 + len(groups)
            snips = {g: tuple(row[base + 4 * i: base + 4 * i + 4])
                     for i, g in enumerate(groups)}
            matched += 1
            by_source[src] = by_source.get(src, 0) + 1
            bucket = by_source_group.setdefault(src, {g: 0 for g in groups})
            for g, on in flags.items():
                if on:
                    by_group[g] += 1
                    bucket[g] += 1
            if examples:
                where = [g for g, on in flags.items() if on]
                text, at, mo, ml = next(
                    (snips[g] for g in where if snips[g][0]), (None, 1, 0, 0)
                )
                shown = snippet(
                    text, pattern, regex, case_sensitive, at or 1, mo or None, ml,
                    max_chars=window_chars(pattern, regex),
                )
                _pick(kept, by_snippet, examples, shown, {
                    "source": src,
                    "groups": where,
                    "snippet": shown,
                })

    return {
        "matched": matched,
        "by_source": _by_count(by_source),
        "by_group": by_group,
        "by_source_group": {
            src: by_source_group[src] for src in _by_count(by_source)
        },
        "examples": [by_snippet[text] for _, text in sorted(kept, key=lambda kv: -kv[0])],
    }


def read_parquet_sql(urls: list[str]) -> str:
    """The FROM clause for a set of shards."""
    return f"read_parquet({urls!r})"


def slugify(pattern: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", pattern.lower()).strip("-")
    return slug[:60] or "pattern"
