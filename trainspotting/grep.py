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
import os
import re
import sys
import urllib.parse

import requests

BASE = "https://datasets-server.huggingface.co"
HUB = "https://huggingface.co"
PARQUET_BRANCH = "refs/convert/parquet"

# Which part of the training example a column belongs to. The three groups are
# the same cut the `context` layer draws: what the model is asked, what it is
# fit to (or pushed between), and what scores it.
GROUPS = ("prompt", "response", "reference")

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
    "outputs": "response",  # RL: reference rollouts, scored to get the passrate
}
STRUCT_TEXT = {
    ("reward_model", "ground_truth"): "reference",
}
MESSAGE_LISTS = ("messages", "chosen", "rejected", "source_prompt")
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
        out.append(("response", f"list_transform(list_filter({q}, m -> {emitted}), m -> m.content)", both))
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

    for col, typ in schema_.items():
        q = _ident(col)
        if col in MESSAGE_LISTS and typ.startswith("STRUCT"):
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
    total = 0
    for path, size in rows:
        parts = [p.strip() for p in path.split(",")]
        for col, sub in leaves:
            if parts[0] == col and (sub is None or parts[-1] == sub):
                total += size
                break
    return total


def total_rows(con, urls: list[str]) -> int:
    """Row count from the footers, so the denominator costs no data reads."""
    return con.execute(
        f"SELECT sum(num_rows) FROM parquet_file_metadata({urls!r})"
    ).fetchone()[0]


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
    return out


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
        opts = "s" if case_sensitive else "si"
        prefix = f"'^(.*?)(' || {_lit(pattern)} || ')'"
        located = f"len(regexp_extract({col}, {prefix}, 1, {_lit(opts)})) + 1"
        matched = f"regexp_matches({col}, {_lit(pattern)}{'' if case_sensitive else ', ' + _lit('i')})"
        return f"CASE WHEN NOT {matched} THEN 0 ELSE {located} END"
    if case_sensitive:
        return f"position({_lit(pattern)} IN {col})"
    return f"position(lower({_lit(pattern)}) IN lower({col}))"


def _window_sql(exprs: list[str], test: str, pattern: str, regex: bool, case_sensitive: bool) -> tuple[str, str]:
    """(window, 1-based start) of the first matching string, centred on the match.

    Written as a correlated scalar subquery over a one-row derived table so the
    list filtering happens once and both the offset and the cut can refer to the
    string by name. Costs nothing over the wire — the column has already been
    read by the time this is evaluated.
    """
    first = f"{_matching(exprs, test)}[1]"
    half = SNIPPET_SOURCE_CHARS // 2
    start = f"greatest(1, {_offset_sql('s', pattern, regex, case_sensitive)} - {half})"
    scalar = f"FROM (SELECT {first} AS s)"
    window = f"(SELECT CASE WHEN s IS NULL THEN NULL ELSE substr(s, {start}, {SNIPPET_SOURCE_CHARS}) END {scalar})"
    at = f"(SELECT CASE WHEN s IS NULL THEN NULL ELSE {start} END {scalar})"
    return window, at


def snippet(
    text: str | None,
    pattern: str,
    regex: bool,
    case_sensitive: bool,
    window_start: int = 1,
) -> str | None:
    """A window of `text` centred on the first match, so a count can be read.

    `text` is already a match-containing window cut by `_window_sql`, and
    `window_start` is where that window began in the original string — which is
    what the leading ellipsis has to know about, since text elided by SQL is just
    as elided as text this function drops.
    """
    if not text:
        return None
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        m = re.search(pattern if regex else re.escape(pattern), text, flags)
    except re.error:
        # The count came from DuckDB's RE2, which accepts syntax `re` does not.
        # Losing the centring is fine; failing the scan over a snippet is not.
        m = None
    start = m.start() if m else 0
    lead = max(0, start - (SNIPPET_CHARS - len(m.group(0) if m else "")) // 2)
    out = text[lead:lead + SNIPPET_CHARS]
    cut_before = lead > 0 or window_start > 1
    return ("…" if cut_before else "") + out + ("…" if lead + SNIPPET_CHARS < len(text) else "")


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
    if text is None:
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
        window, at = _window_sql(exprs[g], test, pattern, regex, case_sensitive)
        select.append(f"{window} AS snip_{g}")
        select.append(f"{at} AS snipat_{g}")

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
            snips = {g: (row[1 + len(groups) + 2 * i], row[2 + len(groups) + 2 * i])
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
                text, at = next((snips[g] for g in where if snips[g][0]), (None, 1))
                shown = snippet(text, pattern, regex, case_sensitive, at or 1)
                _pick(kept, by_snippet, examples, shown, {
                    "source": src,
                    "groups": where,
                    "snippet": shown,
                })

    return {
        "matched": matched,
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
        "by_group": by_group,
        "by_source_group": by_source_group,
        "examples": [by_snippet[text] for _, text in sorted(kept, key=lambda kv: -kv[0])],
    }


def read_parquet_sql(urls: list[str]) -> str:
    """The FROM clause for a set of shards."""
    return f"read_parquet({urls!r})"


def slugify(pattern: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", pattern.lower()).strip("-")
    return slug[:60] or "pattern"
