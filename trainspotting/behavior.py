"""Turn an observed behavior into search queries over the training data.

Users usually start from a transcript ("the model claims its knowledge cutoff
is September 2021"), not from a search string. This module extracts the phrases
in that text worth searching for: long enough to be selective, and anchored on
at least one token unlikely to appear by chance — a name, a number, a word in
the long tail. Extraction is local and deterministic; no model is called, so
the queries are inspectable before anything is spent searching with them.

The fan-out over stages lives in the CLI. Ranking what comes back is the
reader's job: a query that hits everywhere may mean the behavior is trained
everywhere, or that the phrase was never distinctive — the per-query counts
are printed so the two are tellable apart.
"""

import re

# Window size in words. The phrases this exists to find run six to eight words
# ("As an AI language model developed by OpenAI", "my knowledge cutoff is
# September 2021"); shorter windows lose the selectivity, longer ones dilute
# the anchor density that ranks them.
WINDOW = 8
MIN_WORDS = 3

# Small on purpose: this list only has to stop function words from counting as
# signal, not model English. "i" covers the pronoun, which capitalization would
# otherwise promote to an anchor mid-sentence.
STOPWORDS = frozenset(
    """a an the and or but nor so yet of to in on at by for from with without
    about as into onto over under is are was were be been being am do does did
    have has had will would can could shall should may might must not no i you
    he she it we they me him her us them my your his its our their this that
    these those there here what which who whom whose when where why how if
    then than too very just also only own same such more most other some any
    all each few both s t don ll re ve d m won""".split()
)

_WORD = re.compile(r"\w+", re.UNICODE)


def _sentences(text: str):
    """Verbatim segments a query may not cross.

    Newlines and sentence punctuation both end a segment: a phrase stitched
    across a sentence boundary is not something the training data can contain
    verbatim, so it is not worth searching for.
    """
    for line in text.splitlines():
        # A closing quote or bracket sits between the punctuation and the
        # space when the sentence being ended is a quoted one, and a boundary
        # missed there is a query stitched across it: `OpenAI." Then it` is not
        # a phrase any training row contains.
        for seg in re.split(r"(?<=[.!?;:])[\"'\u201d\u2019)\]]*\s+", line):
            seg = seg.strip()
            if seg:
                yield seg


def _weight(token: str, sentence_initial: bool) -> int:
    """0 for function words, 1 for ordinary content, 2 for an anchor.

    An anchor is a token unlikely to appear by chance: it carries a digit
    ("2021") or is capitalized somewhere other than the start of its sentence
    ("OpenAI", "September"). Sentence-initial capitals prove nothing — every
    sentence has one.
    """
    m = _WORD.search(token)
    if not m:
        return 0
    core = m.group(0)
    if core.lower() in STOPWORDS:
        return 0
    if any(ch.isdigit() for ch in core):
        return 2
    if core[0].isupper() and not sentence_initial:
        return 2
    return 1


def distinctive_ngrams(text: str, max_queries: int = 6) -> list[str]:
    """The best word windows in `text` to search the training data for.

    Every window of up to WINDOW words within a sentence is a candidate, scored
    by the summed weights of its tokens; a window with no anchor is dropped
    outright, because an all-common-word phrase matches training rows by
    coincidence rather than provenance. Selection is greedy by score, and a
    window is skipped when it brings no anchor an already-chosen query covers
    already, or when it repeats one verbatim (boilerplate recurs in
    transcripts) — so `max_queries` buys `max_queries` distinct anchors rather
    than eight offsets of the best sentence.
    """
    candidates = []  # (score, order, anchors, query)
    for s_idx, seg in enumerate(_sentences(text)):
        tokens = seg.split()
        if len(tokens) < MIN_WORDS:
            continue
        weights = [_weight(t, i == 0) for i, t in enumerate(tokens)]
        size = min(WINDOW, len(tokens))
        for start in range(len(tokens) - size + 1):
            window = weights[start : start + size]
            if 2 not in window:
                continue
            score = sum(window)
            anchors = {(s_idx, start + i) for i, w in enumerate(window) if w == 2}
            # Trim leading and trailing function words off the emitted query.
            # Search is an AND over the query's tokens, so a boundary "so I"
            # only narrows the match to rows that also contain those words near
            # the anchor — excluding the training row that phrased the same
            # distinctive span slightly differently. Dedup reads the window's
            # anchors, not the trimmed span, so trimming cannot let the same
            # anchor through twice.
            lo, hi = start, start + size - 1
            while lo < hi and weights[lo] == 0:
                lo += 1
            while hi > lo and weights[hi] == 0:
                hi -= 1
            query = " ".join(tokens[lo : hi + 1])
            candidates.append((score, len(candidates), anchors, query))

    # Order stays in the key so equal scores resolve to the earlier window and
    # the result is deterministic in the input text alone.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen: list[str] = []
    taken: set[tuple[int, int]] = set()
    seen: set[str] = set()
    for _, _, anchors, query in candidates:
        if len(chosen) >= max_queries:
            break
        # Overlap alone is not a reason to drop a window — what makes a second
        # query worth issuing is a second anchor. Rejecting anything that
        # touched a chosen window meant one sentence yielded one query however
        # many distinct anchors it held: the documented example, "As an AI
        # language model developed by OpenAI, my knowledge cutoff is September
        # 2021", returned the OpenAI window and no way to reach `September
        # 2021` at any `--max-queries`. A candidate now has to bring an anchor
        # nothing chosen has covered, which keeps near-duplicate windows out
        # without hiding the other half of the sentence.
        if not anchors - taken or query.lower() in seen:
            continue
        chosen.append(query)
        taken |= anchors
        seen.add(query.lower())
    return chosen
