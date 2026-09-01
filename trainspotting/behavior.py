"""Turn an observed behavior into search queries over the training data.

Users usually start from a transcript ("the model claims its knowledge cutoff
is September 2021"), not from a search string. This module extracts the phrases
in that text worth searching for: anchored on at least one token unlikely to
appear by chance — a name, a number, a word in the long tail — and as much of
the surrounding phrase as the sentence holding it allows. The anchor is what
makes a query selective, not its length, so `Assistant: ChatGPT` yields
`ChatGPT` rather than nothing. Extraction is local and deterministic; no model
is called, so the queries are inspectable before anything is spent searching
with them.

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

# The punctuation that can end a segment, captured so the splitter can say
# which one did: a full stop starts a new sentence after it and a colon
# does not, and `_weight` treats the following capital differently.
_BOUNDARY = re.compile(r"([.!?;:])[\"'\u201d\u2019)\]}]*\s+")


def _sentences(text: str):
    """(segment, opens_a_sentence) for each verbatim span a query may not cross.

    Newlines and sentence punctuation both end a segment: a phrase stitched
    across a sentence boundary is not something the training data can contain
    verbatim, so it is not worth searching for. A closing quote or bracket sits
    between the punctuation and the space when the sentence being ended is a
    quoted one, and a boundary missed there is a query stitched across it —
    `OpenAI." Then it` is not a phrase any training row contains.

    The flag says whether the segment's first word actually begins a sentence,
    which is not the same as beginning a segment. `_weight` refuses to treat a
    sentence's opening capital as an anchor, because every sentence has one;
    but a colon or semicolon capitalizes nothing, so the word after one carries
    its capital on its own account. Without the distinction, splitting
    `Assistant: Claude` at the colon made `Claude` look sentence-initial and
    the identity being traced scored as an ordinary word.
    """
    for line in text.splitlines():
        parts = _BOUNDARY.split(line)
        # `parts` alternates segment, delimiter, segment, ... — the delimiter
        # that closed one segment is what decides whether the next one opens a
        # sentence, so it is tracked across the loop rather than re-derived.
        opens = True
        for i in range(0, len(parts), 2):
            # The captured punctuation goes back on the segment it closed, so a
            # query still reads as the text it was cut from ("…September 2021.")
            # while the splitter keeps the delimiter it needs to classify what
            # follows. Anything between it and the space — a closing quote or
            # bracket — stays dropped.
            delim = parts[i + 1] if i + 1 < len(parts) else ""
            seg = ((parts[i] or "") + delim).strip()
            if seg:
                yield seg, opens
            if delim:
                opens = delim in ".!?"


def _weight(token: str, sentence_initial: bool) -> int:
    """0 for function words, 1 for ordinary content, 2 for an anchor.

    An anchor is a token unlikely to appear by chance: it carries a digit
    ("2021", "T-800"), carries a capital past its first letter ("ChatGPT",
    "OpenAI", "AI"), or is capitalized somewhere other than the start of its
    sentence ("September"). Only the last of those depends on position, and it
    has to: every sentence capitalizes its opening word, so a leading "Weather"
    is evidence of nothing.

    An interior capital is different, and it is worth its own case rather than
    falling in with the ordinary title-cased opening word. English capitalizes
    the first letter of a sentence and nothing else in it; a token shaped
    ChatGPT or OpenAI was not shaped by that rule, so it is a name wherever it
    sits. Without this a transcript opening on the name — "ChatGPT is a large
    language model trained to assist people", which is the behavior `trace`
    exists to find — yielded no query at all and sent the user to `ask`.
    """
    segments = _WORD.findall(token)
    if not segments:
        return 0
    # Every word segment of the token, and read before the stopword list rather
    # than after it. Punctuation inside a whitespace-delimited token hides the
    # rest of it two different ways: `version-4217` and `iso-9001` read as the
    # ordinary words before their hyphens, and `T-800`, `S&P500` and `I/O-2024`
    # read as nothing at all, because `t`, `s` and `i` are on the list for the
    # contractions they end. A digit or an interior capital is a claim about
    # the whole token, so it settles the weight before a rule about one segment
    # of it gets a say — and `don't`, `it's` and `I'd` still reach the list,
    # since none of them carries either signal.
    word = "".join(segments)
    if any(ch.isdigit() for ch in word):
        return 2
    if any(ch.isupper() for ch in word[1:]):
        return 2
    core = segments[0]
    if core.lower() in STOPWORDS:
        return 0
    # The title-case test stays on the first segment: that is the letter the
    # sentence-capitalization rule acts on.
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
    for s_idx, (seg, opens) in enumerate(_sentences(text)):
        # No minimum segment length. The anchor requirement below is the real
        # floor on how generic a query may be, and a word count on top of it
        # threw away the shortest segments that are nothing but anchor:
        # `Knowledge cutoff: September 2021` splits at the colon into two
        # two-word segments, `Assistant: ChatGPT` into two one-word ones, and a
        # three-word minimum answered all of them with "no distinctive phrase".
        tokens = seg.split()
        # Only the first word of a segment that really opens a sentence gets the
        # no-anchor treatment. After a colon nothing forced the capital, so
        # `Assistant: Claude` keeps `Claude` as the name it is.
        weights = [_weight(t, i == 0 and opens) for i, t in enumerate(tokens)]
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
