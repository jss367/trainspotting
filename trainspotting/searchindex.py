"""Which committed sample files could contain a given string, so the site can
search without downloading all of them.

The page ships ~30 MB of sampled training examples under `docs/data/`. Someone
looking for a string ("ChatGPT") wants every match across every stage, and the
honest way to find them is to read all that text — which is a rude thing to
fetch on a keystroke. So this indexes the samples at export time: the page loads
the index (about a megabyte gzipped), works out which files could hold the
query, and fetches only those. A string in no sample is answered without a
single fetch.

The unit is a **three-character run, not a word**. Words index smaller and
filter harder, but the page matches substrings the way ⌘F does, and a word
index answers "no" to every query that is part of a word: "GPT" is not a token
of "ChatGPT", "quation" is not a token of "équation", and Chinese writes no
spaces at all, so a word index would report those as absent. Every substring of
three characters or more is a run of trigrams, so a trigram index cannot lose a
match — the price is a coarser filter (the trigrams of a long word are each
common, so several files survive) and a slightly bigger file.

The index is a prefilter, not the answer. It is a deliberate superset: it
indexes every string in a record, including ids and shard paths the page never
searches, and a file whose text holds all of a query's trigrams scattered apart
survives it. The page then substring-scans what it fetched, and that scan is
what produces the hits. A superset costs a wasted fetch; a subset would silently
lose matches, which is why nothing here filters for quality.

Both sides must cut the same trigrams or the prefilter drops files that match.
Python strings iterate by code point and JavaScript strings by UTF-16 unit, so
docs/index.html spreads the query with `[...q]` before slicing — the emoji and
mathematical-bold characters in these prompts are astral, and a `slice(j, j+3)`
there would cut different trigrams than this does.
"""

import json

NGRAM = 3


def trigrams(text: str) -> set:
    """The lowercase three-character runs of one string.

    Shorter than three characters yields nothing: the page falls back to reading
    every sample rather than pretending a one-character query is indexed.
    """
    chars = list(text.lower())
    return {"".join(chars[i:i + NGRAM]) for i in range(len(chars) - NGRAM + 1)}


def _strings(value):
    """Every string anywhere in a record.

    Walking values rather than the record's JSON text is deliberate: in JSON
    source a newline is a literal backslash-n, which would index the trigram
    "\\nl" for a record whose text holds a real line break, and lose "1\\nli"
    — the one the page, which searches parsed strings, actually asks for.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def build(files: dict) -> dict:
    """Index `{filename: parsed json}` into `{files: [...], grams: {gram: [i]}}`.

    Postings are file indexes into `files`, not a bitmask: a bitmask is smaller
    but caps out at 32 files in JavaScript's bitwise operators, and this repo
    already ships twelve.
    """
    names = sorted(files)
    grams = {}
    for i, name in enumerate(names):
        seen = set()
        for record in files[name].get("records") or []:
            for s in _strings(record):
                seen |= trigrams(s)
        for gram in seen:
            grams.setdefault(gram, []).append(i)
    return {"ngram": NGRAM, "files": names, "grams": {g: grams[g] for g in sorted(grams)}}


def build_from_paths(paths) -> dict:
    """`build` over files on disk, keyed by their bare filename."""
    return build({p.name: json.loads(p.read_text()) for p in paths})
