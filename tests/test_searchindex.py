"""The trigram index behind the site's search box.

The index decides which sample files the page bothers to download, so the cost
of a wrong answer is asymmetric: a file kept that holds nothing wastes a fetch,
a file dropped that holds a match loses that match silently. These tests are
mostly about the second kind — including the three cases that made this a
trigram index rather than a word index (a query inside a word, a query inside
an accented word, and a language that writes no spaces).

`trigrams` is also half of a contract with docs/index.html, which cuts the
query with `searchGrams`. The cases below are where a plausible-but-different
rule on either side would drop files.
"""

import json
from pathlib import Path

import pytest

from trainspotting import searchindex

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


def index_of(**files):
    """Index one record per named file, each record holding one string."""
    return searchindex.build({name: {"records": [{"text": text}]} for name, text in files.items()})


def survives(index, query):
    """The files a query prefilters to, the way the page computes it."""
    grams = searchindex.trigrams(query)
    if not grams:
        return set(index["files"])
    keep = set(range(len(index["files"])))
    for g in grams:
        keep &= set(index["grams"].get(g, []))
    return {index["files"][i] for i in keep}


class TestTrigrams:
    def test_cuts_every_run_of_three(self):
        assert searchindex.trigrams("chatgpt") == {"cha", "hat", "atg", "tgp", "gpt"}

    def test_lowercases(self):
        assert searchindex.trigrams("ChatGPT") == searchindex.trigrams("chatgpt")

    def test_keeps_spaces_and_punctuation(self):
        # The page matches substrings literally, so a phrase's spaces are part
        # of what is searched and have to be part of what is indexed.
        assert "s a" in searchindex.trigrams("as an AI")

    def test_shorter_than_a_trigram_yields_nothing(self):
        # Which the page reads as "cannot prefilter", not as "matches nothing".
        assert searchindex.trigrams("ai") == set()

    def test_cuts_by_code_point_not_utf16_unit(self):
        # These prompts open with 💬 and mathematical-bold 𝗖𝗵𝗮𝘁𝗚𝗣𝗧, which are
        # astral. Cutting by UTF-16 unit would split a surrogate pair and index
        # trigrams the page could never ask for.
        assert searchindex.trigrams("💬𝗖𝗵") == {"💬𝗖𝗵"}


class TestBuild:
    def test_maps_each_trigram_to_the_files_holding_it(self):
        idx = index_of(**{"a.json": "shared alpha", "b.json": "shared beta"})
        assert idx["files"] == ["a.json", "b.json"]
        assert idx["grams"]["sha"] == [0, 1]
        assert idx["grams"]["lph"] == [0]
        assert idx["grams"]["bet"] == [1]

    def test_indexes_strings_at_any_depth(self):
        # A DPO record keeps its responses under chosen/rejected → turns → text,
        # and an RL record its reference answer under reward → ground_truth.
        idx = searchindex.build({
            "dpo.json": {"records": [{"chosen": {"turns": [{"text": "As ChatGPT, sure"}]}}]},
            "rl.json": {"records": [{"reward": {"ground_truth": {"text": "As an AI language model"}}}]},
        })
        assert survives(idx, "ChatGPT") == {"dpo.json"}
        assert survives(idx, "AI language model") == {"rl.json"}

    def test_newlines_do_not_glue_characters_together(self):
        # Indexing the record's JSON *text* would see a literal backslash-n and
        # index "1\\nl", losing the "1\nl" the page asks for.
        idx = index_of(**{"a.json": "line1\nline2"})
        assert survives(idx, "line1\nline2") == {"a.json"}

    def test_a_file_with_no_records_contributes_nothing(self):
        idx = searchindex.build({"empty.json": {}, "a.json": {"records": [{"text": "hello"}]}})
        assert idx["files"] == ["a.json", "empty.json"]
        assert survives(idx, "hello") == {"a.json"}


class TestPrefilterKeepsWhatMatches:
    """The cases a word index got wrong. Each query is a real substring of the
    indexed text, so the file must survive."""

    @pytest.mark.parametrize("text, query", [
        ("Interact as ChatGPT", "GPT"),                 # inside a word
        ("résoudre l'équation suivante", "quation"),    # inside an accented word
        ("这是训练数据", "训练数据"),                      # no spaces to tokenize on
        ("💬 𝗖𝗵𝗮𝘁𝗚𝗣𝗧 Interact", "𝗖𝗵𝗮𝘁"),                  # astral characters
        ("As an AI language model, I", "an AI lang"),   # phrase spanning spaces
        ("filtered_wc_sample_500k", "wc_sample"),       # spanning an underscore
    ])
    def test_query_inside_the_text_survives(self, text, query):
        assert text.lower().count(query.lower()), "test bug: query is not in the text"
        assert survives(index_of(**{"a.json": text, "b.json": "unrelated"}), query) == {"a.json"}

    def test_a_string_in_nothing_survives_nowhere(self):
        # The fast path: no fetch at all, and the page can say "no match".
        assert survives(index_of(**{"a.json": "nothing to see"}), "zzzznotathing") == set()


class TestAgainstCommittedSamples:
    """The index the site actually ships, if this checkout has one."""

    @pytest.fixture(scope="class")
    def shipped(self):
        path = DATA / "search-index.json"
        if not path.exists():
            pytest.skip("no exported index — run scripts/export_site_data.py")
        return json.loads(path.read_text())

    def test_every_indexed_file_is_a_sample_the_page_can_read(self, shipped):
        assert shipped["files"]
        assert all(f.endswith((".context.json", ".docs.json")) for f in shipped["files"])
        assert shipped["ngram"] == searchindex.NGRAM

    def test_postings_are_sorted_file_indexes(self, shipped):
        for gram, files in shipped["grams"].items():
            assert files == sorted(set(files)), gram
            assert all(0 <= i < len(shipped["files"]) for i in files), gram

    @pytest.mark.parametrize("query", ["ChatGPT", "as an AI language model", "OpenAI"])
    def test_the_prefilter_keeps_every_file_that_really_contains_it(self, shipped, query):
        kept = survives(shipped, query)
        for name in shipped["files"]:
            if query.lower() in (DATA / name).read_text().lower():
                assert name in kept, f"{query!r} is in {name} but the index drops it"
