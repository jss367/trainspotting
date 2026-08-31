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


class TestPageFieldsAgainstCommittedSamples:
    """What the page's `searchFields` reads out of a record, run for real.

    The index and the scan have to cover the same text, and only one of them is
    Python: `searchindex._strings` walks every string in a record, so anything
    the page's field list leaves out is a silent false negative — the prefilter
    keeps the file, the page downloads it, and the scan reports no match for
    text that is in the data. This runs the page's own function against the
    committed samples to keep the two honest.
    """

    SRC = Path(__file__).resolve().parent.parent / "docs" / "index.html"

    @staticmethod
    def _function_source(name):
        """`function name(...){...}` out of index.html, by brace matching."""
        src = TestPageFieldsAgainstCommittedSamples.SRC.read_text()
        start = src.index(f"function {name}(")
        depth, i = 0, src.index("{", start)
        while True:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    return src[start:i + 1]
            i += 1

    @pytest.fixture(scope="class")
    def run_search_fields(self, tmp_path_factory):
        """Call the page's searchFields on a list of records, via node."""
        import shutil
        import subprocess
        if not shutil.which("node"):
            pytest.skip("needs node to run the page's own function")
        if not self.SRC.exists():
            pytest.skip("no docs/index.html in this checkout")
        script = tmp_path_factory.mktemp("js") / "fields.js"
        script.write_text(
            self._function_source("searchFields")
            + "\nconst recs = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
            + "\nconsole.log(JSON.stringify(recs.map(searchFields)));\n"
        )

        def run(records):
            out = subprocess.run(
                ["node", str(script)], input=json.dumps(records),
                capture_output=True, text=True, check=True,
            )
            return json.loads(out.stdout)
        return run

    @pytest.fixture(scope="class")
    def context_records(self):
        files = sorted(DATA.glob("*.context.json"))
        if not files:
            pytest.skip("no committed context samples")
        return {f.name: json.loads(f.read_text())["records"] for f in files}

    def test_reads_user_and_system_turns_the_prompt_does_not_carry(
        self, run_search_fields, context_records
    ):
        """`prompt_full` is the FIRST user turn (extract._first), so a system
        instruction and every later user turn are text only the turn list has.
        Skipping those turns as "the prompt again" hid them from search."""
        checked = 0
        for name, records in context_records.items():
            hidden = []
            for rec in records:
                prompt = (rec.get("prompt_full") or {}).get("text", "")
                for side in (None, "chosen", "rejected"):
                    turns = (rec.get(side) or {}).get("turns") if side else rec.get("turns")
                    for turn in turns or []:
                        if turn["role"] in ("user", "system") and turn["text"] not in prompt:
                            hidden.append((rec, turn["text"]))
                            break
                    else:
                        continue
                    break
            if not hidden:
                continue
            fields = run_search_fields([rec for rec, _ in hidden])
            for (_, text), got in zip(hidden, fields):
                assert any(text in f["text"] for f in got), (
                    f"{name}: a {len(text)}-character turn is in the record and in the "
                    "trigram index, but searchFields never returns it"
                )
                checked += 1
        assert checked, "no multi-turn or system-prompted records in the samples to check"

    def test_returns_each_text_once(self, run_search_fields, context_records):
        """A DPO pair repeats the user turn in both sides and the prompt again
        at the top; the same text under three names is three copies of one hit."""
        for name, records in context_records.items():
            for got in run_search_fields(records[:60]):
                texts = [f["text"] for f in got]
                assert len(texts) == len(set(texts)), name
