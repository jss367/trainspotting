"""Language detection: chunking, the noise strip, and the mixed-language prompts
the per-line weighting exists for.

`detect` runs the real py3langid model — these prompts are the shapes the Dolci
mixes actually contain. The threshold arithmetic is tested against a stub
classifier instead, so a model update can't quietly move what "confidence 0.5"
means.
"""

import pytest

from trainspotting import languages

# An English judge/translation wrapper around a Chinese payload. Classifying the
# whole blob at once is the failure this module was written to avoid.
EN_WRAPPER_ZH_PAYLOAD = """You are given a question in Chinese. Translate it into English, then answer it.

问题：为什么在高海拔地区烧水时，水的沸点会明显下降？请结合大气压强的变化来解释这个现象，并说明这对高原地区的烹饪有什么实际影响。
如果可以的话，请举一个具体的例子来说明，比如在海拔四千米的地方煮饭需要注意什么。
"""

ZH_ASIDE_IN_EN_PROMPT = """Rate the following response on a scale of 1 to 5 for how helpful, accurate and complete it is. Explain your reasoning before giving the final numeric score in the required format.
The user asked about the weather and the assistant gave a thorough forecast for the week ahead.
用户的问题是关于天气的。
"""


class StubIdentifier:
    """Returns a scripted (code, prob) per chunk, in order."""

    def __init__(self, answers):
        self.answers = list(answers)

    def classify(self, chunk):
        return self.answers.pop(0)


@pytest.fixture
def stub(monkeypatch):
    def install(answers):
        monkeypatch.setattr(languages, "_identifier", StubIdentifier(answers))

    return install


def test_mixed_prompt_reports_the_language_holding_the_most_text():
    """The payload is Chinese and the instruction is English, so the prompt
    counts as Chinese — but the English wrapper drags the confidence well below
    the 1.0 a whole-blob classification hands out."""
    code, confidence = languages.detect(EN_WRAPPER_ZH_PAYLOAD)

    assert code == "zh"
    assert languages.MIN_CONFIDENCE < confidence < 0.8


def test_a_short_foreign_aside_does_not_flip_an_english_prompt():
    code, confidence = languages.detect(ZH_ASIDE_IN_EN_PROMPT)

    assert code == "en"
    assert confidence > languages.MIN_CONFIDENCE


def test_a_prompt_that_is_only_code_and_math_is_undetermined():
    """Not "whatever Latin-ish language its symbols resemble"."""
    prompt = """```python
def solve(n):
    return sum(range(n))
```
See $O(n^2)$ and https://example.com/docs \\textbf{x}
"""
    assert languages.detect(prompt) == (languages.UNDETERMINED, 0.0)


def test_short_prompts_are_undetermined_rather_than_guessed():
    assert languages.detect("Hi there") == (languages.UNDETERMINED, 0.0)
    assert languages.detect("") == (languages.UNDETERMINED, 0.0)


def test_scripts_with_combining_marks_clear_the_length_threshold():
    """Devanagari vowel signs are not `isalpha`; counting only those would put
    this Hindi prompt under MIN_LETTERS and report it as undetermined."""
    hindi = "क्या आप बता सकते हैं"

    assert sum(1 for c in hindi if c.isalpha()) < languages.MIN_LETTERS
    assert languages._letters(hindi) >= languages.MIN_LETTERS
    assert languages.detect(hindi)[0] == "hi"


def test_an_even_split_is_rejected_rather_than_resolved_by_dict_order(stub):
    """A tie lands exactly on the threshold, which is `<=`, so nothing wins."""
    stub([("en", 1.0), ("zh", 1.0)])
    text = "abcdefghijklmnop\nqrstuvwxyzabcdef"

    code, share = languages.detect(text)

    assert code == languages.UNDETERMINED
    assert share == pytest.approx(0.5)


def test_confidence_divides_by_the_full_letter_mass(stub):
    """A single hesitant chunk stays hesitant. Dividing by the summed weight
    instead would hand every one-line prompt a flat 1.0."""
    stub([("en", 0.4)])

    assert languages.detect("the quick brown fox jumps") == (languages.UNDETERMINED, pytest.approx(0.4))


def test_a_confident_single_chunk_passes(stub):
    stub([("de", 0.95)])

    code, share = languages.detect("der schnelle braune fuchs")

    assert (code, share) == ("de", pytest.approx(0.95))


def test_strip_noise_drops_every_non_language_span():
    text = (
        "Look at ```\ncode block\n``` and `inline` and $x^2$ and $$y$$ and "
        "\\[z\\] and \\(w\\) and https://example.com/a and www.example.com "
        "and \\textbf{bold}"
    )
    stripped = languages.strip_noise(text)

    for gone in ("code block", "inline", "x^2", "y", "z", "w", "example.com", "textbf"):
        assert gone not in stripped
    assert "Look at" in stripped


class TestChunks:
    def test_splits_on_every_line_break_not_on_blank_lines(self):
        """Translation templates put the instruction and the payload on adjacent
        lines; a paragraph-level split would leave them in one chunk."""
        text = "Translate this sentence into French please\nLe chat dort sur le canapé"

        assert languages._chunks(text) == [
            "Translate this sentence into French please",
            "Le chat dort sur le canapé",
        ]

    def test_a_runt_merges_into_the_chunk_before_it(self):
        assert languages._chunks("a long enough line of text\nrunt") == [
            "a long enough line of text\nrunt"
        ]

    def test_short_lines_accumulate_until_worth_classifying(self):
        assert languages._chunks("ab\ncd\nef\ngh\nij\nkl\nmn") == [
            "ab\ncd\nef\ngh\nij\nkl\nmn"
        ]

    def test_a_prompt_too_short_to_chunk_is_still_returned(self):
        """Nothing to merge into, so it comes back whole; `detect` is what
        rejects it for being short."""
        assert languages._chunks("hello") == ["hello"]

    def test_no_chunk_is_dropped(self):
        text = (
            "Answer the following question carefully\n"
            "short\n"
            "Then explain your reasoning in full sentences\n"
            "x\n"
            "y"
        )
        chunks = languages._chunks(text)

        assert "\n".join(chunks) == text
        assert len(chunks) == 2

    def test_every_chunk_but_none_is_worth_classifying(self):
        text = "The first line here is long enough\nand so is this second one\nrunt"
        chunks = languages._chunks(text)

        assert len(chunks) == 2
        assert all(languages._letters(c) >= languages.MIN_LETTERS for c in chunks)


def test_name_falls_back_to_the_code():
    assert languages.name("zh") == "Chinese"
    assert languages.name(languages.UNDETERMINED) == "undetermined"
    assert languages.name("xx") == "xx"


def test_detect_all_preserves_order():
    prompts = [EN_WRAPPER_ZH_PAYLOAD, "Hi", ZH_ASIDE_IN_EN_PROMPT]

    assert [code for code, _ in languages.detect_all(prompts)] == [
        "zh",
        languages.UNDETERMINED,
        "en",
    ]
