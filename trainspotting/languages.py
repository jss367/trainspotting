"""Detect which natural language a training prompt is written in, locally.

No model is called here. py3langid ships its own classifier, so a language
breakdown costs nothing beyond re-fetching the sample — unlike the HHH labels,
which need Claude. Detection is over 97 languages.

Prompts are stripped of code, math, and URLs first. A Python function or a
LaTeX expression carries no natural-language signal, and a detector run on one
returns whichever language its symbol distribution happens to resemble; the
post-training mixes are full of both. Anything left too short or too
low-confidence is reported as `undetermined` rather than guessed at.
"""

import re
import unicodedata

MIN_LETTERS = 12    # below this a detector is guessing; a bare "$x^2 + 1$" reads as whatever
MIN_CONFIDENCE = 0.5
UNDETERMINED = "undetermined"

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_URL = re.compile(r"https?://\S+|www\.\S+")

# ISO 639-1 codes py3langid emits, in English. Kept here rather than pulled from
# a locale library so the site's labels and the CLI's agree exactly.
NAMES = {
    "af": "Afrikaans", "am": "Amharic", "an": "Aragonese", "ar": "Arabic",
    "as": "Assamese", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German",
    "dz": "Dzongkha", "el": "Greek", "en": "English", "eo": "Esperanto",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fi": "Finnish", "fo": "Faroese", "fr": "French", "ga": "Irish",
    "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz",
    "la": "Latin", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "nb": "Norwegian Bokmal", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "or": "Odia", "pa": "Punjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "qu": "Quechua",
    "ro": "Romanian", "ru": "Russian", "rw": "Kinyarwanda", "se": "Northern Sami",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish",
    "ug": "Uyghur", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "vo": "Volapuk", "wa": "Walloon", "xh": "Xhosa", "zh": "Chinese",
    "zu": "Zulu", UNDETERMINED: "undetermined",
}


def name(code: str) -> str:
    return NAMES.get(code, code)


_identifier = None


def _ident():
    """Load the model once. Deferred so importing this module stays cheap."""
    global _identifier
    if _identifier is None:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        _identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    return _identifier


def strip_noise(text: str) -> str:
    """Drop the spans that carry no natural-language signal."""
    for pattern in (_FENCE, _MATH, _URL, _INLINE_CODE, _LATEX_CMD):
        text = pattern.sub(" ", text)
    return text


def _letters(text: str) -> int:
    """Letters, counting combining marks. Indic vowel signs are not `isalpha`, and
    a sentence of Chinese is a third the character count of the same sentence in
    English, so a plain `isalpha` threshold would silently drop the very scripts
    a language breakdown exists to surface."""
    return sum(1 for c in text if c.isalpha() or unicodedata.category(c).startswith("M"))


def _chunks(text: str) -> list[str]:
    """Lines, with runts merged forward so every chunk is worth classifying.

    Split on every line break, not on blank lines: translation and judge
    templates put the English instruction and the foreign-language payload on
    adjacent lines, and a paragraph-level split leaves them in one chunk.
    """
    out, buf = [], ""
    for line in text.split("\n"):
        buf = f"{buf}\n{line}" if buf else line
        if _letters(buf) >= MIN_LETTERS:
            out.append(buf)
            buf = ""
    if buf and out:
        out[-1] += "\n" + buf
    elif buf:
        out.append(buf)
    return out


def detect(text: str) -> tuple[str, float]:
    """(language code, confidence). `undetermined` when there's too little to go on.

    Detection is per line, weighted by length, because a lot of these prompts
    are mixed: an English judge or translation template wrapped around a
    question in another language. Classifying the whole blob at once returns
    neither language — one such prompt came back as Latin, confidently.

    The winner is the language holding the most text. Confidence divides its
    weight by the full letter mass, not by the summed weight, so both ways of
    being unsure pull it down: text split between languages, and text the
    detector was never confident about in the first place. Dividing by the
    summed weight would hand every single-line prompt a flat 1.0 and the
    threshold would never fire. A tie lands exactly on the threshold and is
    rejected, because picking the winner out of a tie only reports dict order.
    """
    stripped = strip_noise(text)
    if _letters(stripped) < MIN_LETTERS:
        return UNDETERMINED, 0.0
    weights: dict[str, float] = {}
    mass = 0
    for chunk in _chunks(stripped):
        code, prob = _ident().classify(chunk)
        letters = _letters(chunk)
        weights[code] = weights.get(code, 0.0) + letters * float(prob)
        mass += letters
    if not mass or not weights:
        return UNDETERMINED, 0.0
    code = max(weights, key=weights.get)
    share = weights[code] / mass
    if share <= MIN_CONFIDENCE:
        return UNDETERMINED, share
    return code, share


def detect_all(prompts: list[str]) -> list[tuple[str, float]]:
    return [detect(p) for p in prompts]
