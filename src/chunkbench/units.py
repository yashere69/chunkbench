"""Units in which overlap is counted. Each function returns (start, end) character spans.

The metric is "how much of the relevant text did we retrieve, and how much else came
with it". What counts as one piece of text changes the answer for scripts like Devanagari:
whitespace words are long, single code points split a letter from its vowel sign, and
grapheme clusters are what a reader sees as one character.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

Spans = list[tuple[int, int]]

VIRAMA = {"्", "্", "੍", "્", "୍", "்", "్", "್", "്"}
JOINERS = {"‌", "‍"}


def words(text: str) -> Spans:
    return [m.span() for m in re.finditer(r"\S+", text)]


def chars(text: str) -> Spans:
    return [(i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]


def graphemes(text: str) -> Spans:
    """Approximate extended grapheme clusters, with Indic conjuncts kept whole.

    A combining mark or joiner attaches to the previous cluster, and a consonant after a
    virama joins it too (क्ष is one cluster, as in Unicode 15.1's conjunct rule). That's
    enough for Devanagari and Latin; it is not a full UAX #29 implementation.
    """
    spans: Spans = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        prev = text[i - 1] if i else ""
        attach = (
            unicodedata.category(ch).startswith("M")
            or ch in JOINERS
            or prev in VIRAMA
            or prev in JOINERS
        )
        if attach and spans and spans[-1][1] == i:
            spans[-1] = (spans[-1][0], i + 1)
        else:
            spans.append((i, i + 1))
    return spans


UNITS: dict[str, Callable[[str], Spans]] = {
    "words": words,
    "chars": chars,
    "graphemes": graphemes,
}
