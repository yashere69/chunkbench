"""Chunking strategies. Each returns (start, end) character spans into the document.

Spans rather than strings, so the metrics can intersect chunks with labelled excerpts
exactly and overlap between chunks is visible. Size is counted in whitespace words for
every strategy, so "200" means the same thing across them.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .units import Spans, words

Chunker = Callable[[str], Spans]


def fixed(size: int, overlap: int = 0) -> Chunker:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    def run(text: str) -> Spans:
        w = words(text)
        out, i = [], 0
        while i < len(w):
            j = min(i + size, len(w))
            out.append((w[i][0], w[j - 1][1]))
            if j == len(w):
                break
            i = j - overlap
        return out

    return run


SEPARATORS = ["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ", ", " "]


def _n_words(text: str, start: int, end: int) -> int:
    return len(re.findall(r"\S+", text[start:end]))


def recursive(size: int, separators: list[str] | None = None) -> Chunker:
    """Split on the coarsest separator that works, then merge pieces up to `size` words.

    Same idea as LangChain's RecursiveCharacterTextSplitter, but sized in words and with
    the Devanagari danda (।) as a sentence separator.
    """
    seps = separators or SEPARATORS

    def split(text: str, start: int, end: int, level: int) -> Spans:
        if _n_words(text, start, end) <= size or level >= len(seps):
            return [(start, end)]
        sep = seps[level]
        pieces, cur = [], start
        while True:
            k = text.find(sep, cur, end)
            if k == -1:
                pieces.append((cur, end))
                break
            pieces.append((cur, k + len(sep)))
            cur = k + len(sep)
        if len(pieces) == 1:
            return split(text, start, end, level + 1)
        out: Spans = []
        for s, e in pieces:
            out.extend(split(text, s, e, level + 1) if _n_words(text, s, e) > size else [(s, e)])
        return out

    def merge(text: str, pieces: Spans) -> Spans:
        out: Spans = []
        for s, e in pieces:
            if out and _n_words(text, out[-1][0], e) <= size:
                out[-1] = (out[-1][0], e)
            else:
                out.append((s, e))
        return out

    def run(text: str) -> Spans:
        spans = merge(text, split(text, 0, len(text), 0))
        return [_trim(text, s, e) for s, e in spans if text[s:e].strip()]

    return run


def headings(size: int) -> Chunker:
    """One chunk per markdown section; sections longer than `size` go through `recursive`."""
    inner = recursive(size)

    def run(text: str) -> Spans:
        starts = [m.start() for m in re.finditer(r"^#{1,6} ", text, re.M)]
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        bounds = [*starts, len(text)]
        out: Spans = []
        for s, e in zip(bounds, bounds[1:], strict=False):
            if not text[s:e].strip():
                continue
            if _n_words(text, s, e) <= size:
                out.append(_trim(text, s, e))
            else:
                out.extend((s + a, s + b) for a, b in inner(text[s:e]))
        return out

    return run


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def parse(spec: str) -> tuple[str, Chunker]:
    """'fixed:200,50', 'recursive:200', 'headings:300' -> (label, chunker)."""
    name, _, args = spec.partition(":")
    try:
        nums = [int(x) for x in args.split(",") if x] if args else []
    except ValueError:
        raise ValueError(f"strategy {spec!r}: sizes must be whole numbers of words") from None
    if any(n <= 0 for n in nums[:1]) or any(n < 0 for n in nums[1:]):
        raise ValueError(f"strategy {spec!r}: size must be positive and overlap not negative")
    if name != "fixed" and len(nums) > 1:
        raise ValueError(f"strategy {spec!r}: only fixed takes an overlap")
    if name == "fixed":
        return spec, fixed(nums[0] if nums else 200, nums[1] if len(nums) > 1 else 0)
    if name == "recursive":
        return spec, recursive(nums[0] if nums else 200)
    if name == "headings":
        return spec, headings(nums[0] if nums else 300)
    raise ValueError(f"unknown strategy {name!r} (fixed, recursive, headings)")
