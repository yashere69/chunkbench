"""Token-level retrieval metrics, following Chroma's chunking evaluation (Smith & Troynikov,
2024): measure overlap between retrieved text and the labelled excerpts, not whether the
"right chunk id" came back.

    recall    = relevant units retrieved / relevant units
    precision = relevant units retrieved / units retrieved (counted with repeats)
    IoU       = relevant units retrieved / (relevant + retrieved - relevant retrieved)

Counting retrieved units with repeats means overlapping chunks that return the same text
twice are penalised in precision and IoU, which is the cost of overlap.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .units import Spans


@dataclass
class Score:
    recall: float
    precision: float
    iou: float


def covered(units: Spans, start: int, end: int) -> set[int]:
    """Indices of units overlapping [start, end)."""
    starts = [u[0] for u in units]
    ends = [u[1] for u in units]
    lo = bisect_right(ends, start)
    hi = bisect_left(starts, end)
    return set(range(lo, hi))


def score(
    units: dict[str, Spans],
    relevant: list[tuple[str, int, int]],
    retrieved: list[tuple[str, int, int]],
) -> Score:
    rel: set[tuple[str, int]] = set()
    for doc, s, e in relevant:
        rel |= {(doc, i) for i in covered(units[doc], s, e)}
    got: set[tuple[str, int]] = set()
    total = 0
    for doc, s, e in retrieved:
        idx = covered(units[doc], s, e)
        total += len(idx)
        got |= {(doc, i) for i in idx}
    hit = len(rel & got)
    recall = hit / len(rel) if rel else 0.0
    precision = hit / total if total else 0.0
    union = len(rel) + total - hit
    return Score(recall, precision, hit / union if union else 0.0)
