"""Load a corpus and labelled questions, chunk, retrieve, score."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml

from .chunkers import parse
from .metrics import score
from .retrieve import BM25
from .units import UNITS


@dataclass
class Question:
    text: str
    doc: str
    excerpts: list[tuple[int, int]]
    lang: str


def load_corpus(folder: Path) -> dict[str, str]:
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))}
    docs |= {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt"))}
    if not docs:
        raise ValueError(f"no .md or .txt files in {folder}")
    return docs


def locate(text: str, excerpt: str) -> tuple[int, int] | None:
    """Find an excerpt verbatim, or with different whitespace.

    Excerpts are usually copied from a rendered page, where line breaks inside a paragraph
    have become spaces. Matching token by token with any run of whitespace in between finds
    those without accepting anything looser.
    """
    k = text.find(excerpt)
    if k != -1:
        return k, k + len(excerpt)
    tokens = excerpt.split()
    if not tokens:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in tokens), text)
    return m.span() if m else None


def suggest(text: str, excerpt: str) -> str | None:
    """The sentence in the document closest to an excerpt that wasn't found."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?!।])\s+|\n+", text) if s.strip()]
    close = difflib.get_close_matches(excerpt.strip(), sentences, n=1, cutoff=0.5)
    return close[0] if close else None


def _problems(raw: object, docs: dict[str, str]) -> list[tuple[int, str]]:
    if not isinstance(raw, list):
        return [(0, "expected a list of questions")]
    out = []
    for i, q in enumerate(raw, start=1):
        if not isinstance(q, dict) or not {"q", "doc", "excerpts"} <= set(q):
            out.append((i, "needs q, doc and excerpts"))
            continue
        text = docs.get(q["doc"])
        if text is None:
            out.append((i, f"unknown doc {q['doc']!r}"))
            continue
        for ex in q["excerpts"]:
            if locate(text, ex) is None:
                hint = suggest(text, ex)
                out.append(
                    (
                        i,
                        f"excerpt not in {q['doc']}: {ex[:60]!r}"
                        + (f"; closest sentence: {hint[:80]!r}" if hint else ""),
                    )
                )
    return out


def check_questions(path: Path, docs: dict[str, str]) -> list[str]:
    """Every problem in a question file, instead of stopping at the first."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [f"question {i}: {msg}" if i else f"{path}: {msg}" for i, msg in _problems(raw, docs)]


def load_questions(path: Path, docs: dict[str, str]) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    problems = _problems(raw, docs)
    if problems:
        i, msg = problems[0]
        more = (
            f" (and {len(problems) - 1} more; run `chunkbench check`)" if len(problems) > 1 else ""
        )
        raise ValueError((f"question {i}: " if i else "") + msg + more)
    out = []
    for q in raw:
        text = docs[q["doc"]]
        spans = [span for ex in q["excerpts"] if (span := locate(text, ex))]
        out.append(Question(q["q"], q["doc"], spans, q.get("lang", "en")))
    return out


def evaluate(
    docs: dict[str, str],
    questions: list[Question],
    strategies: list[str],
    k: int,
    unit_names: list[str],
) -> list[dict]:
    units = {u: {name: UNITS[u](text) for name, text in docs.items()} for u in unit_names}
    rows = []
    for spec in strategies:
        label, chunker = parse(spec)
        chunks = [(name, s, e) for name, text in docs.items() for s, e in chunker(text)]
        index = BM25([docs[n][s:e] for n, s, e in chunks])
        per_lang: dict[tuple[str, str], list] = {}
        for q in questions:
            got = [chunks[i] for i in index.top(q.text, k)]
            rel = [(q.doc, s, e) for s, e in q.excerpts]
            for u in unit_names:
                per_lang.setdefault((u, q.lang), []).append(score(units[u], rel, got))
        sizes = [len(docs[n][s:e].split()) for n, s, e in chunks]
        for (u, lang), scores in sorted(per_lang.items()):
            rows.append(
                {
                    "strategy": label,
                    "unit": u,
                    "lang": lang,
                    "questions": len(scores),
                    "chunks": len(chunks),
                    "avg_words": round(mean(sizes), 1),
                    "recall": round(100 * mean(s.recall for s in scores), 1),
                    "precision": round(100 * mean(s.precision for s in scores), 1),
                    "iou": round(100 * mean(s.iou for s in scores), 1),
                }
            )
    return rows
