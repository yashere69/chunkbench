"""Load a corpus and labelled questions, chunk, retrieve, score."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .chunkers import parse
from .metrics import score
from .retrieve import build
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


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark else ""
        raise ValueError(f"{path}: not valid YAML{where}") from e


def check_questions(path: Path, docs: dict[str, str]) -> list[str]:
    """Every problem in a question file, instead of stopping at the first."""
    raw = _read_yaml(path)
    return [f"question {i}: {msg}" if i else f"{path}: {msg}" for i, msg in _problems(raw, docs)]


def load_questions(path: Path, docs: dict[str, str]) -> list[Question]:
    raw = _read_yaml(path)
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


def select(ranked: list[int], sizes: list[int], k: int, budget: int | None) -> list[int]:
    """Top k, or with a budget, best-first chunks until the next would exceed it (always at
    least one). A budget compares chunkers fairly: two 100-word chunks give the reader
    twice the text of two 50-word chunks, and top-k hides that."""
    if budget is None:
        return ranked[:k]
    out: list[int] = []
    used = 0
    for i in ranked:
        if out and used + sizes[i] > budget:
            break
        out.append(i)
        used += sizes[i]
    return out


def evaluate(
    docs: dict[str, str],
    questions: list[Question],
    strategies: list[str],
    k: int,
    unit_names: list[str],
    retrievers: list[str] | None = None,
    budget: int | None = None,
    embed_url: str | None = None,
    cache: Path | None = None,
) -> list[dict]:
    units = {u: {name: UNITS[u](text) for name, text in docs.items()} for u in unit_names}
    rows = []
    for spec in strategies:
        label, chunker = parse(spec)
        chunks = [(name, s, e) for name, text in docs.items() for s, e in chunker(text)]
        texts = [docs[n][s:e] for n, s, e in chunks]
        words = [len(t.split()) for t in texts]
        for rname in retrievers or ["bm25"]:
            index = build(rname, texts, embed_url, cache)
            rows += _score_one(label, rname, index, chunks, words, questions, units, k, budget)
    return rows


def _score_one(label, rname, index, chunks, words, questions, units, k, budget) -> list[dict]:
    per_lang: dict[tuple[str, str], list] = {}
    taken: list[int] = []
    for q in questions:
        picked = select(index.rank(q.text), words, k, budget)
        taken.append(sum(words[i] for i in picked))
        got = [chunks[i] for i in picked]
        rel = [(q.doc, s, e) for s, e in q.excerpts]
        for u in units:
            per_lang.setdefault((u, q.lang), []).append(score(units[u], rel, got))
    return [
        {
            "strategy": label,
            "retriever": rname,
            "unit": u,
            "lang": lang,
            "questions": len(scores),
            "chunks": len(chunks),
            "avg_words": round(mean(words), 1) if words else 0.0,
            "context_words": round(mean(taken), 1) if taken else 0.0,
            "recall": round(100 * mean(s.recall for s in scores), 1),
            "precision": round(100 * mean(s.precision for s in scores), 1),
            "iou": round(100 * mean(s.iou for s in scores), 1),
        }
        for (u, lang), scores in sorted(per_lang.items())
    ]
