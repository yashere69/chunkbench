"""Load a corpus and labelled questions, chunk, retrieve, score."""

from __future__ import annotations

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


def load_questions(path: Path, docs: dict[str, str]) -> list[Question]:
    out = []
    for i, q in enumerate(yaml.safe_load(path.read_text(encoding="utf-8")), start=1):
        text = docs.get(q["doc"])
        if text is None:
            raise ValueError(f"question {i}: unknown doc {q['doc']!r}")
        spans = []
        for ex in q["excerpts"]:
            k = text.find(ex)
            if k == -1:
                raise ValueError(
                    f"question {i}: excerpt not found verbatim in {q['doc']}: {ex[:60]!r}"
                )
            spans.append((k, k + len(ex)))
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
