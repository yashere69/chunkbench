"""Retrievers. BM25 is the default: no embeddings, so results are reproducible and need no
API key. Dense and hybrid retrieval live in `embed.py`; `build` picks one by name.

The tokeniser keeps Devanagari vowel signs inside words. Python's \\w doesn't include
combining marks, so a plain \\w+ would cut किताब into pieces.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[^\W_][\wऀ-ॣ]*|[ऀ-ॣ]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.avgdl = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        df: Counter[str] = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, toks in zip(self.tf, self.toks, strict=True):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(toks) / (self.avgdl or 1))
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        ranked = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [i for i in ranked[:k] if sc[i] > 0]

    def rank(self, query: str) -> list[int]:
        """Chunks that share at least one term with the query, best first."""
        sc = self.scores(query)
        return [i for i in sorted(range(len(sc)), key=lambda i: (-sc[i], i)) if sc[i] > 0]


RETRIEVERS = "bm25, wordllama, openai:<model>, or hybrid:<dense> (BM25 + dense, fused)"


def build(spec: str, chunks: list[str], embed_url: str | None = None, cache=None):
    """A retriever from its name: see RETRIEVERS."""
    from .embed import Dense, Hybrid, OpenAIEmbedder, wordllama

    def dense(name: str):
        if name == "wordllama":
            return Dense(chunks, wordllama())
        if name.startswith("openai:"):
            if not embed_url:
                raise ValueError("openai:<model> needs --embed-url, e.g. http://localhost:11434/v1")
            return Dense(chunks, OpenAIEmbedder(embed_url, name.split(":", 1)[1], cache))
        raise ValueError(f"unknown retriever {name!r}: use {RETRIEVERS}")

    if spec == "bm25":
        return BM25(chunks)
    if spec.startswith("hybrid:"):
        return Hybrid(BM25(chunks), dense(spec.split(":", 1)[1]))
    return dense(spec)
