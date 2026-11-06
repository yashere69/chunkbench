"""BM25 over chunks. No embeddings, so results are reproducible and need no API key.

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
