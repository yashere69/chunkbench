"""Dense retrieval, kept optional: an embedder turns texts into vectors, and `Dense` ranks
chunks by cosine similarity to the question.

Two embedders ship:

- `wordllama`: WordLlama's static embeddings (MIT, weights inside the PyPI wheel), so a
  dense baseline runs offline with no server. `pip install "chunkbench[wordllama]"`.
- `openai:<model>`: any OpenAI-compatible `/v1/embeddings` endpoint (`--embed-url`), which
  covers Ollama, llama.cpp's server, vLLM, Text Embeddings Inference and hosted APIs. The
  key, if one is needed, comes from CHUNKBENCH_API_KEY. Vectors are cached on disk by
  model and text, so re-running a benchmark doesn't re-embed the corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

Embedder = Callable[[list[str]], list[list[float]]]


def wordllama() -> Embedder:
    try:
        import wordllama as wl_pkg
        from wordllama import WordLlama
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ValueError(
            'the wordllama retriever needs: pip install "chunkbench[wordllama]"'
        ) from e
    pkg = Path(wl_pkg.__file__).parent
    bundled = (pkg / "weights").is_dir() and (pkg / "tokenizers").is_dir()
    model = WordLlama.load(cache_dir=pkg, disable_download=True) if bundled else WordLlama.load()

    def embed(texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in model.embed(texts)]

    return embed


class OpenAIEmbedder:
    """Calls POST {url}/embeddings in batches and caches every vector it gets back."""

    def __init__(self, url: str, model: str, cache: Path | None = None, batch: int = 64):
        self.url, self.model, self.batch = url.rstrip("/"), model, batch
        self.cache_path = cache
        self.cache: dict[str, list[float]] = {}
        if cache and cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["vector"]

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()

    def _request(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("CHUNKBENCH_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.url}/embeddings", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)["data"]
        return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        fresh = []
        for i in range(0, len(missing), self.batch):
            part = missing[i : i + self.batch]
            for text, vec in zip(part, self._request(part), strict=True):
                self.cache[self._key(text)] = vec
                fresh.append({"key": self._key(text), "vector": vec})
        if fresh and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(row) + "\n" for row in fresh)
        return [self.cache[self._key(t)] for t in texts]


def _unit(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Dense:
    """Rank chunks by cosine similarity between question and chunk embeddings."""

    def __init__(self, chunks: list[str], embed: Embedder):
        self.embed = embed
        self.vecs = [_unit(v) for v in embed(chunks)] if chunks else []

    def rank(self, query: str) -> list[int]:
        (q,) = self.embed([query])
        q = _unit(q)
        sims = [sum(a * b for a, b in zip(q, v, strict=True)) for v in self.vecs]
        return sorted(range(len(sims)), key=lambda i: (-sims[i], i))


class Hybrid:
    """Reciprocal rank fusion of two rankings (k=60, as in Cormack et al., 2009)."""

    def __init__(self, *retrievers: object, k: int = 60):
        self.parts, self.k = retrievers, k

    def rank(self, query: str) -> list[int]:
        score: dict[int, float] = {}
        for part in self.parts:
            for pos, i in enumerate(part.rank(query)):  # type: ignore[attr-defined]
                score[i] = score.get(i, 0.0) + 1 / (self.k + pos + 1)
        return sorted(score, key=lambda i: (-score[i], i))
