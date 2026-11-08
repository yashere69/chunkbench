"""Retrievers, budgets and the embedding client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chunkbench.bench import select
from chunkbench.cli import main
from chunkbench.embed import Dense, Hybrid, OpenAIEmbedder
from chunkbench.retrieve import BM25, build

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def letters(texts):
    """A toy embedder: letter counts. Deterministic, so ranking can be checked exactly."""
    return [[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]


def test_budget_takes_best_first_chunks_until_the_next_would_not_fit():
    sizes = [50, 30, 40, 10]
    assert select([0, 1, 2, 3], sizes, k=2, budget=None) == [0, 1]
    assert select([0, 1, 2, 3], sizes, k=2, budget=85) == [0, 1]
    assert select([2, 0, 1], sizes, k=2, budget=95) == [2, 0]
    assert select([0, 1], sizes, k=2, budget=10) == [0]  # always at least one chunk


def test_dense_and_hybrid_rankings():
    chunks = ["zzz zzz", "abc abc", "kurta size chart"]
    dense = Dense(chunks, letters)
    assert dense.rank("abc")[0] == 1
    fused = Hybrid(BM25(chunks), dense)
    assert fused.rank("size chart")[0] == 2
    assert sorted(fused.rank("size chart")) == [0, 1, 2]


class _Embeddings(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Embeddings.calls.append({"body": body, "auth": self.headers.get("Authorization")})
        data = [{"index": i, "embedding": [len(t), 1.0]} for i, t in enumerate(body["input"])]
        out = json.dumps({"data": list(reversed(data))}).encode()  # order must not matter
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_openai_compatible_client_batches_sorts_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKBENCH_API_KEY", "k-123")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Embeddings)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    cache = tmp_path / "cache.jsonl"
    try:
        emb = OpenAIEmbedder(url, "nomic-embed-text", cache, batch=2)
        assert emb(["a", "bbb", "cc"]) == [[1, 1.0], [3, 1.0], [2, 1.0]]
        assert len(_Embeddings.calls) == 2 and _Embeddings.calls[0]["auth"] == "Bearer k-123"
        assert _Embeddings.calls[0]["body"]["model"] == "nomic-embed-text"
        again = OpenAIEmbedder(url, "nomic-embed-text", cache)  # a new run reads the cache
        assert again(["bbb"]) == [[3, 1.0]] and len(_Embeddings.calls) == 2
    finally:
        server.shutdown()


def test_bad_retriever_choices_are_errors(capsys):
    corpus = str(EXAMPLES / "corpus")
    assert main(["run", corpus, "--retriever", "openai:x"]) == 2
    assert "--embed-url" in capsys.readouterr().err
    assert main(["run", corpus, "--retriever", "tfidf"]) == 2
    assert "unknown retriever" in capsys.readouterr().err
    assert main(["run", corpus, "--budget", "0"]) == 2
    with pytest.raises(ValueError, match="unknown"):
        build("dense", ["x"])


def test_wordllama_runs_offline_when_installed():
    pytest.importorskip("wordllama")
    dense = build("wordllama", ["the return window is seven days", "colour may bleed"])
    assert dense.rank("how many days do I have to send it back")[0] == 0


def test_unreachable_embedding_server_is_a_clean_error(tmp_path, capsys):
    corpus = str(Path(__file__).resolve().parent.parent / "examples" / "corpus")
    args = ["run", corpus, "--retriever", "openai:x", "--embed-url", "http://127.0.0.1:9/v1"]
    assert main([*args, "--cache", str(tmp_path / "c")]) == 2
    assert "can't reach http://127.0.0.1:9/v1/embeddings" in capsys.readouterr().err
