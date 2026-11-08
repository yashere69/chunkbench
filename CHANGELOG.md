# Changelog

## 0.3.0

- Pluggable retrievers: `--retriever bm25|wordllama|openai:<model>|hybrid:<dense>`
  (repeatable). WordLlama runs offline from its PyPI wheel
  (`pip install "chunkbench[wordllama]"`). `openai:<model>` calls any OpenAI-compatible
  `/v1/embeddings` endpoint (`--embed-url`) and caches vectors on disk. `hybrid:` fuses BM25
  with a dense ranking by reciprocal rank.
- `--budget N`: compare strategies at an equal number of retrieved words instead of top-k.
- Output gains `retriever` and `context_words` columns.
- `tools/compare_retrievers.py` writes `results/retrievers.md`.

## 0.2.0

- Excerpts match across whitespace differences (copied from rendered pages)
- `chunkbench check`: every broken excerpt at once, with the closest sentence
- Malformed question files give a clear error

## 0.1.0

- Strategies: fixed windows with overlap, recursive (danda-aware), markdown headings
- Units: words, code points, grapheme clusters with Indic conjuncts
- BM25 retrieval with a Devanagari-safe tokenizer
- Token-level recall, precision and IoU after Chroma's method
- Sample bilingual corpus with 25 verbatim-labelled questions
