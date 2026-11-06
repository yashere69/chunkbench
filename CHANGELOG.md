# Changelog

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
