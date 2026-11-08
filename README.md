# chunkbench

Chunking choices for RAG, scored the way Chroma's research scored them: by the exact text
spans that answer each question, not by chunk ids. chunkbench chunks a folder of markdown
with several strategies and retrieves for each labelled question. Retrieval is BM25 by
default, a local embedding model, any OpenAI-compatible embedding endpoint, or a fusion of
BM25 and embeddings. It reports how much of the right text came back (recall) and how much
padding came with it (precision, IoU). It can compare strategies at an equal word budget
rather than an equal number of chunks, and it counts Hindi in graphemes, so a chunker that
splits a conjunct is caught.

![chunkbench comparing five strategies with BM25 and a BM25+WordLlama fusion at a 100-word budget, on English and Hindi questions](docs/demo.png)

```bash
pip install git+https://github.com/yashere69/chunkbench
chunkbench run examples/corpus
chunkbench run my-docs/ --questions my-questions.yaml --strategy recursive:80 --strategy headings:200
chunkbench run my-docs/ --budget 300 --retriever bm25 --retriever hybrid:openai:nomic-embed-text \
    --embed-url http://localhost:11434/v1     # Ollama; any OpenAI-compatible server works
```

## The metric

The scoring follows Chroma's [chunking evaluation](https://research.trychroma.com/evaluating-chunking)
(Smith & Troynikov, 2024). A question is labelled with the *excerpts* that answer it, quoted
verbatim from the document, not with a chunk id. That's what makes strategies with
different boundaries comparable. For each question:

- **recall**: share of the excerpt's units that appear in the retrieved chunks
- **precision**: excerpt units retrieved ÷ all units retrieved, counting overlapped text
  every time it comes back
- **IoU**: excerpt units retrieved ÷ (excerpt + retrieved − excerpt units retrieved)

Recall alone always favours big chunks: retrieve the whole document and recall is 100%.
IoU is what tells you how much irrelevant text you're paying for in the prompt.

## What the sample corpus shows

`examples/corpus` holds three short policy documents for a fictional store, two in English
and one in Hindi, with 25 questions. Top-2 retrieval, units counted in words:

| Strategy | Avg words/chunk | Recall (en / hi) | IoU (en / hi) |
|---|---|---|---|
| fixed:50,0 | 45 | 77.9 / 100 | 15.1 / 15.5 |
| fixed:100,50 | 95 | 100 / 100 | 9.1 / 7.6 |
| recursive:30 | 22 | 68.6 / 85.7 | 26.7 / 28.8 |
| recursive:60 | 40 | 94.4 / 85.7 | 19.9 / 15.0 |
| headings:120 | 47 | 94.4 / 100 | 16.8 / 17.8 |

Full output: [results/sample.txt](results/sample.txt). On these documents, splitting on
headings gives nearly the recall of 100-word overlapping windows at about twice the IoU,
because the documents are already sectioned by topic. Overlap is the worst deal here: it
buys a few points of recall and pays for them with repeated text. Your documents will
differ, and that's the reason to run it on them.

The corpus is small, its documents and questions were written for this repository, and with
only 7 Hindi questions one question moves the Hindi numbers by 14 points. Treat this as a
worked example, not a finding about chunking in general.

### Equal budgets and other retrievers

Top-2 hides a confound: two 95-word chunks hand the reader twice as much text as two
45-word chunks, so big chunks win recall partly by reading more. `--budget 100` takes
best-first chunks until 100 words. Under it, `fixed:100,50`'s English recall falls from 100
to 88.9, and `recursive:30` rises from 68.6 to 79.7, because it can now take four chunks.
The strategy ranking changes once each one reads the same amount.

[results/retrievers.md](results/retrievers.md) (`python tools/compare_retrievers.py`)
repeats the table with three retrievers: BM25, WordLlama and their fusion. WordLlama is a
static embedding model whose weights ship in its PyPI wheel, so it runs offline. On the
English questions it beats BM25 on small chunks (recall 81.9 against 68.6 for
`recursive:30` at top-2). On the Hindi questions it collapses (28.6 against 85.7), because
it was trained on English. Reciprocal-rank fusion keeps most of both. A modern
multilingual embedding model would likely do better on Hindi than any of these three. Run
one through `--embed-url` before trusting a number from this table.

## Units, and Hindi

Precision and IoU depend on what you count. Whitespace words, code points, and grapheme
clusters (what a reader sees as one character, so क्ष counts as one and not three)
give different totals in Devanagari. `--unit graphemes` and `--unit chars` let you check
whether that changes your ranking. On this corpus it doesn't: the unit moves IoU by under a
point, while strategy and size move it by 20. That's worth knowing, because it means
word-level numbers are safe here. For scripts written without spaces (Thai, Khmer),
word-level counting breaks, and the grapheme unit is the one to use.

Two Hindi details needed explicit handling:

- The recursive splitter treats the danda (।) as a sentence end.
- Python's `\w` doesn't match Devanagari vowel signs, so a naive tokenizer cuts
  किताब into pieces and BM25 never matches it. Both the BM25 tokenizer and the grapheme
  unit handle combining marks explicitly.

## Strategies

| Spec | What it does |
|---|---|
| `fixed:N,O` | windows of N words, O words of overlap |
| `recursive:N` | split on paragraph, line, sentence (incl. ।), clause, then word, and merge pieces up to N words |
| `headings:N` | one chunk per markdown section, recursive inside sections longer than N |

Sizes are in whitespace words for every strategy, so `recursive:60` and `fixed:60,0` are
the same size. Chunks are character spans into the original text rather than copied
strings, which is what lets the metrics intersect them exactly with the excerpts.

## Question file

```yaml
- q: Can I return a dupatta bought on sale?
  doc: shipping-and-returns.md
  excerpts: ["Some things cannot be returned: dupattas bought on sale, ..."]
- q: मुफ़्त शिपिंग कितने के ऑर्डर पर है?
  doc: niti-hindi.md
  lang: hi
  excerpts: ["₹1,499 से ऊपर के प्रीपेड ऑर्डर पर शिपिंग मुफ़्त है।"]
```

Every excerpt must appear in its document. Differences in whitespace are allowed, since
an excerpt copied from a rendered page has spaces where the source has line breaks.
Anything else is an error, which catches labels drifting out of date when a document is
edited. `chunkbench check my-docs/` lists every broken excerpt at once, with the closest
sentence in the document, so you can fix a question file in one pass:

```
$ chunkbench check my-docs/
question 2: excerpt not in shipping-and-returns.md: 'We never ask for your card number or OTP.'; closest sentence: 'We never ask for your card number, PIN or OTP.'
1 problem(s)
```

## Related work and limits

- [chunkr's benchmark PR](https://github.com/d1pankarmedhi/chunkr/pull/6) runs Chroma's
  own five-corpus, 472-question benchmark across popular chunking libraries. Use that to
  compare libraries on a standard set; use this to compare settings on your own documents.
- Other tools share the name and part of the job.
  [chunk-bench](https://pypi.org/project/chunk-bench/) (PyPI) ships similar chunkers and
  scores retrieval by `relevant_terms` keywords, with recall, precision, MRR and F1, but no
  span-level IoU. [chunkbench-rag](https://pypi.org/project/chunkbench-rag/) is a harness
  where you bring your own chunker and embedder, with recall@k and a regression gate.
  chunkbench is the one with Chroma-style span metrics, equal-budget comparison, question
  files checked against the documents (`chunkbench check`), and grapheme-aware Hindi.
- The embedding client is plain HTTP. Rate limits are the server's business, and every
  vector is cached in `.chunkbench-cache/`, so a crashed run resumes where it stopped.
- No semantic or LLM-based chunkers.

Stable: the question-file format, the metrics and the `run`/`check` commands. Experimental:
the retriever names and the budget rule (the chunk that would overflow is dropped, never
truncated).

MIT licensed.
