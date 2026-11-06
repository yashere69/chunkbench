from pathlib import Path

import pytest

from chunkbench.bench import evaluate, load_corpus, load_questions
from chunkbench.cli import main
from chunkbench.metrics import covered, score
from chunkbench.retrieve import BM25, tokenize
from chunkbench.units import words

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


def test_metric_hand_computed():
    text = "a b c d e f g h"
    units = {"d": words(text)}
    rel = [("d", 2, 5)]  # "b c"
    # retrieve "a b c d" (4 units) and, overlapping, "c d" (2 units)
    got = [("d", 0, 7), ("d", 4, 7)]
    s = score(units, rel, got)
    assert s.recall == 1.0
    assert s.precision == pytest.approx(2 / 6)  # repeats count against precision
    assert s.iou == pytest.approx(2 / (2 + 6 - 2))


def test_covered_counts_partial_overlap():
    u = words("alpha beta gamma")
    assert covered(u, 3, 8) == {0, 1}  # 'ha be'


def test_bm25_ranks_the_matching_chunk_first():
    idx = BM25(["shipping is free above 1499", "wash mul in cold water", "refunds in five days"])
    assert idx.top("how do I wash mul", 1) == [1]
    assert idx.top("zzz", 2) == []


def test_tokenizer_keeps_devanagari_words_whole():
    assert tokenize("किताब की क़ीमत") == ["किताब", "की", "क़ीमत"]


def test_questions_must_quote_the_document(tmp_path):
    docs = {"a.md": "hello world"}
    q = tmp_path / "q.yaml"
    q.write_text("- {q: hi, doc: a.md, excerpts: [goodbye]}\n")
    with pytest.raises(ValueError, match="excerpt not in a.md"):
        load_questions(q, docs)


def test_sample_corpus_end_to_end():
    docs = load_corpus(CORPUS)
    qs = load_questions(CORPUS / "questions.yaml", docs)
    rows = evaluate(docs, qs, ["recursive:30", "fixed:100,50"], 2, ["words"])
    by = {(r["strategy"], r["lang"]): r for r in rows}
    # smaller chunks trade recall for precision
    assert by[("recursive:30", "en")]["iou"] > by[("fixed:100,50", "en")]["iou"]
    assert by[("fixed:100,50", "en")]["recall"] >= by[("recursive:30", "en")]["recall"]


def test_cli(capsys):
    assert main(["run", str(CORPUS)]) == 0
    assert "25 questions" in capsys.readouterr().out
    assert main(["run", "/nonexistent"]) == 2


def test_excerpt_with_different_line_breaks_is_found(tmp_path):
    docs = {"a.md": "Shipping is free on prepaid\norders above 1499. Below that 79."}
    q = tmp_path / "q.yaml"
    q.write_text(
        "- {q: 'free?', doc: a.md, excerpts: ['Shipping is free on prepaid orders above 1499.']}\n"
    )
    (question,) = load_questions(q, docs)
    s, e = question.excerpts[0]
    assert docs["a.md"][s:e] == "Shipping is free on prepaid\norders above 1499."


def test_check_lists_every_problem_with_a_suggestion(tmp_path, capsys):
    import shutil

    for md in CORPUS.glob("*.md"):
        shutil.copy(md, tmp_path)
    (tmp_path / "questions.yaml").write_text(
        "- {q: otp, doc: shipping-and-returns.md, excerpts: ['We never ask for your card number"
        " or OTP.']}\n- {q: x, doc: nope.md, excerpts: [a]}\n- {q: y}\n"
    )
    assert main(["check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "closest sentence: 'We never ask for your card number, PIN or OTP.'" in out
    assert "unknown doc 'nope.md'" in out and "needs q, doc and excerpts" in out
    assert "3 problem(s)" in out
    assert main(["check", str(CORPUS)]) == 0
