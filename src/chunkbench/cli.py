"""chunkbench command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .bench import evaluate, load_corpus, load_questions

DEFAULT = ["fixed:50,0", "fixed:100,50", "recursive:30", "recursive:60", "headings:120"]


def render(rows: list[dict]) -> str:
    head = (
        f"{'strategy':<16}{'lang':<6}{'unit':<11}{'chunks':>7}{'words':>7}"
        f"{'recall':>8}{'prec':>7}{'IoU':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<16}{r['lang']:<6}{r['unit']:<11}{r['chunks']:>7}"
            f"{r['avg_words']:>7}{r['recall']:>8}{r['precision']:>7}{r['iou']:>7}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chunkbench")
    ap.add_argument("--version", action="version", version=f"chunkbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score chunking strategies on a corpus")
    r.add_argument("corpus", type=Path, help="folder of .md/.txt files")
    r.add_argument("--questions", type=Path, help="YAML (default: <corpus>/questions.yaml)")
    r.add_argument("--strategy", action="append", help=f"repeatable (default: {DEFAULT})")
    r.add_argument("--unit", action="append", choices=["words", "chars", "graphemes"])
    r.add_argument("-k", type=int, default=2, help="chunks retrieved per question")
    r.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    try:
        docs = load_corpus(args.corpus)
        questions = load_questions(args.questions or args.corpus / "questions.yaml", docs)
        rows = evaluate(docs, questions, args.strategy or DEFAULT, args.k, args.unit or ["words"])
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{len(docs)} documents, {len(questions)} questions, top-{args.k} BM25\n")
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
