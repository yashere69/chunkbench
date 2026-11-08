"""chunkbench command line."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import yaml

from . import __version__
from .bench import check_questions, evaluate, load_corpus, load_questions
from .retrieve import RETRIEVERS

DEFAULT = ["fixed:50,0", "fixed:100,50", "recursive:30", "recursive:60", "headings:120"]


def render(rows: list[dict]) -> str:
    head = (
        f"{'strategy':<15}{'retriever':<17}{'lang':<5}{'unit':<10}{'chunks':>7}{'words':>6}"
        f"{'context':>8}{'recall':>8}{'prec':>7}{'IoU':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['strategy']:<15}{r['retriever']:<17}{r['lang']:<5}{r['unit']:<10}"
            f"{r['chunks']:>7}{r['avg_words']:>6}{r['context_words']:>8}"
            f"{r['recall']:>8}{r['precision']:>7}{r['iou']:>7}"
        )
    return "\n".join(lines)


def _utf8_output() -> None:
    """Write UTF-8 even when output is redirected on Windows, where it would otherwise use
    the locale's code page and fail on Devanagari or Hinglish text."""
    for stream in (sys.stdout, sys.stderr):
        enc = (stream.encoding or "").lower().replace("-", "")
        if enc != "utf8" and isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    ap = argparse.ArgumentParser(prog="chunkbench")
    ap.add_argument("--version", action="version", version=f"chunkbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score chunking strategies on a corpus")
    r.add_argument("corpus", type=Path, help="folder of .md/.txt files")
    r.add_argument("--questions", type=Path, help="YAML (default: <corpus>/questions.yaml)")
    r.add_argument("--strategy", action="append", help=f"repeatable (default: {DEFAULT})")
    r.add_argument("--unit", action="append", choices=["words", "chars", "graphemes"])
    r.add_argument("-k", type=int, default=2, help="chunks retrieved per question")
    r.add_argument(
        "--budget",
        type=int,
        help="retrieve best-first chunks up to this many words instead of top-k (fairer "
        "across chunk sizes)",
    )
    r.add_argument("--retriever", action="append", help=f"repeatable: {RETRIEVERS}")
    r.add_argument("--embed-url", help="OpenAI-compatible base URL for openai:<model>")
    r.add_argument(
        "--cache",
        type=Path,
        default=Path(".chunkbench-cache/embeddings.jsonl"),
        help="where embeddings from --embed-url are cached",
    )
    r.add_argument("--json", type=Path)
    c = sub.add_parser("check", help="validate a question file and suggest fixes")
    c.add_argument("corpus", type=Path)
    c.add_argument("--questions", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        try:
            docs = load_corpus(args.corpus)
            problems = check_questions(args.questions or args.corpus / "questions.yaml", docs)
        except (OSError, ValueError, yaml.YAMLError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        for line in problems:
            print(line)
        print(f"{len(problems)} problem(s)" if problems else "all excerpts found")
        return 1 if problems else 0
    try:
        docs = load_corpus(args.corpus)
        questions = load_questions(args.questions or args.corpus / "questions.yaml", docs)
        if args.budget is not None and args.budget < 1:
            raise ValueError("--budget must be a positive number of words")
        rows = evaluate(
            docs,
            questions,
            args.strategy or DEFAULT,
            args.k,
            args.unit or ["words"],
            args.retriever or ["bm25"],
            args.budget,
            args.embed_url,
            args.cache,
        )
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    picked = f"up to {args.budget} words" if args.budget else f"top-{args.k}"
    print(f"{len(docs)} documents, {len(questions)} questions, {picked} per question\n")
    print(render(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
