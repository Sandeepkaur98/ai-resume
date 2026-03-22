"""
Optional CLI for ingesting PDFs and querying the in-memory store.

Usage (from repo root, with rag_app on PYTHONPATH via script dir):

    python rag_app/main.py ingest path/to/a.pdf path/to/b.pdf
    python rag_app/main.py search "Python developer with ML experience"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure rag_app/ is importable when run as python rag_app/main.py
_RAG_ROOT = Path(__file__).resolve().parent
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from app.ingest import ingest_pdf_file
from app.llm import get_embedding_model
from app.query import semantic_search
from app.vector_store import InMemoryVectorStore


def _cmd_ingest(args: argparse.Namespace) -> int:
    model = get_embedding_model()
    store = InMemoryVectorStore()
    for p in args.files:
        path = Path(p)
        if not path.is_file():
            print(f"Skip (not a file): {path}", file=sys.stderr)
            continue
        with path.open("rb") as f:
            ingest_pdf_file(store, model, path.name, f)
        print(f"Indexed: {path.name} (total: {len(store.records)})")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    model = get_embedding_model()
    store = InMemoryVectorStore()
    for p in args.files:
        path = Path(p)
        if not path.is_file():
            print(f"Skip: {path}", file=sys.stderr)
            continue
        with path.open("rb") as f:
            ingest_pdf_file(store, model, path.name, f)
    if not store.records:
        print("No PDFs indexed. Provide --files with valid PDF paths.", file=sys.stderr)
        return 1
    results = semantic_search(args.query, store, model, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['filename']}\t{r['score']:.4f}")
        print(f"   {r['text_preview'][:200]}...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume semantic search CLI (local demo).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDF files (prints progress; ephemeral store).")
    p_ingest.add_argument("files", nargs="+", help="PDF file paths")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_search = sub.add_parser("search", help="Load PDFs, then run one query.")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument("--files", nargs="+", required=True, help="PDF paths to index first")
    p_search.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_search.set_defaults(func=_cmd_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
