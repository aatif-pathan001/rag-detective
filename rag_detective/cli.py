"""
Command-Line Interface (CLI) for RAG Detective.
Enables terminal-based document ingestion, hybrid searching, and investigation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rag_detective.config import default_config
from rag_detective.engine import RAGDetectiveEngine
from rag_detective.models import InvestigationRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_detective.cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-detective",
        description="RAG Detective Backend CLI: Ingestion, Hybrid Search, and Investigation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document (PDF, Markdown, TXT)")
    ingest_parser.add_argument("file_path", type=str, help="Path to document file to index")

    # Command: query / investigate
    query_parser = subparsers.add_parser("query", help="Run full RAG investigation query")
    query_parser.add_argument("question", type=str, help="Investigation question")
    query_parser.add_argument("--top-k", type=int, default=3, help="Top evidence chunks (default: 3)")
    query_parser.add_argument("--alpha", type=float, default=0.7, help="Sparse weight for RRF (default: 0.7)")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Direct hybrid / dense / sparse search")
    search_parser.add_argument("term", type=str, help="Search query")
    search_parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid", help="Search mode")
    search_parser.add_argument("--top-k", type=int, default=3, help="Results to return (default: 3)")

    # Command: stats
    subparsers.add_parser("stats", help="Show system statistics and indexed chunk count")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST backend")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        try:
            import uvicorn
            print(f"Starting RAG Detective REST Backend on http://{args.host}:{args.port} ...")
            ui_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
            print(f"RAG Detective web UI: http://{ui_host}:{args.port}/")
            uvicorn.run("rag_detective.api.app:app", host=args.host, port=args.port, reload=args.reload)
        except ImportError:
            print("Error: uvicorn is not installed. Install with `pip install uvicorn fastapi`.")
            sys.exit(1)
        return

    # Initialize Engine for local commands
    engine = RAGDetectiveEngine()

    if args.command == "ingest":
        path = Path(args.file_path)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        print(f"Ingesting '{path.name}'...")
        res = engine.ingest_file(path)
        print(f"Success! Indexed {res.total_chunks} chunks.")
        print(f"Total corpus chunks: {res.metadata.get('corpus_total_chunks', res.total_chunks)}")

    elif args.command == "query":
        req = InvestigationRequest(
            user_query=args.question,
            top_k=args.top_k,
            alpha=args.alpha,
        )
        print(f"\n[RAG Detective] Investigating lead: '{args.question}'...")
        res = engine.investigate(req)

        print("\n" + "=" * 60)
        print("INVESTIGATION REPORT")
        print("=" * 60)
        print(f"Answer:\n{res.answer}\n")
        print("-" * 60)
        print(f"Evidence Chunks ({len(res.evidence)}):")
        for i, doc in enumerate(res.evidence, start=1):
            print(f"  [{i}] ID: {doc.chunk_id} (RRF Score: {doc.score})")
            print(f"      {doc.content[:150]}...")
        print("=" * 60)

    elif args.command == "search":
        print(f"Searching for '{args.term}' (mode: {args.mode}, top_k: {args.top_k})...")
        results = engine.search(args.term, top_k=args.top_k, mode=args.mode)
        print(f"\nFound {len(results)} matches:")
        for r in results:
            print(f"  Rank {r.rank} [{r.engine} score: {r.score}] - ID: {r.chunk_id}")
            print(f"    {r.content[:120]}...\n")

    elif args.command == "stats":
        stats = engine.get_stats()
        print("\nRAG Detective System Stats:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
