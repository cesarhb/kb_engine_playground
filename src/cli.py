"""
CLI entrypoint: ingestion, agent cli, or agent server.
Usage:
  python -m src.cli ingestion
  python -m src.cli cli "Your question?"
  python -m src.cli server
"""
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="kb_engine_playground")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingestion", help="Run ingestion pipeline (fetch → parse → chunk → embed → store)")

    cli_parser = subparsers.add_parser("cli", help="Ask the RAG agent a question (CLI)")
    cli_parser.add_argument("question", nargs="+", help="Question to ask")
    cli_parser.add_argument("--collection", default="kb_engine_playground", help="Vector store collection name")

    subparsers.add_parser("server", help="Start the RAG agent server (for LangGraph Studio or API)")

    search_parser = subparsers.add_parser("search", help="Search the vector DB (validate embeddings)")
    search_parser.add_argument("query", nargs="*", help="Search query (optional; if omitted, show count only)")
    search_parser.add_argument("--collection", default="kb_engine_playground", help="Vector store collection name")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results (default 5)")

    args = parser.parse_args()

    # Ensure we're in project root when run as module
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(project_root):
        os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.command == "ingestion":
        from src.ingestion.pipeline import run_pipeline
        run_pipeline()
        return

    if args.command == "cli":
        question = " ".join(args.question)
        if not question:
            print("Provide a question, e.g. cli 'How do I use LangChain?'")
            sys.exit(1)
        from src.agent.graph import run_rag
        answer = run_rag(question, collection_name=args.collection)
        print(answer)
        return

    if args.command == "server":
        from src.agent.server import run_server
        run_server()
        return

    if args.command == "search":
        from src.ingestion.store import get_vector_store
        store = get_vector_store(collection_name=args.collection)
        if not args.query:
            # Count documents in collection (approximate via similarity_search with a generic query)
            try:
                docs = store.similarity_search("architecture", k=1)
                print("Vector store is reachable. Run with a query to search, e.g.: search 'reliability pillar'")
            except Exception as e:
                print("Error connecting to vector store:", e, file=sys.stderr)
                sys.exit(1)
            return
        query = " ".join(args.query)
        docs = store.similarity_search(query, k=args.k)
        print(f"Found {len(docs)} result(s) for '{query}':\n")
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            source = meta.get("source_id") or meta.get("source_url") or meta.get("source") or "—"
            print(f"--- Result {i} (source: {source}) ---")
            print(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
            print()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
