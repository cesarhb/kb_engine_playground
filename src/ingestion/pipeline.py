"""
Ingestion pipeline: fetch → parse → chunk → embed → store.
Run as: python -m src.ingestion.pipeline
"""
import os
import sys
import threading

from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

# Allow running from project root or from container (PYTHONPATH=/app)
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root if os.path.isdir(project_root) else os.getcwd())

from langchain_core.documents import Document

from src.ingestion.loaders import load_all_documents
from src.ingestion.store import (
    EMBED_MODEL_MAX_CHARS,
    get_embedding_model,
    get_language_from_metadata,
    get_text_splitter,
    get_text_splitter_for_language,
    get_vector_store,
)
from langchain_text_splitters import Language


def _chunk_documents_with_language_awareness(docs, chunk_size: int, chunk_overlap: int):
    """Split docs using language-aware splitters for GitHub (e.g. .md, .rst); default for PDFs."""
    from langchain_core.documents import Document

    chunks: list[Document] = []
    # Group by (language or "default") to reuse splitters
    by_lang: dict[Language | str, list] = {}
    for d in docs:
        lang = get_language_from_metadata(d.metadata)
        key = lang if lang is not None else "default"
        by_lang.setdefault(key, []).append(d)
    for key, group in by_lang.items():
        splitter = (
            get_text_splitter_for_language(
                key, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            if key != "default"
            else get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        chunks.extend(splitter.split_documents(group))
    return chunks


def _resplit_oversize_chunks(
    chunks: list[Document], max_chars: int
) -> list[Document]:
    """
    Re-split any chunk that exceeds max_chars into smaller chunks (no truncation).
    RecursiveCharacterTextSplitter can return chunks larger than chunk_size when
    there is no separator (e.g. long lines in GitHub .md/.rst). This step guarantees
    every chunk is <= max_chars so the embedding model never sees overlength input.
    """
    result: list[Document] = []
    for doc in chunks:
        content = doc.page_content
        if len(content) <= max_chars:
            result.append(doc)
            continue
        # Split into max_chars-sized sub-chunks; all content is kept.
        for start in range(0, len(content), max_chars):
            piece = content[start : start + max_chars]
            if piece.strip():
                result.append(Document(page_content=piece, metadata=doc.metadata))
    return result


# Chunks per batch for embed+store; progress is printed after each batch.
EMBED_BATCH_SIZE = 50
HEARTBEAT_INTERVAL_SEC = 15


@contextmanager
def _heartbeat_while(interval_sec: int, message: str):
    """Context manager: print a heartbeat every interval_sec while the block runs."""
    stop = threading.Event()

    def _run():
        while not stop.wait(interval_sec):
            print(f"  [heartbeat] {message}", flush=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


def run_pipeline(
    config_path: str = "config/doc_sources.yaml",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    collection_name: str = "kb_engine_playground",
) -> None:
    """
    Load config, fetch docs, chunk (language-aware for GitHub), re-split any oversize
    chunks, then embed and store. Chunk size is tied to embedding model limit (see store.EMBED_MODEL_MAX_CHARS).
    """
    max_chars = EMBED_MODEL_MAX_CHARS
    size = chunk_size if chunk_size is not None else max_chars
    overlap = chunk_overlap if chunk_overlap is not None else max(50, size // 10)

    print("Ingestion logs and errors appear below (no need to check Ollama container logs).", flush=True)
    print(f"Chunking: size={size} chars, overlap={overlap} (embedding model max={max_chars} chars).", flush=True)
    print("Loading doc sources from", config_path, flush=True)
    with _heartbeat_while(HEARTBEAT_INTERVAL_SEC, "fetching and parsing documents..."):
        docs = load_all_documents(config_path)
    print(f"Loaded {len(docs)} document(s)", flush=True)

    if not docs:
        print("No documents to process. Check config/doc_sources.yaml.")
        return

    chunks = _chunk_documents_with_language_awareness(docs, size, overlap)
    print(f"Split into {len(chunks)} chunks", flush=True)

    # Re-split any chunk still over the limit (split, do not truncate). LangChain can
    # return chunks > chunk_size when there's no separator (e.g. long GitHub lines).
    n_before = len(chunks)
    chunks = _resplit_oversize_chunks(chunks, max_chars)
    if len(chunks) != n_before:
        print(
            f"Re-split oversize chunks: {n_before} → {len(chunks)} (all ≤{max_chars} chars).",
            flush=True,
        )

    # Guarantee: no chunk exceeds embedding model limit (fail fast if invariant broken).
    oversize = [i for i, d in enumerate(chunks) if len(d.page_content) > max_chars]
    if oversize:
        raise AssertionError(
            f"Invariant violated: {len(oversize)} chunk(s) still exceed {max_chars} chars (indices: {oversize[:5]}...). "
            "Re-split logic must keep all chunks ≤ EMBED_MODEL_MAX_CHARS."
        )

    embedding = get_embedding_model()
    vector_store = get_vector_store(
        collection_name=collection_name,
        embedding=embedding,
    )

    total = len(chunks)
    num_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    print(
        f"Embedding and storing in batches of {EMBED_BATCH_SIZE} ({num_batches} batches)...",
        flush=True,
    )
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_num = (i // EMBED_BATCH_SIZE) + 1
        with _heartbeat_while(
            HEARTBEAT_INTERVAL_SEC,
            f"embedding batch {batch_num}/{num_batches} (Ollama + pgvector)...",
        ):
            try:
                vector_store.add_documents(batch)
            except Exception as e:
                print(
                    f"\n[ERROR] Batch {batch_num}/{num_batches} failed (Ollama/embedding or pgvector):",
                    str(e),
                    file=sys.stderr,
                    flush=True,
                )
                raise
        done = min(i + EMBED_BATCH_SIZE, total)
        print(f"  batch {batch_num}/{num_batches} done ({done}/{total} chunks)", flush=True)
    print("Stored embeddings in pgvector.", flush=True)


if __name__ == "__main__":
    run_pipeline()
