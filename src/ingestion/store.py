"""
Store embeddings in Postgres via pgvector using LangChain's PGVector.
"""
import os
from typing import Any

# -----------------------------------------------------------------------------
# Embedding model context limit (validated)
# -----------------------------------------------------------------------------
# mxbai-embed-large-v1: max sequence length 512 tokens (Hugging Face model card,
# https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1; mixedbread.ai docs).
# Chunking must keep every chunk under this (in chars, conservatively).
# ~1 char/token is safe for mixed tokenizers; 500 chars leaves headroom.
EMBED_MODEL_MAX_TOKENS = 512
EMBED_MODEL_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "500"))

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import PGVector

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def get_embedding_model() -> Embeddings:
    """Return embeddings model from env: ollama or openai."""
    provider = (os.getenv("EMBEDDING_PROVIDER") or "ollama").strip().lower()
    model = os.getenv("EMBEDDING_MODEL") or "mxbai-embed-large"

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return OllamaEmbeddings(
            model=model,
            base_url=base_url,
            keep_alive=300,  # 5 minutes (seconds); avoid keeping model in VRAM indefinitely
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_vector_store(
    connection_string: str | None = None,
    collection_name: str = "kb_engine_playground",
    embedding: Embeddings | None = None,
) -> VectorStore:
    """Build PGVector store. Uses DATABASE_URL if connection_string not provided."""
    conn = connection_string or os.getenv("DATABASE_URL")
    if not conn:
        raise ValueError("DATABASE_URL or connection_string required")
    emb = embedding or get_embedding_model()
    return PGVector(
        connection_string=conn,
        embedding_function=emb,
        collection_name=collection_name,
        use_jsonb=True,
        pre_delete_collection=False,
    )


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Default splitter for generic text (e.g. PDFs). Uses EMBED_MODEL_MAX_CHARS by default."""
    size = chunk_size if chunk_size is not None else EMBED_MODEL_MAX_CHARS
    overlap = chunk_overlap if chunk_overlap is not None else max(50, size // 10)
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


# File extension -> Language for GitHub/code docs (langchain's built-in separators)
_EXT_TO_LANGUAGE: dict[str, Language] = {
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".rst": Language.RST,
}
# .txt and unknown extensions use default splitter (None)


def get_language_from_metadata(metadata: dict) -> Language | None:
    """Infer Language from doc metadata (e.g. source URL or file_extension from GitHub loader)."""
    ext = (metadata.get("file_extension") or "").strip().lower()
    if not ext:
        source = (metadata.get("source") or "").strip()
        if "." in source:
            ext = "." + source.rsplit(".", 1)[-1].split("/")[0].split("?")[0].lower()
    return _EXT_TO_LANGUAGE.get(ext) if ext else None


def get_text_splitter_for_language(
    language: Language | None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Language-aware splitter for GitHub/code docs; respects embedding model char limit."""
    size = chunk_size if chunk_size is not None else EMBED_MODEL_MAX_CHARS
    overlap = chunk_overlap if chunk_overlap is not None else max(50, size // 10)
    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=size,
            chunk_overlap=overlap,
        )
    return get_text_splitter(chunk_size=size, chunk_overlap=overlap)
