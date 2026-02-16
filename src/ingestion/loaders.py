"""
Config-driven document loading using LangChain.
Supports N sources from config: url, urls, pdf_url, pdf_urls, github_repo.
"""
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve

import yaml
from langchain_community.document_loaders import GithubFileLoader, WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document


def load_config(config_path: str | Path = "config/doc_sources.yaml") -> list[dict[str, Any]]:
    """Load doc_sources.yaml and return the list of source configs."""
    path = Path(config_path)
    if not path.is_absolute():
        # Prefer project root (parent of src/)
        for root in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
            candidate = root / path
            if candidate.exists():
                path = candidate
                break
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def _load_pdf_from_url(url: str) -> list[Document]:
    """Download PDF from URL to a temp file and load with PyPDFLoader."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        urlretrieve(url, tmp_path)
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


def _make_github_file_filter(
    extensions: tuple[str, ...] = (".md", ".rst", ".txt"),
    include_paths: list[str] | None = None,
) -> Callable[[str], bool]:
    """Return a file_filter callable for GithubFileLoader."""

    def _filter(path: str) -> bool:
        if not any(path.endswith(ext) for ext in extensions):
            return False
        if not include_paths:
            return True
        for p in include_paths:
            if path == p:
                return True
            if p.endswith("/") or "/" in p.rstrip("/"):
                if path.startswith(p.rstrip("/") + "/"):
                    return True
            elif path.startswith(p):
                return True  # e.g. p "README" matches "README.md"
        return False

    return _filter


def load_documents_for_source(source: dict[str, Any]) -> list[Document]:
    """
    Load documents for one source config.
    Adds source_id to metadata for each document.
    """
    source_id = source.get("id", "unknown")
    loader_type = (source.get("type") or "url").strip().lower()

    if loader_type == "url":
        url = source.get("url")
        if not url:
            raise ValueError(f"Source {source_id}: type=url requires 'url'")
        loader = WebBaseLoader(url)
        docs = loader.load()
    elif loader_type == "urls":
        urls = source.get("urls") or []
        if not urls:
            raise ValueError(f"Source {source_id}: type=urls requires 'urls' list")
        loader = WebBaseLoader(urls)
        docs = loader.load()
    elif loader_type == "pdf_url":
        url = source.get("url")
        if not url:
            raise ValueError(f"Source {source_id}: type=pdf_url requires 'url'")
        docs = _load_pdf_from_url(url)
        for d in docs:
            d.metadata["source_url"] = url
    elif loader_type == "pdf_urls":
        urls = source.get("urls") or []
        if not urls:
            raise ValueError(f"Source {source_id}: type=pdf_urls requires 'urls' list")
        docs = []
        for url in urls:
            part = _load_pdf_from_url(url)
            for d in part:
                d.metadata["source_url"] = url
            docs.extend(part)
    elif loader_type == "github_repo":
        repo = source.get("repo")
        if not repo:
            raise ValueError(f"Source {source_id}: type=github_repo requires 'repo' (owner/name)")
        branch = source.get("branch", "main")
        include_paths = source.get("include_paths") or []
        extensions = tuple(source.get("file_extensions") or [".md", ".rst", ".txt"])
        file_filter = _make_github_file_filter(extensions=extensions, include_paths=include_paths or None)
        access_token = source.get("access_token") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        loader = GithubFileLoader(
            repo=repo,
            branch=branch,
            access_token=access_token,
            file_filter=file_filter,
        )
        docs = list(loader.lazy_load())
        # Set file_extension from source URL for language-aware splitting in pipeline
        for d in docs:
            src = (d.metadata.get("source") or "").strip()
            if "." in src:
                ext = "." + src.rsplit(".", 1)[-1].split("/")[0].split("?")[0].lower()
                d.metadata["file_extension"] = ext
    else:
        raise ValueError(f"Source {source_id}: unsupported type={loader_type}")

    for d in docs:
        d.metadata["source_id"] = source_id
    return docs


def load_all_documents(config_path: str | Path = "config/doc_sources.yaml") -> list[Document]:
    """Load all documents from all configured sources."""
    sources = load_config(config_path)
    all_docs: list[Document] = []
    for source in sources:
        docs = load_documents_for_source(source)
        all_docs.extend(docs)
    return all_docs
