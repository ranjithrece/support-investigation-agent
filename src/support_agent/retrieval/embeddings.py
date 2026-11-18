"""Ollama embeddings + Chroma vector store (persistent, cosine)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chromadb
from langchain_ollama import OllamaEmbeddings

from support_agent.config import get_settings

COLLECTION = "tickets"
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

_emb: OllamaEmbeddings | None = None


def embedder() -> OllamaEmbeddings:
    global _emb
    if _emb is None:
        s = get_settings()
        _emb = OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url)
    return _emb


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return embedder().embed_documents([_DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    return embedder().embed_query(_QUERY_PREFIX + text)


def collection(path: str | None = None) -> Any:
    client = chromadb.PersistentClient(path=path or str(get_settings().chroma_dir))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
