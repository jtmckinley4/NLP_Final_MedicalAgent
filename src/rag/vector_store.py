"""Low-level vector store wrapper for ChromaDB access."""

from __future__ import annotations

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None
_collection: Any = None


def get_model() -> SentenceTransformer:
    """Load the embedding model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection() -> Any:
    """Open (or create) the persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def query_medical_kb_raw(query: str, top_k: int = 5) -> dict[str, Any]:
    """Run semantic search and return raw Chroma query payload."""
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()[0]
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
