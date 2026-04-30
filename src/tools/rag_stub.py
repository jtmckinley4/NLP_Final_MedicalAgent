"""Public retrieval tool wrapper for medical RAG queries."""

from __future__ import annotations

from src.models.schemas import RetrievedChunk
from src.rag.vector_store import query_medical_kb_raw

def search_medical_kb(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Search medical KB and return `RetrievedChunk` records."""
    results = query_medical_kb_raw(query=query, top_k=top_k)
    ids = results.get("ids", [[]])
    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])

    row_ids = ids[0] if ids else []
    row_docs = documents[0] if documents else []
    row_meta = metadatas[0] if metadatas else []

    output: list[RetrievedChunk] = []
    for idx, (doc, meta) in enumerate(zip(row_docs, row_meta)):
        chunk_id = row_ids[idx] if idx < len(row_ids) else f"chunk_{idx}"
        metadata = meta or {}
        output.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                source=str(metadata.get("source", "medical_kb")),
                title=str(metadata.get("question", "Medical knowledge base")),
                snippet=str(doc),
            )
        )
    return output

