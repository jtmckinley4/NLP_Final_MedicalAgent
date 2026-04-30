"""Public retrieval tool wrapper for medical RAG queries."""

from __future__ import annotations

from src.models.schemas import RetrievedChunk
from src.rag.vector_store import query_medical_kb_raw

def search_medical_kb(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Search medical KB and return `RetrievedChunk` records."""
    results = query_medical_kb_raw(query=query, top_k=top_k)
    
    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = float(1 / (1 + distance))
        output.append({
            "text": results["documents"][0][i],
            "question": results["metadatas"][0][i].get("question", ""),
            "source": results["metadatas"][0][i].get("source", ""),
            "score": score,
        })
    return output

