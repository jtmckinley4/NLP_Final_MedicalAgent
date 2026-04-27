"""Stub retrieval tool used before real RAG integration."""

from __future__ import annotations

from pydantic_ai import ModelRetry

from src.models.schemas import RetrievedChunk


def search_medical_kb(query: str, top_k: int = 3) -> list[RetrievedChunk]:
    """Return deterministic mock evidence chunks for a query."""
    cleaned = query.strip()
    if not cleaned:
        raise ModelRetry("Query cannot be empty. Ask a specific medical question.")
    if top_k < 1 or top_k > 5:
        raise ModelRetry("top_k must be between 1 and 5.")

    starter_chunks = [
        RetrievedChunk(
            chunk_id="kb-001",
            source="Mayo Clinic (stub)",
            title="General symptom guidance",
            snippet=(
                "Early triage can include symptom duration, severity, and red flags "
                "such as chest pain, breathing issues, or confusion."
            ),
        ),
        RetrievedChunk(
            chunk_id="kb-002",
            source="CDC (stub)",
            title="When to seek urgent care",
            snippet=(
                "Patients should seek urgent care when warning signs are present "
                "or symptoms worsen despite basic supportive care."
            ),
        ),
        RetrievedChunk(
            chunk_id="kb-003",
            source="WHO (stub)",
            title="Patient education principles",
            snippet=(
                "Medical guidance should include limitations, risk context, and "
                "clear escalation instructions."
            ),
        ),
    ]
    return starter_chunks[:top_k]

