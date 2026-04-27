"""Simple evidence validation helper."""

from __future__ import annotations

from src.models.schemas import EvidenceValidationResult, RetrievedChunk


def validate_evidence_support(
    answer_text: str,
    evidence_chunks: list[RetrievedChunk],
) -> EvidenceValidationResult:
    """Perform a lightweight lexical support check."""
    if not evidence_chunks:
        return EvidenceValidationResult(
            supported=False,
            missing_claims=["No evidence chunks were supplied."],
            notes="Validation failed because evidence list was empty.",
        )

    joined_snippets = " ".join(chunk.snippet.lower() for chunk in evidence_chunks)
    missing_claims: list[str] = []
    for sentence in answer_text.split("."):
        fragment = sentence.strip().lower()
        if len(fragment) < 20:
            continue
        words = [w for w in fragment.split() if len(w) > 4]
        if words and not any(w in joined_snippets for w in words[:3]):
            missing_claims.append(sentence.strip())

    supported = len(missing_claims) == 0
    notes = (
        "All major claims appear grounded in retrieved snippets."
        if supported
        else "Some claims did not match retrieval snippets in this simple check."
    )
    return EvidenceValidationResult(
        supported=supported,
        missing_claims=missing_claims,
        notes=notes,
    )

