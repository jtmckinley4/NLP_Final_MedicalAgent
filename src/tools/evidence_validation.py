"""Deterministic evidence validation tool.

Performs two concrete checks the LLM cannot reliably self-certify:
1. Citation ID integrity — do cited IDs actually exist in the retrieved set?
2. Answer coverage — what fraction of answer sentences have lexical overlap
   with any retrieved chunk?

The safety agent uses these results as grounded signal, not as a pass/fail gate.
"""

from __future__ import annotations

import re

from src.models.schemas import EvidenceValidationResult, RetrievedChunk

# Stopwords to ignore when checking lexical overlap — these are too common
# to be meaningful evidence of grounding
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "not", "as", "if", "when", "than", "also",
    "about", "more", "some", "such", "there", "their", "they", "which",
    "what", "how", "any", "all", "most", "other", "your", "you",
}
_MIN_SENTENCE_LENGTH = 30  # characters — skip very short fragments
_OVERLAP_THRESHOLD = 2     # content words that must appear in chunks to count as grounded


def _content_words(text: str) -> set[str]:
    """Extract lowercase content words, stripping punctuation and stopwords."""
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, filtering out very short fragments."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LENGTH]


def validate_evidence_support(
    answer_text: str,
    evidence_chunks: list[RetrievedChunk],
    cited_ids: list[str] | None = None,
) -> EvidenceValidationResult:
    """Run deterministic grounding checks on a draft answer.

    Performs two checks:
    - Citation ID integrity: verifies that every cited ID exists in the
      retrieved chunk set (catches hallucinated citation IDs).
    - Sentence coverage: for each substantive sentence in the answer, checks
      whether at least `_OVERLAP_THRESHOLD` content words appear anywhere in
      the retrieved snippets. Reports the fraction of sentences that pass.

    Args:
        answer_text: The direct_answer text from the draft answer.
        evidence_chunks: The RetrievedChunk objects available as evidence.
        cited_ids: Optional list of citation_id values from the draft's
            citations. Used for the ID integrity check.
    """
    notes_parts: list[str] = []
    missing_claims: list[str] = []

    # ------------------------------------------------------------------
    # Check 1: Citation ID integrity
    # ------------------------------------------------------------------
    available_ids = {chunk.chunk_id for chunk in evidence_chunks}
    bad_citation_ids: list[str] = []

    if cited_ids:
        for cid in cited_ids:
            if cid not in available_ids:
                bad_citation_ids.append(cid)
        if bad_citation_ids:
            notes_parts.append(
                f"Citation ID check FAILED: {len(bad_citation_ids)} cited ID(s) not "
                f"found in retrieved set: {bad_citation_ids}."
            )
            missing_claims.extend(
                f"Cited ID not in evidence: {cid}" for cid in bad_citation_ids
            )
        else:
            notes_parts.append(
                f"Citation ID check PASSED: all {len(cited_ids)} cited ID(s) exist "
                "in the retrieved set."
            )
    else:
        notes_parts.append("Citation ID check SKIPPED: no cited_ids provided.")

    # ------------------------------------------------------------------
    # Check 2: Sentence-level lexical coverage
    # ------------------------------------------------------------------
    if not evidence_chunks:
        notes_parts.append("Coverage check SKIPPED: no evidence chunks provided.")
        return EvidenceValidationResult(
            supported=False,
            missing_claims=["No evidence chunks were supplied."],
            notes=" | ".join(notes_parts),
        )

    # Build a single set of all content words across all chunks
    chunk_vocab: set[str] = set()
    for chunk in evidence_chunks:
        chunk_vocab.update(_content_words(chunk.snippet))

    sentences = _split_sentences(answer_text)
    if not sentences:
        notes_parts.append("Coverage check SKIPPED: no substantive sentences found.")
        supported = len(bad_citation_ids) == 0
        return EvidenceValidationResult(
            supported=supported,
            missing_claims=missing_claims,
            notes=" | ".join(notes_parts),
        )

    grounded_count = 0
    ungrounded: list[str] = []
    for sentence in sentences:
        words = _content_words(sentence)
        overlap = words & chunk_vocab
        if len(overlap) >= _OVERLAP_THRESHOLD:
            grounded_count += 1
        else:
            ungrounded.append(sentence)

    coverage_ratio = grounded_count / len(sentences)
    notes_parts.append(
        f"Coverage check: {grounded_count}/{len(sentences)} sentences "
        f"({coverage_ratio:.0%}) have >= {_OVERLAP_THRESHOLD} content words "
        "overlapping with retrieved evidence."
    )

    if ungrounded:
        missing_claims.extend(ungrounded)
        notes_parts.append(
            f"{len(ungrounded)} sentence(s) had insufficient lexical overlap with evidence."
        )

    # Overall: pass if no bad citation IDs and majority of sentences are grounded
    supported = len(bad_citation_ids) == 0 and coverage_ratio >= 0.5

    return EvidenceValidationResult(
        supported=supported,
        missing_claims=missing_claims,
        notes=" | ".join(notes_parts),
    )