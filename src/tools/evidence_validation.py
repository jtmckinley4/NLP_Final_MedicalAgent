"""NLP-enhanced evidence validation tool.

Performs three concrete checks the LLM cannot reliably self-certify:
1. Citation ID integrity  — do cited IDs exist in the retrieved set?
2. TF-IDF cosine similarity — how semantically similar is each answer sentence
   to the evidence body?
3. Named entity overlap  — do the entities named in the answer (drugs,
   conditions, substances) actually appear in the retrieved evidence?

The safety agent calls this tool and uses the results as grounded signal when
deciding whether to approve, revise, or flag the draft answer.

Dependencies (both already required by the project):
    scikit-learn   — TfidfVectorizer, cosine_similarity
    spacy          — NER pipeline (en_core_web_sm)

Install the spaCy model once:
    uv run python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.schemas import EvidenceValidationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_SENTENCE_LENGTH = 30   # characters — skip very short fragments
_TFIDF_THRESHOLD = 0.10     # minimum cosine similarity to count as grounded
_SPACY_MODEL = "en_core_web_sm"

# spaCy's en_core_web_sm entity labels most relevant to medical text.
# PERSON is excluded — patient names are not medical claims.
_RELEVANT_ENTITY_LABELS = {
    "ORG",      # pharmaceutical companies, health organisations
    "PRODUCT",  # named drugs and medical products
    "GPE",      # locations relevant to disease prevalence
    "LAW",      # regulations, drug approvals
    "NORP",     # nationalities/groups relevant to epidemiology
    "FAC",      # medical facilities
}


# ---------------------------------------------------------------------------
# Lazy model loader — load once, reuse across calls
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_nlp():
    """Load the spaCy NER model once and cache it."""
    return spacy.load(_SPACY_MODEL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, filtering out very short fragments."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LENGTH]


def _extract_entities(text: str) -> set[str]:
    """Return lowercase named entity strings from the relevant label set."""
    nlp = _load_nlp()
    doc = nlp(text)
    return {
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ in _RELEVANT_ENTITY_LABELS
    }


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------
def validate_evidence_support(
    answer_text: str,
    evidence_text: str,
    cited_ids: list[str] | None = None,
    available_ids: list[str] | None = None,
) -> EvidenceValidationResult:
    """Run NLP-enhanced grounding checks on a draft answer.

    Performs three checks:

    1. Citation ID integrity — verifies that every cited ID exists in the
       set of retrieved chunk IDs, catching hallucinated citation references.

    2. TF-IDF cosine similarity — for each substantive sentence in the answer,
       computes cosine similarity against the full evidence body using TF-IDF
       vectors. Reports the fraction of sentences that meet the similarity
       threshold. More principled than raw word-overlap counting.

    3. Named entity overlap — extracts named entities from both the answer and
       the evidence using spaCy's en_core_web_sm model. Flags any entities in
       the answer that do not appear in the evidence, which may indicate
       unsupported claims about specific drugs, organisations, or conditions.

    The safety agent uses the returned EvidenceValidationResult to decide
    whether to approve the draft, revise specific claims, or lower confidence.
    It should NOT treat this as a binary pass/fail gate.

    Args:
        answer_text: The direct_answer text from the draft answer.
        evidence_text: All retrieved evidence snippets joined into a single
            string. The LLM constructs this by concatenating the evidence
            chunks visible in its context.
        cited_ids: Citation IDs from the draft answer's citations list.
            Used for the ID integrity check. Pass an empty list if none.
        available_ids: The chunk_ids of all retrieved evidence chunks.
            Used for the ID integrity check. Pass an empty list if none.
    """
    notes_parts: list[str] = []
    missing_claims: list[str] = []

    _cited = list(cited_ids or [])
    _available = set(available_ids or [])

    # ------------------------------------------------------------------
    # Check 1: Citation ID integrity
    # ------------------------------------------------------------------
    if _cited and _available:
        bad_citation_ids = [cid for cid in _cited if cid not in _available]
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
                f"Citation ID check PASSED: all {len(_cited)} cited ID(s) exist "
                "in the retrieved set."
            )
    else:
        bad_citation_ids = []
        notes_parts.append("Citation ID check SKIPPED: cited_ids or available_ids not provided.")

    # ------------------------------------------------------------------
    # Check 2: TF-IDF cosine similarity
    # ------------------------------------------------------------------
    sentences = _split_sentences(answer_text)
    coverage_ratio = 0.0
    ungrounded: list[str] = []

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    corpus = [evidence_text] + sentences
    tfidf_matrix = vectorizer.fit_transform(corpus)
    evidence_vec = tfidf_matrix[0]
    sentence_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(sentence_vecs, evidence_vec).flatten()

    grounded_count = 0
    for sentence, score in zip(sentences, similarities):
        if score >= _TFIDF_THRESHOLD:
            grounded_count += 1
        else:
            ungrounded.append(f"[sim={score:.2f}] {sentence}")

    coverage_ratio = grounded_count / len(sentences) if sentences else 0.0
    notes_parts.append(
        f"TF-IDF similarity check: {grounded_count}/{len(sentences)} sentences "
        f"({coverage_ratio:.0%}) meet the similarity threshold "
        f"(>= {_TFIDF_THRESHOLD:.2f}) against the evidence body."
    )
    if ungrounded:
        missing_claims.extend(ungrounded)
        notes_parts.append(
            f"{len(ungrounded)} sentence(s) had low TF-IDF similarity to evidence."
        )

    # ------------------------------------------------------------------
    # Check 3: Named entity overlap (spaCy en_core_web_sm)
    # ------------------------------------------------------------------
    answer_entities = _extract_entities(answer_text)
    evidence_entities = _extract_entities(evidence_text)
    unsupported_entities = sorted(answer_entities - evidence_entities)
    supported_entities = answer_entities & evidence_entities

    notes_parts.append(
        f"Entity overlap check: {len(supported_entities)}/{len(answer_entities)} "
        f"answer entities found in evidence. "
        f"Supported: {sorted(supported_entities) or 'none'}. "
        f"Not in evidence: {unsupported_entities or 'none'}."
    )
    if unsupported_entities:
        missing_claims.extend(
            f"Named entity not found in evidence: '{e}'"
            for e in unsupported_entities
        )

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    tfidf_ok = coverage_ratio >= 0.5 if sentences else True
    entities_ok = len(unsupported_entities) == 0
    citation_ok = len(bad_citation_ids) == 0

    supported = citation_ok and tfidf_ok and entities_ok

    return EvidenceValidationResult(
        supported=supported,
        missing_claims=missing_claims,
        notes=" | ".join(notes_parts),
    )