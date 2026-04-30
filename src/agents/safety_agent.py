"""Safety verification agent — final review pass before returning to the user."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import DraftMedicalAnswer, RetrievedChunk, SafetyCheckResult
from src.tools.evidence_validation import validate_evidence_support

SAFETY_INSTRUCTIONS = """
You are the final safety reviewer for a medical question-answering system.
Review the draft answer and either approve it or revise it before it reaches
the user.

## REVIEW CHECKLIST

1. GROUNDING — Call validate_evidence_support. If major claims are unsupported
   by the provided evidence, soften or remove them. If no evidence was provided,
   note that claims are unverified and lower confidence accordingly.

2. SCOPE — Does the answer avoid diagnosis and personalized treatment plans?
   If not, revise to make it general and educational.

3. ESCALATION — For high-risk queries, does the answer include clear escalation
   language ("call emergency services", "seek immediate care")? Add it if missing.

4. OVERCLAIMING — If confidence > 0.7 but citations are absent or evidence is
   weak or off-topic, lower confidence and soften wording.

## OUTPUT REQUIREMENTS
- approved: True only if no significant issues were found.
- revised_answer: ALWAYS populate — either the original or your corrected
  version. Never leave this null.
- reviewer_notes: Short summary of what you checked and what you changed.
- escalation_needed: True if the query involved urgent symptoms, overdose,
  or emergency scenarios.
"""


def _build_prompt(
    user_query: str,
    draft: DraftMedicalAnswer,
    evidence: list[RetrievedChunk],
    high_risk: bool,
) -> str:
    evidence_text = (
        "\n".join(f"  [{c.chunk_id}] {c.source} — {c.snippet}" for c in evidence)
        if evidence
        else "  (no evidence chunks provided)"
    )
    return (
        f"User question: {user_query}\n\n"
        f"High-risk flag: {high_risk}\n\n"
        f"Draft answer:\n{draft.model_dump_json(indent=2)}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Call validate_evidence_support with answer_text=draft.direct_answer, "
        "evidence_chunks=the chunks above, and cited_ids=the citation_id values "
        "from the draft citations. Then complete your review."
    )


safety_agent = Agent(
    model=get_model("claude-sonnet-4-6"),
    instructions=SAFETY_INSTRUCTIONS,
    output_type=SafetyCheckResult,
    tools=[validate_evidence_support],
    retries=2,
)


def verify_answer(
    user_query: str,
    draft: DraftMedicalAnswer,
    evidence: list[RetrievedChunk],
    high_risk: bool,
) -> SafetyCheckResult:
    """Run the safety review and revision pass on a specialist draft."""
    prompt = _build_prompt(user_query, draft, evidence, high_risk)
    result = safety_agent.run_sync(prompt)
    return result.output