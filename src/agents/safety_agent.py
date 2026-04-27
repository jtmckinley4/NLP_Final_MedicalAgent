"""Safety verification agent."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import DraftMedicalAnswer, RetrievedChunk, SafetyCheckResult
from src.tools.evidence_validation import validate_evidence_support

SAFETY_INSTRUCTIONS = (
    "You are the final safety reviewer. "
    "Ensure the answer is educational, grounded, and includes escalation language when risk is high. "
    "Revise unsafe claims."
)

safety_agent = Agent(
    model=get_model(),
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
    """Run safety verification and revision pass."""
    result = safety_agent.run_sync(
        f"User question: {user_query}\n"
        f"Draft answer: {draft.model_dump()}\n"
        f"Evidence: {[chunk.model_dump() for chunk in evidence]}\n"
        f"High risk: {high_risk}\n"
        "Return structured safety review and revised answer."
    )
    return result.output

