"""Router agent decides which path should handle the question."""

from __future__ import annotations

from pydantic_ai import Agent

from src.agents.general_composer_agent import compose_general_answer
from src.agents.medication_specialist_agent import compose_medication_answer
from src.agents.safety_agent import verify_answer
from src.config import get_model
from src.models.schemas import (
    DraftMedicalAnswer,
    MemorySummary,
    RetrievedChunk,
    RouterAgentOutput,
    SafetyCheckResult,
    SpecialistResult,
)

ROUTER_INSTRUCTIONS = (
    "You are the top-level routing agent for a medical QA assistant. "
    "You must choose one route: clarify, refuse_or_defer, general_medical, or medication. "
    "For general_medical or medication routes, call the corresponding specialist tool and then "
    "call the safety review tool before returning final output. "
    "For clarify/refuse routes, call fallback tool and return a safe answer. "
    "Always return RouterAgentOutput."
)


def _clarification_answer(question: str) -> DraftMedicalAnswer:
    return DraftMedicalAnswer(
        direct_answer="I need a little more detail to answer safely.",
        evidence_summary="No evidence retrieved because clarification is required first.",
        citations=[],
        safety_note="Share symptom timing, severity, and relevant medications.",
        confidence=0.2,
        follow_up_question=question,
    )


def _defer_answer() -> DraftMedicalAnswer:
    return DraftMedicalAnswer(
        direct_answer=(
            "I cannot safely provide detailed guidance for this scenario. "
            "Please contact a licensed clinician or emergency services now."
        ),
        evidence_summary="High-risk signal triggered safe deferral path.",
        citations=[],
        safety_note="If this is an emergency, call emergency services immediately.",
        confidence=0.15,
        follow_up_question="Would you like a concise list of urgent warning signs?",
    )


def delegate_general(user_query: str) -> SpecialistResult:
    """Delegate to the general specialist agent."""
    return compose_general_answer(user_query)


def delegate_medication(user_query: str) -> SpecialistResult:
    """Delegate to the medication specialist agent."""
    return compose_medication_answer(user_query)


def delegate_safety(
    user_query: str,
    draft: DraftMedicalAnswer,
    evidence: list[RetrievedChunk],
    risk_level: str,
) -> SafetyCheckResult:
    """Run safety verifier on delegated specialist output."""
    return verify_answer(
        user_query=user_query,
        draft=draft,
        evidence=evidence,
        high_risk=risk_level == "high",
    )


def fallback_answer(route: str, clarification_question: str | None = None) -> DraftMedicalAnswer:
    """Generate safe fallback answers for clarify/refuse routes."""
    if route == "clarify":
        return _clarification_answer(clarification_question or "Can you share more details?")
    return _defer_answer()


router_agent = Agent(
    model=get_model(),
    instructions=ROUTER_INSTRUCTIONS,
    output_type=RouterAgentOutput,
    tools=[delegate_general, delegate_medication, delegate_safety, fallback_answer],
    retries=2,
)


def run_router(user_query: str, memory: MemorySummary | None) -> RouterAgentOutput:
    """Route query and delegate to specialist/safety tools as needed."""
    result = router_agent.run_sync(
        f"User question: {user_query}\nMemory: {memory.model_dump_json() if memory else 'None'}"
    )
    return result.output

