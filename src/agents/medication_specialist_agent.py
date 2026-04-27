"""Medication specialist agent with stubbed drug lookup support."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import SpecialistResult
from src.tools.drug_lookup import lookup_drug_info_openfda
from src.tools.rag_stub import search_medical_kb

MEDICATION_INSTRUCTIONS = (
    "You are a medication-focused assistant. "
    "Use available tool data for drug lookup and evidence retrieval. "
    "Always call lookup_drug_info_openfda and search_medical_kb before final output. "
    "Never prescribe or set a diagnosis. Include safety and escalation language."
)

medication_agent = Agent(
    model=get_model(),
    instructions=MEDICATION_INSTRUCTIONS,
    output_type=SpecialistResult,
    tools=[lookup_drug_info_openfda, search_medical_kb],
    retries=2,
)


def _infer_drug_name(user_query: str) -> str:
    lower = user_query.lower()
    if "ibuprofen" in lower or "advil" in lower:
        return "ibuprofen"
    if "acetaminophen" in lower or "paracetamol" in lower or "tylenol" in lower:
        return "acetaminophen"
    tokens = [token.strip(".,!?") for token in lower.split()]
    return tokens[-1] if tokens else "unknown"


def compose_medication_answer(user_query: str) -> SpecialistResult:
    """Compose medication-aware answer with agent-driven tool calls."""
    drug_name_hint = _infer_drug_name(user_query)
    prompt = (
        f"User question: {user_query}\n"
        f"Possible medication mention: {drug_name_hint}"
    )
    result = medication_agent.run_sync(prompt)
    return result.output

