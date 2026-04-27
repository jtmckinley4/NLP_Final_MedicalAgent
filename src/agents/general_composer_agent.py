"""General medical specialist agent."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import SpecialistResult
from src.tools.rag_stub import search_medical_kb

GENERAL_INSTRUCTIONS = (
    "You provide educational, non-diagnostic medical answers. "
    "Use your tools to retrieve evidence and return structured output. "
    "Always call search_medical_kb before finalizing your response. "
    "Be clear about uncertainty and include safety boundaries."
)

general_agent = Agent(
    model=get_model(),
    instructions=GENERAL_INSTRUCTIONS,
    output_type=SpecialistResult,
    tools=[search_medical_kb],
    retries=2,
)


def compose_general_answer(user_query: str) -> SpecialistResult:
    """Compose a general medical answer with agent-driven tool calls."""
    result = general_agent.run_sync(f"User question: {user_query}")
    return result.output

