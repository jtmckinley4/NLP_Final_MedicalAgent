"""General medical specialist agent."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import MemorySummary, SpecialistResult
from src.tools.rag_stub import search_medical_kb

GENERAL_INSTRUCTIONS = """
You are a general medical information specialist. Answer non-medication medical
questions using ONLY the evidence returned by search_medical_kb.

## YOUR ONLY SOURCE OF TRUTH IS THE TOOL
Call search_medical_kb first. Whatever it returns is your evidence — treat it
as the complete record available to you. Do not supplement it with your own
medical knowledge, even if you think the chunks are incomplete or off-topic.

## IF THE EVIDENCE DOESN'T COVER THE QUESTION
Say so honestly in direct_answer. Example: "The retrieved evidence does not
specifically address this question. Based on the available information, I can
only say: [what the chunks do say]." Then cite those chunks anyway and set
confidence below 0.4.

STRICT RULES:
- Never make claims beyond what the retrieved evidence supports.
- Never give a medical diagnosis or suggest a treatment.
- Never give personal medical advice.
- Always include a safety_note reminding users to consult a pharmacist or clinician.

## OUTPUT FIELDS
- direct_answer: Grounded in the retrieved chunks only. If coverage is weak,
  say so explicitly.
- evidence_summary: What the chunks covered and how relevant they were.
- citations: One entry per chunk used. citation_id = chunk_id, claim = the
  specific snippet text that supports your answer.
- safety_note: Always include. Add "seek immediate medical care" if the
  question involves potentially urgent symptoms.
- confidence: 0.7+ only if the chunks directly answer the question. 0.3 or
  below if coverage is weak or off-topic.
"""


def _build_prompt(user_query: str, memory: MemorySummary | None) -> str:
    parts = [f"User question: {user_query}"]
    if memory:
        if memory.last_topics:
            parts.append(f"Recent topics in this conversation: {', '.join(memory.last_topics)}")
        if memory.patient_context:
            parts.append(f"Session context: {memory.patient_context}")
        if memory.safety_flags:
            parts.append(f"Active safety flags: {', '.join(memory.safety_flags)}")
    return "\n".join(parts)


general_agent = Agent(
    model=get_model("claude-sonnet-4-6"),
    instructions=GENERAL_INSTRUCTIONS,
    output_type=SpecialistResult,
    tools=[search_medical_kb],
    retries=2,
)


def compose_general_answer(
    user_query: str,
    memory: MemorySummary | None = None,
) -> SpecialistResult:
    """Compose a general medical answer grounded in retrieved evidence."""
    prompt = _build_prompt(user_query, memory)
    result = general_agent.run_sync(prompt)
    return result.output