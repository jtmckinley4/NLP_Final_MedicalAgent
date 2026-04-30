"""Conversation memory management.

Uses a minimal PydanticAI agent (no tools, no retries) to extract a structured
MemorySummary from each completed turn. This gives us proper semantic extraction
(resolving references, identifying entities, detecting safety signals) without
the overhead of a full reasoning agent.
"""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import DraftMedicalAnswer, MemorySummary

_EXTRACTION_INSTRUCTIONS = """
You extract structured conversation memory from medical Q&A turns.
Return only what is explicitly present in the conversation — do not infer
or speculate beyond what was said.

## FIELDS

last_topics: A list of short phrases (2-5 words each) naming the medical
  subjects discussed. Include the new turn's topic and carry forward any
  still-relevant prior topics. Keep the most recent 5. Examples:
  "ibuprofen side effects", "type 2 diabetes symptoms", "blood pressure treatment"

patient_context: A single sentence capturing any persistent user context that
  would help interpret future follow-up questions — e.g. medications mentioned,
  conditions referenced, or stated sensitivities. If nothing new emerged and
  prior context is still valid, carry it forward unchanged. If none exists,
  use null.

safety_flags: A list of short flag strings for any high-risk signals that
  should persist into future turns. Use these exact strings when applicable:
  - "recent_urgent_advice" — system gave escalation/emergency language
  - "overdose_mention" — user mentioned overdose or excessive dosing
  - "high_risk_interaction" — a dangerous drug interaction was discussed
  Keep only flags still relevant. Remove flags if the topic has clearly changed.
  Empty list if no flags apply.
"""


def _build_extraction_prompt(
    user_query: str,
    final_answer: DraftMedicalAnswer,
    previous_memory: MemorySummary | None,
) -> str:
    prior = (
        previous_memory.model_dump_json(indent=2)
        if previous_memory
        else "null (first turn)"
    )
    return (
        f"Previous memory:\n{prior}\n\n"
        f"New user question: {user_query}\n\n"
        f"System answer summary: {final_answer.direct_answer[:400]}\n"
        f"Safety note: {final_answer.safety_note}\n"
        f"Confidence: {final_answer.confidence}\n\n"
        "Extract updated MemorySummary for the next turn."
    )


# Minimal agent — no tools, no retries, pure structured extraction.
# Haiku is sufficient here; this is extraction not reasoning.
_memory_agent = Agent(
    model=get_model("claude-haiku-4-5"),
    instructions=_EXTRACTION_INSTRUCTIONS,
    output_type=MemorySummary,
)


def get_default_memory() -> MemorySummary:
    """Return a blank memory object for the start of a new session."""
    return MemorySummary(patient_context=None, last_topics=[], safety_flags=[])


def update_memory_summary(
    previous_memory: MemorySummary | None,
    user_query: str,
    final_answer: DraftMedicalAnswer,
) -> MemorySummary:
    """Extract and return an updated MemorySummary after a completed turn.

    Falls back to the previous memory (or default) if extraction fails,
    so a memory update error never crashes the pipeline.

    Args:
        previous_memory: The MemorySummary from the prior turn, or None.
        user_query: The user's question for this turn.
        final_answer: The final DraftMedicalAnswer returned to the user.
    """
    prompt = _build_extraction_prompt(user_query, final_answer, previous_memory)
    try:
        result = _memory_agent.run_sync(prompt)
        return result.output
    except Exception:
        # Memory update failure is non-fatal — return prior state or default
        return previous_memory or get_default_memory()