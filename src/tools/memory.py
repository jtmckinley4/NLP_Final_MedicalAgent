"""Conversation memory helpers."""

from __future__ import annotations

from src.models.schemas import DraftMedicalAnswer, MemorySummary


def get_default_memory() -> MemorySummary:
    """Create a default memory object for a new chat session."""
    return MemorySummary(patient_context=None, last_topics=[], safety_flags=[])


def update_memory_summary(
    previous_memory: MemorySummary | None,
    user_query: str,
    final_answer: DraftMedicalAnswer,
) -> MemorySummary:
    """Update simple memory summary fields after each answer."""
    memory = previous_memory.model_copy(deep=True) if previous_memory else get_default_memory()

    short_topic = " ".join(user_query.strip().split()[:6]) or "general"
    memory.last_topics.append(short_topic)
    memory.last_topics = memory.last_topics[-5:]

    if "urgent" in final_answer.safety_note.lower() or "emergency" in final_answer.safety_note.lower():
        memory.safety_flags.append("recent_urgent_advice")
        memory.safety_flags = memory.safety_flags[-5:]

    if memory.patient_context is None:
        memory.patient_context = "User is asking educational medical questions."

    return memory

