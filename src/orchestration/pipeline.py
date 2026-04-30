"""Core orchestration pipeline for one user turn."""

from __future__ import annotations

from src.agents.router_agent import run_router
from src.agents.safety_agent import verify_answer
from src.models.schemas import MemorySummary, PipelineResult
from src.tools.memory import get_default_memory, update_memory_summary


def run_turn(user_query: str, previous_memory: MemorySummary | None) -> PipelineResult:
    """Execute one full turn: route → specialist → safety review → memory update.

    Args:
        user_query: The raw user question for this turn.
        previous_memory: Memory summary from the prior turn, or None for a new session.

    Returns:
        PipelineResult containing the final answer, route, risk level, and updated memory.
    """
    base_memory = previous_memory or get_default_memory()

    # Step 1: Router decides route and delegates to the appropriate specialist.
    # The router receives memory so it can resolve follow-up references.
    router_output = run_router(user_query, base_memory)

    # Step 2: Safety verification agent reviews the specialist draft.
    safety_result = verify_answer(
        user_query=user_query,
        draft=router_output.answer,
        evidence=router_output.evidence,
        high_risk=router_output.risk_level == "high",
    )

    # Step 3: Use the safety-revised answer as the final output.
    final_answer = safety_result.revised_answer

    # Step 4: Update memory from this turn for use in the next turn.
    new_memory = update_memory_summary(base_memory, user_query, final_answer)

    return PipelineResult(
        route=router_output.route,
        risk_level=router_output.risk_level,
        answer=final_answer,
        memory=new_memory,
    )