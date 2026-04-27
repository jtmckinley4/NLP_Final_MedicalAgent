"""Core orchestration pipeline for one user turn."""

from __future__ import annotations

from src.agents.router_agent import run_router
from src.models.schemas import MemorySummary, PipelineResult
from src.tools.memory import get_default_memory, update_memory_summary


def run_turn(user_query: str, previous_memory: MemorySummary | None) -> PipelineResult:
    """Execute one turn: router delegation + memory update."""
    base_memory = previous_memory or get_default_memory()
    router_output = run_router(user_query, base_memory)
    new_memory = update_memory_summary(base_memory, user_query, router_output.answer)
    return PipelineResult(
        route=router_output.route,
        risk_level=router_output.risk_level,
        answer=router_output.answer,
        memory=new_memory,
    )

