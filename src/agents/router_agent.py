"""Router agent — top-level control agent that delegates to specialists."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from src.agents.general_composer_agent import compose_general_answer
from src.agents.medication_specialist_agent import compose_medication_answer
from src.config import get_model
from src.models.schemas import (
    DraftMedicalAnswer,
    MemorySummary,
    RouterAgentOutput,
    SpecialistResult,
)


@dataclass
class RouterDeps:
    """Dependencies injected into the router agent at runtime.

    Using deps (rather than baking memory into the prompt only) means each
    tool function can forward memory to the specialist it calls, without
    exposing memory as an LLM-visible tool parameter.
    """
    memory: MemorySummary | None

ROUTER_INSTRUCTIONS = """
You are the top-level routing and orchestration agent for a medical question-answering
system. Your job is to decide what to do with each user query and carry it out.

ROUTING RULES — pick exactly one route:

1. "clarify" — The query is too vague or uses unresolvable references (e.g., "is it
   dangerous?", "what should I take?"). Do NOT call a specialist. Instead, generate a
   targeted clarification question and return a safe placeholder answer.

2. "refuse_or_defer" — The query asks for diagnosis, personalized treatment decisions,
   or something clearly unsafe (e.g., "what diagnosis do I have?", "how much extra
   can I safely take?"). Do NOT call a specialist. Return a polite refusal and suggest
   they speak to a clinician.

3. "general_medical" — The query is an answerable general medical question about
   symptoms, conditions, prevention, or non-medication treatments. Call the
   delegate_general tool.

4. "medication" — The query is specifically about a medication: side effects, warnings,
   interactions, dosage information. Call the delegate_medication tool.

RISK LEVEL — always set one:
- "high": possible urgent symptoms (chest pain, difficulty breathing, overdose, stroke
  signs), emergency scenarios, or requests that could lead to harm if mishandled.
- "medium": treatment, medication, side effects, or prevention questions.
- "low": general informational questions with no urgency signals.

TOOL USAGE:
- Before calling any specialist tool, rewrite the user_query to resolve any
  pronoun or reference using the memory context. For example, if the user asked
  "what are its side effects?" and memory shows the topic is amoxicillin, pass
  "what are the side effects of amoxicillin?" to the tool — not the original.
- For "general_medical": call delegate_general(user_query) — this returns a
  SpecialistResult with an answer and evidence.
- For "medication": call delegate_medication(user_query) — same return type.
- For "clarify" or "refuse_or_defer": call generate_fallback_answer(route,
  clarification_question) to get the appropriate safe answer.
- Do NOT call multiple specialist tools for one query. Pick the best route and
  call exactly one tool.

OUTPUT — RouterAgentOutput:
- route: the chosen route string
- risk_level: the assigned risk level
- rationale: 1-2 sentences explaining why you chose this route and risk level
- answer: the DraftMedicalAnswer from whichever tool you called
- evidence: the evidence list from the SpecialistResult (empty for clarify/refuse)
"""


# ---------------------------------------------------------------------------
# Tool functions registered on the router agent
# ---------------------------------------------------------------------------
# Each tool takes RunContext[RouterDeps] as its first argument. PydanticAI
# injects this automatically and keeps it invisible to the LLM — the model
# only sees the remaining parameters when deciding how to call the tool.

def delegate_general(ctx: RunContext[RouterDeps], user_query: str) -> SpecialistResult:
    """Call the general medical composer agent to answer a non-medication question.

    Use this for questions about symptoms, conditions, prevention, and general
    treatments. Returns a SpecialistResult with a draft answer and evidence chunks.
    """
    return compose_general_answer(user_query, memory=ctx.deps.memory)


def delegate_medication(ctx: RunContext[RouterDeps], user_query: str) -> SpecialistResult:
    """Call the medication specialist agent to answer a medication-specific question.

    Use this for questions about drug side effects, warnings, interactions, and
    usage information. Returns a SpecialistResult with a draft answer and evidence.
    """
    return compose_medication_answer(user_query, memory=ctx.deps.memory)


def generate_fallback_answer(
    ctx: RunContext[RouterDeps],
    route: str,
    clarification_question: str | None = None,
) -> DraftMedicalAnswer:
    """Generate a safe fallback answer for clarify or refuse_or_defer routes.

    Args:
        route: Either "clarify" or "refuse_or_defer".
        clarification_question: The specific question to ask the user (for "clarify"
            route). Leave None for "refuse_or_defer".

    Returns a DraftMedicalAnswer suitable for returning directly to the user.
    """
    if route == "clarify":
        question = clarification_question or "Could you share more details about your situation?"
        return DraftMedicalAnswer(
            direct_answer="I need a bit more information to answer safely and accurately.",
            evidence_summary="No evidence retrieved — clarification is needed before answering.",
            citations=[],
            safety_note=(
                "If you are experiencing severe or worsening symptoms, please seek "
                "medical care rather than waiting."
            ),
            confidence=0.2,
            follow_up_question=question,
        )
    # refuse_or_defer
    return DraftMedicalAnswer(
        direct_answer=(
            "I'm not able to provide guidance for this type of request. "
            "This system is designed for general medical information only — "
            "it cannot diagnose conditions or make personalized treatment decisions. "
            "Please consult a licensed clinician or pharmacist."
        ),
        evidence_summary="Request is outside the scope of this system.",
        citations=[],
        safety_note=(
            "If this is an emergency, call emergency services immediately. "
            "For personalized medical advice, please see a qualified healthcare provider."
        ),
        confidence=0.1,
        follow_up_question=None,
    )


# ---------------------------------------------------------------------------
# Router agent
# ---------------------------------------------------------------------------

router_agent = Agent(
    model=get_model(),
    instructions=ROUTER_INSTRUCTIONS,
    output_type=RouterAgentOutput,
    deps_type=RouterDeps,
    tools=[delegate_general, delegate_medication, generate_fallback_answer],
    retries=2,
)


def _build_prompt(user_query: str, memory: MemorySummary | None) -> str:
    parts = [f"User question: {user_query}"]
    if memory:
        if memory.last_topics:
            parts.append(
                f"Recent conversation topics: {', '.join(memory.last_topics[-3:])}"
            )
        if memory.patient_context:
            parts.append(f"Session context: {memory.patient_context}")
        if memory.safety_flags:
            parts.append(
                f"Active safety flags (from prior turns): {', '.join(memory.safety_flags)}"
            )
        parts.append(
            "Use the above memory to resolve any follow-up references in the question "
            "(e.g., 'that medication', 'it', 'the condition you mentioned')."
        )
    return "\n".join(parts)


def run_router(user_query: str, memory: MemorySummary | None) -> RouterAgentOutput:
    """Route the query and delegate to the appropriate specialist."""
    prompt = _build_prompt(user_query, memory)
    result = router_agent.run_sync(prompt, deps=RouterDeps(memory=memory))
    return result.output