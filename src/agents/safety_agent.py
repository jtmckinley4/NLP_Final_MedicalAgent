"""Safety verification agent — final review pass before returning to the user."""

from __future__ import annotations

from pydantic_ai import Agent

from src.config import get_model
from src.models.schemas import DraftMedicalAnswer, RetrievedChunk, SafetyCheckResult
from src.tools.evidence_validation import validate_evidence_support

SAFETY_INSTRUCTIONS = """
You are the final safety reviewer for a medical question-answering system.
Review the draft answer and either approve it or revise it before it reaches
the user.

## STEP 1 — CALL validate_evidence_support FIRST (required)

Before forming any opinion on the draft, call validate_evidence_support with:
  - answer_text: the draft's direct_answer field (copy it verbatim)
  - evidence_text: ALL retrieved evidence snippets joined into one string,
    separated by newlines. You can see the snippets in the prompt below —
    concatenate them exactly as shown.
  - cited_ids: the list of citation_id values from the draft's citations
    (empty list if there are none)
  - available_ids: the list of chunk_id values from the retrieved evidence
    (empty list if there are none)

Use the tool's returned report — TF-IDF similarity scores, entity overlap,
and citation ID checks — as your primary grounding signal.

## STEP 2 — REVIEW CHECKLIST

1. GROUNDING — If the tool reports low TF-IDF similarity or unsupported medical
   entities, soften or qualify those specific claims. Do NOT remove citations
   whose IDs the tool confirmed exist in the retrieved set.

2. SCOPE — Does the answer avoid diagnosis and personalized treatment plans?
   If not, revise to make it general and educational.

3. ESCALATION — For high-risk queries, does the answer include clear escalation
   language ("call emergency services", "seek immediate care")? Add it if missing.

4. OVERCLAIMING — If confidence > 0.7 but the tool reports weak TF-IDF coverage
   or unsupported entities, lower confidence to match the evidence quality.

## OUTPUT REQUIREMENTS
- approved: True only if no significant issues were found.
- revised_answer: ALWAYS populate — either the original or your corrected version.
- reviewer_notes: Short summary of what the tool reported and what you changed.
- escalation_needed: True if the query involved urgent symptoms, overdose,
  or emergency scenarios.

## CITATION OUTPUT REQUIREMENT (CRITICAL)
- When producing revised_answer, you MUST include the citations field.
- Preserve all citations whose IDs the tool confirmed as valid.
- Remove only citations the tool explicitly flagged as invalid.
- Do NOT drop the citations field.
- Do NOT return an empty citations list unless the draft originally had none or all were invalid.
- The revised_answer MUST contain: direct_answer, evidence_summary, citations, safety_note, confidence, and follow_up_question (if present).
"""


def _build_prompt(
    user_query: str,
    draft: DraftMedicalAnswer,
    evidence: list[RetrievedChunk],
    high_risk: bool,
) -> str:
    # Present evidence so the LLM can construct the joined string argument
    # and also extract available_ids for the citation integrity check.
    evidence_lines = (
        "\n".join(
            f"[chunk_id={c.chunk_id}] source={c.source} | {c.snippet}"
            for c in evidence
        )
        if evidence
        else "(no evidence chunks provided)"
    )
    available_ids = [c.chunk_id for c in evidence]
    cited_ids = [c.citation_id for c in draft.citations] if draft.citations else []

    return (
        f"User question: {user_query}\n\n"
        f"High-risk flag: {high_risk}\n\n"
        f"Draft answer:\n{draft.model_dump_json(indent=2)}\n\n"
        f"Retrieved evidence chunks (join these snippets for evidence_text):\n"
        f"{evidence_lines}\n\n"
        f"Available chunk IDs (pass as available_ids): {available_ids}\n"
        f"Draft citation IDs (pass as cited_ids): {cited_ids}\n\n"
        "Call validate_evidence_support now using the values above, then complete "
        "your review following the checklist in your instructions."
    )


safety_agent = Agent(
    model=get_model("claude-sonnet-4-6"),
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
    """Run the safety review and revision pass on a specialist draft.

    The safety agent calls validate_evidence_support as a tool, passing the
    evidence as a plain joined string and the chunk/citation IDs as lists —
    all values the LLM can construct directly from the prompt context.
    """
    prompt = _build_prompt(user_query, draft, evidence, high_risk)
    result = safety_agent.run_sync(prompt)
    return result.output