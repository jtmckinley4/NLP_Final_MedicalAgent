"""CLI entrypoint for the application skeleton."""

from __future__ import annotations

import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import logfire
from src.models.schemas import MemorySummary, PipelineResult
from src.orchestration.pipeline import run_turn


def _configure_runtime_output() -> None:
    """Suppress runtime warning noise from dependency stack."""
    warnings.filterwarnings("ignore")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    try:
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except Exception:
        # Keep startup robust if transformers isn't available.
        pass


def _print_result(result: PipelineResult) -> None:
    answer = result.answer
    print("\n--- Response ---")
    print(f"Route: {result.route} (risk: {result.risk_level})")
    print(f"Direct answer: {answer.direct_answer}")
    print(f"Evidence summary: {answer.evidence_summary}")
    if answer.citations:
        print("Citations:")
        for citation in answer.citations:
            print(f"  - [{citation.citation_id}] {citation.source}: {citation.claim}")
    else:
        print("Citations: none")
    print(f"Safety note: {answer.safety_note}")
    print(f"Confidence: {answer.confidence:.2f}")
    if answer.follow_up_question:
        print(f"Follow-up question: {answer.follow_up_question}")
    print("------------------------\n")


def main() -> None:
    """Run interactive CLI loop."""
    _configure_runtime_output()
    print("CLI started. Type 'exit' or 'quit' to stop.")
    memory: MemorySummary | None = None
    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Ending session.")
            break
        if not user_query:
            print("Please enter a question.")
            continue

        result = run_turn(user_query=user_query, previous_memory=memory)
        memory = result.memory
        _print_result(result)


if __name__ == "__main__":
    main()

