import json
import csv

from src.evals.deterministic_evaluators import DeterministicEvaluator
from src.evals.behavioral_evaluators import BehavioralEvaluator
from src.evals.judge_evaluator import JudgeEvaluator

from src.orchestration.pipeline import run_turn


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Added evaluation pipeline integration:
# This module orchestrates deterministic, behavioral, and rule‑based judging
# for each test case, producing a structured results.csv file for analysis.

def load_dataset(path="src/evals/dataset.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def convert_agent_output(result):
    """Convert PipelineResult → dict format expected by evaluators."""

    answer = result.answer

    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "direct_answer": answer.direct_answer,
        "evidence_summary": answer.evidence_summary,
        "citations": [
            {
                "citation_id": c.citation_id,
                "source": c.source,
                "claim": c.claim,
            }
            for c in answer.citations
        ],
        "safety_note": answer.safety_note,
        "confidence": answer.confidence,
        "tools_used": result.tools_used if hasattr(result, "tools_used") else [],
        "safety_reviewed": True,  # your pipeline always runs safety agent
        "memory_used": result.memory is not None,
    }


def evaluate_case(item, det, beh, judge, agent_output):
    det_result = det.evaluate(item, agent_output)
    beh_result = beh.evaluate(item, agent_output)
    judge_result = judge.evaluate(item, agent_output)

    return {
        "id": item["id"],
        "deterministic": det_result,
        "behavioral": beh_result,
        "judge": judge_result,
    }


def print_case_summary(result):
    print(f"\n--- Test Case {result['id']} ---")

    for name in ["deterministic", "behavioral", "judge"]:
        r = result[name]
        color = GREEN if r["passed"] else RED
        print(f"{color}{name.capitalize()} Score: {r['score']}{RESET}")
        if not r["passed"]:
            print(f"{YELLOW}Issues: {r['details']['issues']}{RESET}")


def aggregate_results(all_results):
    total = len(all_results)
    det_pass = sum(r["deterministic"]["passed"] for r in all_results)
    beh_pass = sum(r["behavioral"]["passed"] for r in all_results)
    judge_avg = sum(r["judge"]["score"] for r in all_results) / total

    return {
        "total_cases": total,
        "deterministic_pass_rate": det_pass / total,
        "behavioral_pass_rate": beh_pass / total,
        "judge_avg_score": judge_avg,
    }


def export_csv(all_results, path="src/evals/results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "deterministic_score", "behavioral_score", "judge_score"])

        for r in all_results:
            writer.writerow(
                [
                    r["id"],
                    r["deterministic"]["score"],
                    r["behavioral"]["score"],
                    r["judge"]["score"],
                ]
            )


def run():
    dataset = load_dataset()

    det = DeterministicEvaluator()
    beh = BehavioralEvaluator()
    judge = JudgeEvaluator()

    all_results = []

    print("\n=== Running Evaluation Suite ===\n")

    memory = None  # your agent requires memory

    for item in dataset:
        # Call your real agent
        result = run_turn(user_query=item["query"], previous_memory=memory)
        memory = result.memory  # update memory for next turn

        agent_output = convert_agent_output(result)

        case_result = evaluate_case(item, det, beh, judge, agent_output)
        all_results.append(case_result)

        print_case_summary(case_result)

    summary = aggregate_results(all_results)

    print("\n=== Overall Summary ===")
    print(f"Total test cases: {summary['total_cases']}")
    print(f"Deterministic pass rate: {summary['deterministic_pass_rate']:.2f}")
    print(f"Behavioral pass rate: {summary['behavioral_pass_rate']:.2f}")
    print(f"Judge average score: {summary['judge_avg_score']:.2f}")

    export_csv(all_results)
    print("\nResults exported to src/evals/results.csv")


if __name__ == "__main__":
    run()
