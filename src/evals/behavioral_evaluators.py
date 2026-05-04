from pydantic_evals.evaluators import Evaluator
from typing import Any

class BehavioralEvaluator(Evaluator):
    """
    Behavioral evaluator for routing, tool usage, and memory behavior.
    """

    def evaluate(self, input: dict[str, Any], output: dict[str, Any]):
        issues = []

        # 1. Routing correctness
        expected_route = input.get("expected_route")
        actual_route = output.get("route")
        if expected_route and actual_route != expected_route:
            issues.append(f"Incorrect route: expected {expected_route}, got {actual_route}")

        # 2. openFDA usage check
        tools_used = output.get("tools_used", [])
        if "lookup_drug_info_openfda" in tools_used and actual_route != "medication":
            issues.append("openFDA used outside medication path.")

        # 3. Safety verifier check
        if not output.get("safety_reviewed", False):
            issues.append("Safety verifier did not run.")

        # 4. Memory behavior
        if input.get("depends_on"):
            if not output.get("memory_used", False):
                issues.append("Memory was expected but not used.")

        return {
            "passed": len(issues) == 0,
            "score": 1.0 if len(issues) == 0 else 0.0,
            "details": {"issues": issues}
        }
