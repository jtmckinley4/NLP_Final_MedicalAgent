from pydantic_evals.evaluators import Evaluator

class JudgeEvaluator(Evaluator):
    """
    Simple rule-based judge evaluator.
    Scores the agent's answer quality using lightweight heuristics.
    """

    def evaluate(self, item, agent_output):
        score = 0
        issues = []

        # 1. Direct answer present?
        if agent_output.get("direct_answer"):
            score += 0.4
        else:
            issues.append("Missing direct answer")

        # 2. Evidence summary present?
        if agent_output.get("evidence_summary"):
            score += 0.3
        else:
            issues.append("Missing evidence summary")

        # 3. Safety note present?
        if agent_output.get("safety_note"):
            score += 0.2
        else:
            issues.append("Missing safety note")

        # 4. Confidence score present?
        if agent_output.get("confidence") is not None:
            score += 0.1
        else:
            issues.append("Missing confidence score")

        passed = score >= 0.7

        return {
            "passed": passed,
            "score": round(score, 2),
            "details": {"issues": issues},
        }
