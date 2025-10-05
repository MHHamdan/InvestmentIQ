"""
Evaluation Framework for InvestmentIQ MVAS

Advanced Feature: Comprehensive evaluation system with custom evaluators,
LLM-as-judge, and summary evaluators for multi-agent performance assessment.
"""

from typing import Dict, Any, List
from langsmith.schemas import Run, Example
from langsmith.evaluation.evaluator import EvaluationResult
from utils.llm_factory import get_llm_factory


class InvestmentIQEvaluators:
    """
    Custom evaluators for InvestmentIQ MVAS agents.

    Evaluator Types:
    1. Financial Analysis Quality
    2. Sentiment Detection Accuracy
    3. Recommendation Consistency
    4. Conflict Resolution Effectiveness
    """

    def __init__(self):
        self.llm_factory = get_llm_factory()

        # Create LLM for judge evaluations (using Hugging Face)
        try:
            self.judge_llm = self.llm_factory.create_chat_model(
                provider="huggingface",
                temperature=0.0,
                max_tokens=500
            )
        except Exception as e:
            print(f"Warning: Could not initialize judge LLM: {e}")
            self.judge_llm = None

    # ====================
    # Custom Evaluators
    # ====================

    def financial_analysis_structure(self, run: Run, example: Example) -> EvaluationResult:
        """
        Evaluate if financial analysis has proper structure.

        Checks for:
        - Financial ratios (gross margin, debt-to-equity, etc.)
        - Risk assessment
        - Health classification
        """
        response = run.outputs.get("data", {})
        financial_data = response.get("financial_analysis", {})

        required_fields = ["financial_health", "key_ratios", "risk_level"]
        found_fields = sum(1 for field in required_fields if field in financial_data)

        score = found_fields / len(required_fields)

        return EvaluationResult(
            key="financial_structure",
            score=score,
            comment=f"Found {found_fields}/{len(required_fields)} required financial analysis fields"
        )

    def sentiment_detection_accuracy(self, run: Run, example: Example) -> EvaluationResult:
        """
        Evaluate qualitative sentiment detection accuracy.

        Checks for:
        - Sentiment score presence
        - Risk level identification
        - Theme extraction
        """
        response = run.outputs.get("data", {})
        qualitative_data = response.get("qualitative_analysis", {})

        # Check sentiment score range
        sentiment = qualitative_data.get("sentiment_score", 0)
        has_valid_sentiment = -1 <= sentiment <= 1

        # Check risk level
        risk_level = qualitative_data.get("risk_level", "")
        has_risk = risk_level in ["Low", "Medium", "High", "Critical"]

        # Check themes
        themes = qualitative_data.get("themes", [])
        has_themes = len(themes) > 0

        score = sum([has_valid_sentiment, has_risk, has_themes]) / 3

        return EvaluationResult(
            key="sentiment_accuracy",
            score=score,
            comment=f"Sentiment valid: {has_valid_sentiment}, Risk: {has_risk}, Themes: {has_themes}"
        )

    def recommendation_consistency(self, run: Run, example: Example) -> EvaluationResult:
        """
        Evaluate recommendation consistency with analysis.

        Checks if recommendation aligns with:
        - Financial health
        - Sentiment signals
        - Risk assessments
        """
        response = run.outputs.get("data", {})
        recommendation = response.get("recommendation", {})
        action = recommendation.get("action", "")

        financial_health = response.get("financial_analysis", {}).get("financial_health", "")
        sentiment_score = response.get("qualitative_analysis", {}).get("sentiment_score", 0)

        # Consistency logic
        is_consistent = True

        # Strong BUY should align with positive signals
        if action == "Strong BUY":
            if financial_health != "Strong" or sentiment_score < 0:
                is_consistent = False

        # HOLD/SELL should align with negative signals
        if action in ["HOLD", "SELL"]:
            if financial_health == "Strong" and sentiment_score > 0:
                is_consistent = False

        score = 1.0 if is_consistent else 0.3

        return EvaluationResult(
            key="recommendation_consistency",
            score=score,
            comment=f"Action: {action}, Financial: {financial_health}, Sentiment: {sentiment_score:.2f}, Consistent: {is_consistent}"
        )

    def conflict_detection_accuracy(self, run: Run, example: Example) -> EvaluationResult:
        """
        Evaluate conflict detection accuracy.

        Checks if system correctly identified conflicts between signals.
        """
        response = run.outputs.get("data", {})
        conflict_detected = response.get("conflict_detected", False)

        financial_health = response.get("financial_analysis", {}).get("financial_health", "")
        sentiment_score = response.get("qualitative_analysis", {}).get("sentiment_score", 0)

        # Expected conflict: Strong financials + Negative sentiment
        expected_conflict = (financial_health == "Strong" and sentiment_score < -0.3)

        # Check if detection matches expectation
        is_accurate = (conflict_detected == expected_conflict)

        score = 1.0 if is_accurate else 0.0

        return EvaluationResult(
            key="conflict_detection",
            score=score,
            comment=f"Expected: {expected_conflict}, Detected: {conflict_detected}, Accurate: {is_accurate}"
        )

    # ====================
    # LLM-as-Judge Evaluators
    # ====================

    def investment_quality_judge(self, run: Run, example: Example) -> EvaluationResult:
        """
        LLM-as-judge evaluator for overall investment recommendation quality.

        Uses Hugging Face model to assess:
        - Strategic depth
        - Risk awareness
        - Actionability
        - Investment soundness
        """
        if not self.judge_llm:
            return EvaluationResult(
                key="investment_quality",
                score=0.5,
                comment="Judge LLM not available"
            )

        response = run.outputs.get("data", {})
        company_id = example.inputs.get("company_id", "Unknown")

        judge_prompt = f"""You are an expert investment analyst evaluating the quality of an investment recommendation.

Company: {company_id}
Recommendation: {response.get('recommendation', {})}

Rate the recommendation quality on a scale of 0-1 considering:
1. Strategic Depth: Does it provide deep strategic insights?
2. Risk Awareness: Are risks properly identified and assessed?
3. Actionability: Is the recommendation clear and actionable?
4. Investment Soundness: Is it a sound investment decision?

Provide a single score between 0.0 and 1.0, followed by brief reasoning.
Format: Score: 0.X | Reasoning: ...
"""

        try:
            from langchain.schema import HumanMessage
            judge_response = self.judge_llm.invoke([HumanMessage(content=judge_prompt)])
            judge_content = str(judge_response.content if hasattr(judge_response, 'content') else judge_response)

            # Parse score
            score = 0.5  # Default
            if "Score:" in judge_content:
                try:
                    score_str = judge_content.split("Score:")[1].split("|")[0].strip()
                    score = float(score_str)
                except:
                    pass

            return EvaluationResult(
                key="investment_quality",
                score=score,
                comment=judge_content[:300]
            )

        except Exception as e:
            return EvaluationResult(
                key="investment_quality",
                score=0.5,
                comment=f"Evaluation error: {str(e)}"
            )

    # ====================
    # Summary Evaluators
    # ====================

    def average_confidence_score(self, outputs: List[dict], reference_outputs: List[dict]) -> dict:
        """
        Summary evaluator: Calculate average recommendation confidence.

        Aggregates confidence scores across all experiments.
        """
        confidence_scores = []

        for output_dict in outputs:
            data = output_dict.get("data", {})
            confidence = data.get("recommendation", {}).get("confidence", 0)
            confidence_scores.append(confidence)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return {
            "key": "avg_confidence",
            "score": avg_confidence,
            "comment": f"Average confidence: {avg_confidence:.2f} across {len(confidence_scores)} runs"
        }

    def conflict_detection_rate(self, outputs: List[dict], reference_outputs: List[dict]) -> dict:
        """
        Summary evaluator: Calculate conflict detection rate.

        Measures how often the system detects signal conflicts.
        """
        total_runs = len(outputs)
        conflicts_detected = sum(
            1 for output in outputs
            if output.get("data", {}).get("conflict_detected", False)
        )

        detection_rate = conflicts_detected / total_runs if total_runs > 0 else 0.0

        return {
            "key": "conflict_rate",
            "score": detection_rate,
            "comment": f"Conflicts detected in {conflicts_detected}/{total_runs} runs ({detection_rate*100:.1f}%)"
        }

    def context_rule_effectiveness(self, outputs: List[dict], reference_outputs: List[dict]) -> dict:
        """
        Summary evaluator: Measure context rule application effectiveness.

        Tracks how often context rules are applied and their historical accuracy.
        """
        total_runs = len(outputs)
        rules_applied = 0
        total_accuracy = 0.0

        for output in outputs:
            data = output.get("data", {})
            context_rule = data.get("recommendation", {}).get("context_rule_applied")

            if context_rule:
                rules_applied += 1
                accuracy = context_rule.get("historical_accuracy", 0)
                total_accuracy += accuracy

        avg_accuracy = total_accuracy / rules_applied if rules_applied > 0 else 0.0

        return {
            "key": "context_effectiveness",
            "score": avg_accuracy,
            "comment": f"Context rules applied: {rules_applied}/{total_runs}, Avg accuracy: {avg_accuracy:.2f}"
        }


# Factory function
def get_evaluators() -> InvestmentIQEvaluators:
    """Get InvestmentIQ evaluators instance."""
    return InvestmentIQEvaluators()
