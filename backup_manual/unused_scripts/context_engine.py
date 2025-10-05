"""
Context Engine Agent

Responsible for storing and applying historical correlation rules to judgment
scenarios. Acts as the memory and pattern-matching component of the system.
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent, AgentRole, AgentResponse
from tools.data_tools import ContextRuleTool


class ContextEngineAgent(BaseAgent):
    """
    Agent specialized in context rule management and application.

    This agent maintains historical patterns and applies context-specific
    rules to resolve conflicting signals and inform decision-making.
    """

    def __init__(self, agent_id: str, rule_tool: ContextRuleTool):
        super().__init__(agent_id, AgentRole.CONTEXT_ENGINE)
        self.rule_tool = rule_tool

    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process context rule application request.

        Expected request format:
        {
            "scenario_type": "contrarian_opportunity",
            "context": {
                "financial_health": "Strong",
                "sentiment": "Very Negative",
                "sector": "technology"
            }
        }

        Returns:
            AgentResponse with applicable rule and recommendation
        """
        scenario_type = request.get("scenario_type")
        context = request.get("context", {})

        if not scenario_type:
            return self.create_response(
                status="error",
                data={"error": "scenario_type is required"},
                metadata={"request": request}
            )

        # Use tool to fetch applicable rule
        sector = context.get("sector")
        rule = await self.rule_tool.get_context_rule(scenario_type, sector)

        if "error" in rule:
            return self.create_response(
                status="error",
                data=rule,
                metadata={"scenario_type": scenario_type}
            )

        # Apply the rule to the context
        application = self._apply_rule(rule, context)

        return self.create_response(
            status="success",
            data={
                "rule_applied": rule,
                "context_assessment": application["assessment"],
                "recommendation": application["recommendation"],
                "confidence": application["confidence"],
                "rationale": application["rationale"]
            },
            metadata={
                "agent_role": self.role.value,
                "scenario_type": scenario_type,
                "tool_used": "ContextRuleTool"
            }
        )

    def _apply_rule(
        self,
        rule: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a context rule to the given context.

        Args:
            rule: The context rule to apply
            context: Current context information

        Returns:
            Application results with recommendation
        """
        rule_conditions = rule.get("conditions", {})
        rule_outcome = rule.get("outcome", {})
        historical_accuracy = rule.get("historical_accuracy", 0.75)

        # Check if conditions match
        conditions_met = self._check_conditions(rule_conditions, context)

        # Generate assessment
        if conditions_met["all_met"]:
            assessment = "Context rule fully applicable"
            confidence = historical_accuracy
            recommendation = rule_outcome.get("recommendation", "No recommendation")
            rationale = self._build_rationale(
                rule,
                context,
                conditions_met,
                historical_accuracy
            )
        else:
            assessment = "Context rule partially applicable"
            confidence = historical_accuracy * 0.6
            recommendation = "Exercise caution - conditions not fully met"
            rationale = f"Only {conditions_met['met_count']}/{conditions_met['total_count']} conditions satisfied"

        return {
            "assessment": assessment,
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "rationale": rationale,
            "conditions_met": conditions_met
        }

    def _check_conditions(
        self,
        rule_conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if rule conditions are met by the context"""
        met_count = 0
        total_count = 0
        details = []

        for condition_key, condition_value in rule_conditions.items():
            total_count += 1
            context_value = context.get(condition_key)

            if isinstance(condition_value, dict):
                # Handle complex conditions (e.g., {"min": 40})
                if "min" in condition_value:
                    if context_value and context_value >= condition_value["min"]:
                        met_count += 1
                        details.append(f"{condition_key} meets minimum: {context_value} >= {condition_value['min']}")
                    else:
                        details.append(f"{condition_key} below minimum: {context_value} < {condition_value['min']}")

                elif "in" in condition_value:
                    if context_value in condition_value["in"]:
                        met_count += 1
                        details.append(f"{condition_key} matches: {context_value}")
                    else:
                        details.append(f"{condition_key} mismatch: {context_value} not in {condition_value['in']}")
            else:
                # Simple equality check
                if context_value == condition_value:
                    met_count += 1
                    details.append(f"{condition_key} matches: {context_value}")
                else:
                    details.append(f"{condition_key} mismatch: {context_value} != {condition_value}")

        return {
            "all_met": met_count == total_count,
            "met_count": met_count,
            "total_count": total_count,
            "details": details
        }

    def _build_rationale(
        self,
        rule: Dict[str, Any],
        context: Dict[str, Any],
        conditions_met: Dict[str, Any],
        historical_accuracy: float
    ) -> str:
        """Build a detailed rationale for the recommendation"""
        rule_description = rule.get("description", "No description available")

        rationale_parts = [
            f"Applied Rule: {rule_description}",
            f"\nHistorical Pattern: This scenario has occurred {rule.get('historical_occurrences', 'multiple')} times "
            f"with a {historical_accuracy*100:.0f}% accuracy rate.",
            f"\nCondition Analysis: {conditions_met['met_count']}/{conditions_met['total_count']} conditions satisfied"
        ]

        # Add specific context insights
        if context.get("financial_health") == "Strong" and context.get("sentiment") in ["Negative", "Very Negative"]:
            rationale_parts.append(
                "\nKey Insight: Strong financial fundamentals combined with negative sentiment "
                "often indicates market overreaction, presenting a contrarian opportunity."
            )

        return "\n".join(rationale_parts)
