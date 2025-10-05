"""
Financial Analyst Agent

Responsible for analyzing financial health, ratios, and reporting quality.
Uses MCP-like tools to access structured financial data.
"""

from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentRole, AgentResponse
from tools.data_tools import FinancialDataTool


class FinancialAnalystAgent(BaseAgent):
    """
    Agent specialized in financial analysis and metrics calculation.

    This agent uses tools to access financial data and computes
    key metrics and ratios for investment decision-making.
    """

    def __init__(self, agent_id: str, data_tool: FinancialDataTool):
        super().__init__(agent_id, AgentRole.FINANCIAL_ANALYST)
        self.data_tool = data_tool

    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process financial analysis request.

        Expected request format:
        {
            "company_id": "COMPANY_X",
            "analysis_type": "comprehensive" | "ratios_only"
        }

        Returns:
            AgentResponse with financial analysis data
        """
        company_id = request.get("company_id")
        analysis_type = request.get("analysis_type", "comprehensive")

        if not company_id:
            return self.create_response(
                status="error",
                data={"error": "company_id is required"},
                metadata={"request": request}
            )

        # Use tool to fetch financial data
        financial_data = await self.data_tool.get_financial_ratios(company_id)

        if "error" in financial_data:
            return self.create_response(
                status="error",
                data=financial_data,
                metadata={"company_id": company_id}
            )

        # Analyze the data
        analysis = self._perform_analysis(financial_data, analysis_type)

        return self.create_response(
            status="success",
            data={
                "company_id": company_id,
                "financial_health": analysis["financial_health"],
                "key_metrics": analysis["key_metrics"],
                "assessment": analysis["assessment"],
                "raw_ratios": financial_data["ratios"]
            },
            metadata={
                "agent_role": self.role.value,
                "analysis_type": analysis_type,
                "tool_used": "FinancialDataTool"
            }
        )

    def _perform_analysis(
        self,
        financial_data: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Perform financial analysis on the data.

        Args:
            financial_data: Financial ratios and raw data
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results dictionary
        """
        ratios = financial_data["ratios"]

        # Assess financial health based on ratios
        gross_margin = ratios["gross_margin"]
        operating_margin = ratios["operating_margin"]
        net_margin = ratios["net_margin"]
        debt_to_equity = ratios["debt_to_equity"]

        # Health scoring
        health_score = 0
        health_factors = []

        # Margin analysis
        if gross_margin >= 40:
            health_score += 3
            health_factors.append("Strong gross margins (>=40%)")
        elif gross_margin >= 30:
            health_score += 2
            health_factors.append("Healthy gross margins (30-40%)")
        else:
            health_score += 1
            health_factors.append("Concerning gross margins (<30%)")

        if operating_margin >= 20:
            health_score += 3
            health_factors.append("Excellent operating efficiency (>=20%)")
        elif operating_margin >= 10:
            health_score += 2
            health_factors.append("Good operating efficiency (10-20%)")
        else:
            health_score += 1
            health_factors.append("Weak operating efficiency (<10%)")

        # Debt analysis
        if debt_to_equity < 0.5:
            health_score += 3
            health_factors.append("Conservative debt levels (<0.5 D/E)")
        elif debt_to_equity < 1.0:
            health_score += 2
            health_factors.append("Moderate debt levels (0.5-1.0 D/E)")
        else:
            health_score += 1
            health_factors.append("High debt levels (>1.0 D/E)")

        # Overall assessment
        max_score = 9
        health_percentage = (health_score / max_score) * 100

        if health_percentage >= 80:
            financial_health = "Strong"
            assessment = "Company demonstrates strong financial fundamentals with healthy margins and conservative leverage."
        elif health_percentage >= 60:
            financial_health = "Moderate"
            assessment = "Company shows acceptable financial health with some areas requiring monitoring."
        else:
            financial_health = "Weak"
            assessment = "Company exhibits financial vulnerabilities that require careful consideration."

        return {
            "financial_health": financial_health,
            "health_score": health_score,
            "health_percentage": round(health_percentage, 1),
            "key_metrics": {
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "net_margin": net_margin,
                "debt_to_equity": debt_to_equity
            },
            "health_factors": health_factors,
            "assessment": assessment
        }
