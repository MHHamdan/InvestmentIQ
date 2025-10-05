"""
Qualitative Signal Agent

Responsible for interpreting unstructured data including news, employee reviews,
and market sentiment. Consolidates Market Sentiment, Workforce, and Market
Intelligence capabilities.
"""

from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentRole, AgentResponse
from tools.data_tools import QualitativeDataTool


class QualitativeSignalAgent(BaseAgent):
    """
    Agent specialized in processing and analyzing qualitative signals.

    This agent interprets unstructured text data to extract sentiment,
    identify key themes, and assess qualitative risks and opportunities.
    """

    def __init__(self, agent_id: str, data_tool: QualitativeDataTool):
        super().__init__(agent_id, AgentRole.QUALITATIVE_SIGNAL)
        self.data_tool = data_tool

    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process qualitative analysis request.

        Expected request format:
        {
            "company_id": "COMPANY_X",
            "focus_areas": ["sentiment", "workforce", "market_position"]
        }

        Returns:
            AgentResponse with qualitative analysis data
        """
        company_id = request.get("company_id")
        focus_areas = request.get("focus_areas", [
            "sentiment", "workforce", "market_position"
        ])

        if not company_id:
            return self.create_response(
                status="error",
                data={"error": "company_id is required"},
                metadata={"request": request}
            )

        # Use tool to process unstructured text
        qualitative_data = await self.data_tool.process_unstructured_text(company_id)

        if "error" in qualitative_data:
            return self.create_response(
                status="error",
                data=qualitative_data,
                metadata={"company_id": company_id}
            )

        # Analyze the qualitative signals
        analysis = self._perform_analysis(qualitative_data, focus_areas)

        return self.create_response(
            status="success",
            data={
                "company_id": company_id,
                "overall_sentiment": analysis["overall_sentiment"],
                "sentiment_score": analysis["sentiment_score"],
                "key_findings": analysis["key_findings"],
                "risk_assessment": analysis["risk_assessment"],
                "signal_strength": analysis["signal_strength"]
            },
            metadata={
                "agent_role": self.role.value,
                "focus_areas": focus_areas,
                "tool_used": "QualitativeDataTool",
                "themes_identified": qualitative_data.get("key_themes", [])
            }
        )

    def _perform_analysis(
        self,
        qualitative_data: Dict[str, Any],
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """
        Perform qualitative signal analysis.

        Args:
            qualitative_data: Processed qualitative data from tool
            focus_areas: Areas to focus analysis on

        Returns:
            Analysis results dictionary
        """
        sentiment_score = qualitative_data.get("sentiment_score", 0.0)
        sentiment_label = qualitative_data.get("sentiment_label", "Neutral")
        themes = qualitative_data.get("key_themes", [])
        raw_text = qualitative_data.get("raw_text", "")

        # Analyze key findings
        key_findings = self._extract_key_findings(raw_text, themes, focus_areas)

        # Assess risks
        risk_assessment = self._assess_risks(
            sentiment_score,
            themes,
            key_findings
        )

        # Determine signal strength
        signal_strength = self._calculate_signal_strength(
            sentiment_score,
            len(themes),
            len(key_findings)
        )

        return {
            "overall_sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "key_findings": key_findings,
            "risk_assessment": risk_assessment,
            "signal_strength": signal_strength
        }

    def _extract_key_findings(
        self,
        raw_text: str,
        themes: List[str],
        focus_areas: List[str]
    ) -> List[Dict[str, str]]:
        """Extract key findings from the text"""
        findings = []
        text_lower = raw_text.lower()

        # Workforce findings
        if "workforce" in focus_areas or "workforce" in themes:
            if any(word in text_lower for word in ["layoff", "exodus", "turnover"]):
                findings.append({
                    "category": "Workforce",
                    "finding": "Significant employee turnover detected",
                    "severity": "high"
                })
            if "hiring" in text_lower:
                findings.append({
                    "category": "Workforce",
                    "finding": "Active hiring initiatives",
                    "severity": "low"
                })

        # Leadership findings
        if "leadership" in themes:
            if any(word in text_lower for word in ["ceo departure", "executive exodus"]):
                findings.append({
                    "category": "Leadership",
                    "finding": "Executive leadership changes",
                    "severity": "high"
                })

        # Market findings
        if "market_position" in focus_areas or "market" in themes:
            if "market share" in text_lower:
                if "loss" in text_lower or "decline" in text_lower:
                    findings.append({
                        "category": "Market Position",
                        "finding": "Market share erosion",
                        "severity": "medium"
                    })

        # Sentiment findings
        if "sentiment" in focus_areas:
            if any(word in text_lower for word in ["scandal", "crisis", "investigation"]):
                findings.append({
                    "category": "Reputation",
                    "finding": "Negative public perception or controversy",
                    "severity": "high"
                })

        return findings

    def _assess_risks(
        self,
        sentiment_score: float,
        themes: List[str],
        findings: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Assess qualitative risks"""
        risk_level = "low"
        risk_factors = []

        # Sentiment-based risk
        if sentiment_score <= -0.6:
            risk_level = "high"
            risk_factors.append("Extremely negative sentiment")
        elif sentiment_score <= -0.3:
            risk_level = "medium"
            risk_factors.append("Negative sentiment trend")

        # Theme-based risk
        high_risk_themes = ["workforce", "leadership"]
        if any(theme in high_risk_themes for theme in themes):
            if risk_level != "high":
                risk_level = "medium"
            risk_factors.append("Organizational instability indicators")

        # Findings-based risk
        high_severity_findings = [
            f for f in findings if f.get("severity") == "high"
        ]
        if len(high_severity_findings) >= 2:
            risk_level = "high"
            risk_factors.append("Multiple high-severity issues identified")

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "high_severity_count": len(high_severity_findings),
            "total_findings": len(findings)
        }

    def _calculate_signal_strength(
        self,
        sentiment_score: float,
        theme_count: int,
        finding_count: int
    ) -> str:
        """Calculate the strength of the qualitative signal"""
        # Strong signals are clear and unambiguous
        if abs(sentiment_score) >= 0.6 and finding_count >= 3:
            return "strong"
        elif abs(sentiment_score) >= 0.3 and theme_count >= 2:
            return "moderate"
        else:
            return "weak"
