"""
Financial Analyst Agent

Evaluates company financial health using real-time fundamentals from FMP.
Gemini analyzes 6 key metrics and explains its reasoning transparently.

Business Logic:
- Fetches: Revenue growth, margins (gross/operating/net), debt-to-equity, ROE
- Analyzes: Profitability trends, leverage, efficiency
- Returns: Sentiment score (-1 to +1), confidence, and step-by-step reasoning
- Output: Used by fusion engine (35% weight) for final recommendation
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai

from tools.fmp_tool import FMPTool
from core.agent_contracts import AgentOutput, SignalType, Evidence
from datetime import datetime

logger = logging.getLogger(__name__)


class FinancialAnalysis(BaseModel):
    """Gemini's structured response - ensures transparency in AI decision-making."""
    sentiment_score: float = Field(description="Score between -1.0 and +1.0")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    summary: str = Field(description="2-3 sentence summary of financial health")
    reasoning: str = Field(description="Step-by-step explanation of how you calculated the sentiment score")
    key_factors: list[str] = Field(description="Top 3 metrics that most influenced your score (e.g., 'Strong revenue growth', 'Weak margins')")


class ADKFinancialAnalyst:
    """
    Evaluates fundamental financial health.

    Think: CFO reviewing quarterly earnings - margins, growth, debt levels.
    """

    def __init__(self, agent_id: str = "financial_analyst"):
        self.agent_id = agent_id
        self.fmp_tool = FMPTool()
        self.use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"

        # Direct Gemini client (simpler than ADK, more reliable)
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

        logger.info(f"ADKFinancialAnalyst initialized (FMP: {self.use_fmp})")

    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Main analysis workflow.

        Steps: Fetch real data → Extract metrics → Gemini analyzes → Return transparent output
        """
        logger.info(f"Analyzing financials for {ticker}")

        # Step 1: Fetch real-time financial ratios from FMP
        if self.use_fmp:
            data = await self._fetch_fmp_data(ticker)
        else:
            data = self._get_sample_data(ticker)

        # Step 2: Extract 6 key metrics (fusion engine needs these exact values)
        metrics = self._extract_metrics(data)

        # Step 3: Gemini evaluates metrics and explains reasoning
        analysis = await self._analyze_with_gemini(ticker, company_name, metrics)

        # Step 4: Package for fusion engine with full transparency
        evidence = [
            Evidence(
                source="fmp_ratios",
                value=metrics,
                timestamp=datetime.utcnow(),
                confidence=0.9,
                description=analysis.summary
            )
        ]

        return AgentOutput(
            signal=SignalType.FINANCIAL,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,  # Raw numbers for fusion math
            sentiment=analysis.sentiment_score,  # Gemini's overall assessment
            confidence=analysis.confidence,
            evidence=evidence,
            metadata={
                "sector": sector,
                "data_source": "FMP" if self.use_fmp else "sample",
                "reasoning": analysis.reasoning,  # WHY this score? (transparency)
                "key_factors": analysis.key_factors  # WHAT drove the decision?
            }
        )

    async def _analyze_with_gemini(
        self,
        ticker: str,
        company_name: str,
        metrics: Dict[str, float]
    ) -> FinancialAnalysis:
        """
        Gemini analyzes metrics and explains its thinking.

        Returns structured output with score + reasoning (no black box).
        """

        prompt = f"""Analyze the financial health of {company_name} ({ticker}) based on these metrics:

Revenue Growth: {metrics.get('revenue_growth', 0):.1%}
Gross Margin: {metrics.get('gross_margin', 0):.1%}
Operating Margin: {metrics.get('operating_margin', 0):.1%}
Net Margin: {metrics.get('net_margin', 0):.1%}
Debt-to-Equity: {metrics.get('debt_to_equity', 0):.2f}
ROE: {metrics.get('roe', 0):.1%}

Provide:
1. Sentiment score (-1.0 to +1.0): Negative if weak fundamentals, positive if strong
2. Confidence (0.0 to 1.0): How confident you are in this assessment
3. Summary: 2-3 sentence explanation of financial health
4. Reasoning: Explain step-by-step how you arrived at the sentiment score. Show your logic for evaluating each metric and how you weighted them.
5. Key factors: List the top 3 specific metrics that most influenced your score (e.g., "Strong 15% revenue growth", "Healthy 42% gross margin", "Elevated 1.2 debt-to-equity ratio")"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=FinancialAnalysis
                )
            )

            result = json.loads(response.text)
            return FinancialAnalysis(**result)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback to neutral
            return FinancialAnalysis(
                sentiment_score=0.0,
                confidence=0.5,
                summary="Analysis unavailable",
                reasoning="Gemini API call failed",
                key_factors=["API error", "No analysis available", "Using neutral score"]
            )

    async def _fetch_fmp_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch real data from FMP."""
        try:
            ratios = await self.fmp_tool.get_financial_ratios(ticker)
            logger.info(f"✅ FMP data fetched for {ticker}")
            return ratios
        except Exception as e:
            logger.warning(f"FMP fetch failed: {e}, using sample data")
            return self._get_sample_data(ticker)

    def _get_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Sample financial data."""
        return {
            "ticker": ticker,
            "revenue_growth": 0.15,
            "gross_margin": 0.42,
            "operating_margin": 0.28,
            "net_margin": 0.22,
            "debt_to_equity": 1.2,
            "roe": 0.18
        }

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract 6 core financial metrics.

        These exact fields are required by the fusion engine's weighted averaging.
        """
        return {
            "revenue_growth": data.get("revenue_growth", 0.0),  # Growth trajectory
            "gross_margin": data.get("gross_margin", 0.0),      # Pricing power
            "operating_margin": data.get("operating_margin", 0.0),  # Operational efficiency
            "net_margin": data.get("net_margin", 0.0),          # Bottom-line profitability
            "debt_to_equity": data.get("debt_to_equity", 0.0),  # Financial leverage
            "roe": data.get("roe", 0.0)                          # Shareholder returns
        }
