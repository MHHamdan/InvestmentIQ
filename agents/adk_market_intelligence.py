"""
Market Intelligence Agent

Evaluates Wall Street consensus using analyst ratings and price targets.
Gemini synthesizes professional opinions and explains its reasoning transparently.

Business Logic:
- Fetches: Analyst ratings (buy/sell/hold), price targets, current price
- Analyzes: Consensus sentiment, upside potential, analyst momentum
- Returns: Sentiment score (-1 to +1), confidence, and step-by-step reasoning
- Output: Used by fusion engine (30% weight) for final recommendation
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


class MarketAnalysis(BaseModel):
    """Gemini's structured response - ensures transparency in AI decision-making."""
    sentiment_score: float = Field(description="Score between -1.0 and +1.0")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    summary: str = Field(description="2-3 sentence summary of analyst consensus")
    reasoning: str = Field(description="Step-by-step explanation of how you calculated the sentiment score")
    key_factors: list[str] = Field(description="Top 3 factors that influenced your score (e.g., 'Strong buy consensus', 'High upside potential')")


class ADKMarketIntelligence:
    """
    Evaluates Wall Street analyst consensus.

    Think: Institutional investor reviewing analyst reports before making a trade.
    """

    def __init__(self, agent_id: str = "market_intelligence"):
        self.agent_id = agent_id
        self.fmp_tool = FMPTool()
        self.use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"

        # Direct Gemini client (simpler than ADK, more reliable)
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

        logger.info(f"ADKMarketIntelligence initialized (FMP: {self.use_fmp})")

    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Main analysis workflow.

        Steps: Fetch analyst data → Extract metrics → Gemini analyzes → Return transparent output
        """
        logger.info(f"Analyzing market intelligence for {ticker}")

        # Step 1: Fetch analyst ratings, price targets, and current price from FMP
        if self.use_fmp:
            data = await self._fetch_fmp_data(ticker)
        else:
            data = self._get_sample_data(ticker)

        # Step 2: Extract key metrics (fusion engine needs these exact values)
        metrics = self._extract_metrics(data)

        # Step 3: Gemini evaluates analyst consensus and explains reasoning
        analysis = await self._analyze_with_gemini(ticker, company_name, metrics)

        # Step 4: Package for fusion engine with full transparency
        evidence = [
            Evidence(
                source="fmp_analysts",
                value=metrics,
                timestamp=datetime.utcnow(),
                confidence=0.85,
                description=analysis.summary
            )
        ]

        return AgentOutput(
            signal=SignalType.MARKET_INTELLIGENCE,
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
    ) -> MarketAnalysis:
        """
        Gemini analyzes analyst consensus and explains its thinking.

        Returns structured output with score + reasoning (no black box).
        """

        prompt = f"""Analyze Wall Street analyst consensus for {company_name} ({ticker}) based on these metrics:

Current Price: ${metrics.get('current_price', 0):.2f}
Average Price Target: ${metrics.get('avg_price_target', 0):.2f}
Upside Potential: {metrics.get('upside_potential', 0):.1f}%
Buy Ratings: {int(metrics.get('buy_count', 0))}
Sell Ratings: {int(metrics.get('sell_count', 0))}
Hold Ratings: {int(metrics.get('hold_count', 0))}
Total Analysts: {int(metrics.get('total_analysts', 0))}

Provide:
1. Sentiment score (-1.0 to +1.0): Negative if bearish consensus, positive if bullish
2. Confidence (0.0 to 1.0): How confident you are in this assessment
3. Summary: 2-3 sentence explanation of analyst consensus
4. Reasoning: Explain step-by-step how you arrived at the sentiment score. Show your logic for evaluating analyst ratings and price targets.
5. Key factors: List the top 3 specific factors that most influenced your score (e.g., "Strong buy consensus (15 buy vs 2 sell)", "20% upside potential", "Recent upgrade momentum")"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=MarketAnalysis
                )
            )

            result = json.loads(response.text)
            return MarketAnalysis(**result)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback to neutral
            return MarketAnalysis(
                sentiment_score=0.0,
                confidence=0.5,
                summary="Analysis unavailable",
                reasoning="Gemini API call failed",
                key_factors=["API error", "No analysis available", "Using neutral score"]
            )

    async def _fetch_fmp_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch real analyst data from FMP."""
        try:
            logger.info(f"Fetching FMP analyst data for {ticker}")

            # Fetch all three: ratings, price targets, current price
            ratings = await self.fmp_tool.get_analyst_ratings(ticker)
            price_targets = await self.fmp_tool.get_price_target_consensus(ticker)
            quote = await self.fmp_tool.get_quote(ticker)

            fmp_ratings = ratings.get("ratings", [])

            # Count buy/sell/hold ratings
            buy_count = sum(1 for r in fmp_ratings if any(x in r.get("newGrade", "").lower() for x in ["buy", "outperform", "strong buy"]))
            sell_count = sum(1 for r in fmp_ratings if any(x in r.get("newGrade", "").lower() for x in ["sell", "underperform", "strong sell"]))
            hold_count = sum(1 for r in fmp_ratings if "hold" in r.get("newGrade", "").lower())

            current_price = quote.get("price", 0.0)
            avg_target = price_targets.get("avg_target", 0.0)

            logger.info(f"✅ FMP analyst data fetched for {ticker}")

            return {
                "ticker": ticker,
                "current_price": current_price,
                "avg_price_target": avg_target,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "total_analysts": len(fmp_ratings),
                "ratings_data": fmp_ratings[:5]  # Keep top 5 for context
            }

        except Exception as e:
            logger.warning(f"FMP fetch failed: {e}, using sample data")
            return self._get_sample_data(ticker)

    def _get_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Sample analyst data."""
        return {
            "ticker": ticker,
            "current_price": 226.0,
            "avg_price_target": 250.0,
            "buy_count": 15,
            "sell_count": 2,
            "hold_count": 8,
            "total_analysts": 25
        }

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key analyst consensus metrics.

        These exact fields are required by the fusion engine's weighted averaging.
        """
        current_price = data.get("current_price", 0.0)
        avg_target = data.get("avg_price_target", 0.0)

        # Calculate upside potential (key metric for investors)
        upside_potential = 0.0
        if current_price > 0:
            upside_potential = ((avg_target - current_price) / current_price) * 100

        return {
            "current_price": current_price,
            "avg_price_target": avg_target,
            "upside_potential": upside_potential,  # % upside from current to target
            "buy_count": float(data.get("buy_count", 0)),  # Bullish analyst count
            "sell_count": float(data.get("sell_count", 0)),  # Bearish analyst count
            "hold_count": float(data.get("hold_count", 0)),  # Neutral analyst count
            "total_analysts": float(data.get("total_analysts", 0))  # Sample size
        }
