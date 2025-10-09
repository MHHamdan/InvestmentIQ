"""
Context Engine Agent

Evaluates macro environment and sector trends affecting investment climate.
Gemini synthesizes economic data and sector performance, explaining reasoning transparently.

Business Logic:
- Fetches: GDP growth, unemployment, fed funds rate (FRED), sector performance (FMP)
- Analyzes: Market cycle phase, sector vs S&P 500, economic tailwinds/headwinds
- Returns: Sentiment score (-1 to +1), confidence, and step-by-step reasoning
- Output: Used by fusion engine (10% weight) for final recommendation
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
import aiohttp

from core.agent_contracts import AgentOutput, SignalType, Evidence
from tools.fmp_tool import FMPTool
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextAnalysis(BaseModel):
    """Gemini's structured response - ensures transparency in AI decision-making."""
    sentiment_score: float = Field(description="Score between -1.0 and +1.0")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    summary: str = Field(description="2-3 sentence summary of macro/sector context")
    reasoning: str = Field(description="Step-by-step explanation of how you calculated the sentiment score")
    key_factors: list[str] = Field(description="Top 3 macro/sector factors that influenced your score (e.g., 'Expansion cycle', 'Sector outperforming')")
    market_cycle: str = Field(description="Current cycle phase: expansion, peak, contraction, or trough")
    sector_outlook: str = Field(description="Sector trend: bullish, stable, or bearish")


class ADKContextEngine:
    """
    Evaluates macroeconomic environment and sector trends.

    Think: Economist assessing market cycle and sector rotation dynamics.
    """

    def __init__(self, agent_id: str = "context_engine"):
        self.agent_id = agent_id
        self.fmp_tool = FMPTool()
        self.fred_api_key = os.getenv("FRED_API_KEY", "")

        # Direct Gemini client (simpler than ADK, more reliable)
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

        logger.info(f"ADKContextEngine initialized (FMP: {self.fmp_tool.enabled}, FRED: {bool(self.fred_api_key)})")

    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Main analysis workflow.

        Steps: Fetch macro/sector data → Extract metrics → Gemini analyzes → Return transparent output
        """
        logger.info(f"Analyzing context for {ticker} in {sector or 'Unknown'} sector")

        # Step 1: Fetch macroeconomic indicators (FRED) and sector performance (FMP)
        data = await self._fetch_context_data(ticker, sector)

        # Step 2: Extract key metrics (fusion engine needs these exact values)
        metrics = self._extract_metrics(data)

        # Step 3: Gemini evaluates macro/sector context and explains reasoning
        analysis = await self._analyze_with_gemini(ticker, company_name, sector, data)

        # Step 4: Package for fusion engine with full transparency
        evidence = [
            Evidence(
                source="fred_fmp_macro",
                value=metrics,
                timestamp=datetime.utcnow(),
                confidence=0.80,
                description=analysis.summary
            )
        ]

        return AgentOutput(
            signal=SignalType.CONTEXT,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,  # Raw numbers for fusion math
            sentiment=analysis.sentiment_score,  # Gemini's overall assessment
            confidence=analysis.confidence,
            evidence=evidence,
            metadata={
                "sector": sector,
                "data_source": "FMP+FRED" if self.fmp_tool.enabled else "sample",
                "reasoning": analysis.reasoning,  # WHY this score? (transparency)
                "key_factors": analysis.key_factors,  # WHAT drove the decision?
                "market_cycle": analysis.market_cycle,
                "sector_outlook": analysis.sector_outlook
            }
        )

    async def _analyze_with_gemini(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str],
        data: Dict[str, Any]
    ) -> ContextAnalysis:
        """
        Gemini analyzes macro/sector context and explains its thinking.

        Returns structured output with score + reasoning (no black box).
        """

        prompt = f"""Analyze the macroeconomic and sector context for {company_name} ({ticker}) in {sector or 'Unknown'} sector:

Economic Indicators:
- GDP Growth: {data.get('gdp_growth', 0):.2f}%
- Unemployment Rate: {data.get('unemployment_rate', 0):.1f}%
- Federal Funds Rate: {data.get('fed_funds_rate', 0):.2f}%

Sector Performance:
- Sector Performance: {data.get('sector_performance', 0):.2f}%
- vs S&P 500: {data.get('sector_vs_sp500', 'unknown')}

Provide:
1. Sentiment score (-1.0 to +1.0): Negative if recessionary/bearish environment, positive if expansionary/bullish
2. Confidence (0.0 to 1.0): How confident you are in this assessment
3. Summary: 2-3 sentence explanation of macro and sector context
4. Reasoning: Explain step-by-step how you arrived at the sentiment score. Evaluate each economic indicator and sector trend.
5. Key factors: List the top 3 macro/sector factors that most influenced your score (e.g., "Strong GDP growth of 2.8%", "Sector outperforming by 5%", "Rising fed funds rate")
6. Market cycle: What phase is the economy in? (expansion, peak, contraction, or trough)
7. Sector outlook: What's the sector trend? (bullish, stable, or bearish)"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=ContextAnalysis
                )
            )

            result = json.loads(response.text)
            return ContextAnalysis(**result)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback to neutral
            return ContextAnalysis(
                sentiment_score=0.0,
                confidence=0.5,
                summary="Analysis unavailable",
                reasoning="Gemini API call failed",
                key_factors=["API error", "No analysis available", "Using neutral score"],
                market_cycle="expansion",
                sector_outlook="stable"
            )

    async def _fetch_context_data(self, ticker: str, sector: Optional[str]) -> Dict[str, Any]:
        """
        Fetch REAL macroeconomic and sector data.

        Uses FRED for economic indicators and FMP for sector performance.
        Falls back to reasonable defaults if APIs unavailable.
        """
        data = {
            "ticker": ticker,
            "sector": sector,
            "gdp_growth": 2.5,  # Default fallback
            "unemployment_rate": 3.8,
            "fed_funds_rate": 5.33,
            "sector_performance": 0.0,
            "sector_vs_sp500": "unknown"
        }

        # Fetch FMP Sector Performance (REAL)
        if self.fmp_tool.enabled and sector:
            try:
                sector_data = await self._fetch_fmp_sector_performance(sector)
                data.update(sector_data)
            except Exception as e:
                logger.warning(f"FMP sector fetch failed: {e}")

        # Fetch FRED Macro Data (REAL if API key available)
        if self.fred_api_key:
            try:
                macro_data = await self._fetch_fred_macro_indicators()
                data.update(macro_data)
            except Exception as e:
                logger.warning(f"FRED fetch failed: {e}")

        return data

    async def _fetch_fmp_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Fetch real sector performance from FMP."""
        try:
            # Note: FMP free tier may not have this endpoint, will gracefully fail
            url = "https://financialmodelingprep.com/api/v3/sectors-performance"
            params = {"apikey": self.fmp_tool.api_key}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 403:
                        logger.warning("FMP sector endpoint requires premium plan")
                        return {"sector_performance": 0.0, "sector_vs_sp500": "unknown"}

                    response.raise_for_status()
                    sectors_data = await response.json()

                    # Find matching sector
                    sector_lower = sector.lower()
                    for s in sectors_data:
                        sector_name = s.get("sector", "").lower()
                        if sector_lower in sector_name or sector_name in sector_lower:
                            perf_str = s.get("changesPercentage", "0%")
                            perf = float(perf_str.replace("%", ""))
                            vs_sp500 = "outperforming" if perf > 0 else "underperforming" if perf < 0 else "inline"

                            logger.info(f"✅ FMP sector data: {sector} = {perf:.2f}%")
                            return {
                                "sector_performance": perf,
                                "sector_vs_sp500": vs_sp500
                            }

                    return {"sector_performance": 0.0, "sector_vs_sp500": "unknown"}

        except Exception as e:
            logger.error(f"FMP sector performance API error: {e}")
            return {"sector_performance": 0.0, "sector_vs_sp500": "unknown"}

    async def _fetch_fred_macro_indicators(self) -> Dict[str, Any]:
        """Fetch real macroeconomic indicators from FRED API."""
        try:
            base_url = "https://api.stlouisfed.org/fred/series/observations"

            async with aiohttp.ClientSession() as session:
                # Fetch all 3 indicators in parallel
                gdp_task = self._fetch_fred_series(session, "GDPC1", base_url)
                unemp_task = self._fetch_fred_series(session, "UNRATE", base_url)
                fed_task = self._fetch_fred_series(session, "FEDFUNDS", base_url)

                gdp_data, unemp_data, fed_data = await asyncio.gather(
                    gdp_task, unemp_task, fed_task, return_exceptions=True
                )

                # Parse results
                gdp_growth = self._calculate_gdp_growth(gdp_data) if not isinstance(gdp_data, Exception) else 2.5
                unemployment = float(unemp_data) if not isinstance(unemp_data, Exception) else 3.8
                fed_funds = float(fed_data) if not isinstance(fed_data, Exception) else 5.33

                logger.info(f"✅ FRED data: GDP={gdp_growth:.2f}%, Unemployment={unemployment:.1f}%, Fed={fed_funds:.2f}%")

                return {
                    "gdp_growth": gdp_growth,
                    "unemployment_rate": unemployment,
                    "fed_funds_rate": fed_funds
                }

        except Exception as e:
            logger.error(f"FRED API error: {e}")
            return {
                "gdp_growth": 2.5,
                "unemployment_rate": 3.8,
                "fed_funds_rate": 5.33
            }

    async def _fetch_fred_series(self, session, series_id: str, base_url: str):
        """Fetch FRED time series - returns list of last 5 values for GDP growth calc."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5  # Need at least 2 for growth calculation
        }

        async with session.get(base_url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
            observations = data.get("observations", [])

            if not observations:
                raise Exception(f"No data for {series_id}")

            # For GDP: return list of values for growth calculation
            # For others: return latest single value
            if series_id == "GDPC1":
                # Return list of GDP values (need 2 for growth calc)
                values = []
                for obs in observations[:2]:  # Get last 2 quarters
                    value = obs.get("value")
                    if value and value != ".":
                        values.append(float(value))
                return values if len(values) >= 2 else [observations[0].get("value", 0)]
            else:
                # For unemployment and fed funds, return single latest value
                for obs in observations:
                    value = obs.get("value")
                    if value and value != ".":
                        return float(value)
                raise Exception(f"No valid data for {series_id}")

    def _calculate_gdp_growth(self, gdp_data) -> float:
        """Calculate GDP growth rate from two quarterly values."""
        try:
            # If we got a list with 2 values, calculate quarter-over-quarter growth
            if isinstance(gdp_data, list) and len(gdp_data) >= 2:
                latest = float(gdp_data[0])
                previous = float(gdp_data[1])
                # Quarterly growth rate: ((latest - previous) / previous) * 100
                growth = ((latest - previous) / previous) * 100
                # Annualize it (multiply by 4 for quarterly data)
                annual_growth = growth * 4
                return round(annual_growth, 2)
            elif isinstance(gdp_data, (int, float)):
                # Single value - can't calculate growth, use default
                return 2.5
            else:
                return 2.5  # Default
        except Exception as e:
            logger.warning(f"GDP growth calculation failed: {e}")
            return 2.5

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key macro/sector metrics.

        These exact fields are required by the fusion engine's weighted averaging.
        """
        # Determine cycle score based on economic indicators
        gdp = data.get("gdp_growth", 2.5)
        unemployment = data.get("unemployment_rate", 3.8)
        fed_rate = data.get("fed_funds_rate", 5.33)

        # Simple cycle scoring: positive GDP + low unemployment = expansion
        cycle_score = 0.0
        if gdp > 2.0 and unemployment < 5.0:
            cycle_score = 0.6  # Expansion
        elif gdp < 0:
            cycle_score = -0.5  # Contraction
        elif unemployment > 6.0:
            cycle_score = -0.2  # Trough
        else:
            cycle_score = 0.2  # Stable

        # Sector score based on performance
        sector_perf = data.get("sector_performance", 0.0)
        sector_score = max(-1.0, min(1.0, sector_perf / 10.0))  # Normalize to -1 to 1

        return {
            "gdp_growth": gdp,  # Economic growth rate
            "unemployment_rate": unemployment,  # Labor market health
            "fed_funds_rate": fed_rate,  # Monetary policy stance
            "sector_performance": sector_perf,  # Sector vs market
            "cycle_score": cycle_score,  # Overall macro environment
            "sector_score": sector_score  # Sector relative strength
        }
