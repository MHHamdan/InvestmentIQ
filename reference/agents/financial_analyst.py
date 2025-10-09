"""
Financial Analyst Agent

Analyzes financial health, ratios, and reporting quality.

MODIFIED: 2025-10-06 - MURTHY - Added FMP integration for real financial data
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.agent_contracts import (
    AgentOutput,
    SignalType,
    Evidence,
    Observation,
    Alert
)
from core.agent_bus import get_agent_bus
from utils.observability import trace_agent

# NEW: Import FMP tool for real financial data
from tools.fmp_tool import FMPTool

logger = logging.getLogger(__name__)


class FinancialAnalystAgent:
    """
    Analyzes financial signals including revenue, margins, and debt levels.

    Data sources:
    - Sample mode: data/samples/financial/*.json
    - Live mode: Financial APIs
    """

    def __init__(self, agent_id: str = "financial_analyst"):
        # EXISTING: Original initialization
        self.agent_id = agent_id
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/financial")
        self.agent_bus = get_agent_bus()

        # NEW: Initialize FMP tool for real financial data
        self.fmp_tool = FMPTool()
        self.use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"

        logger.info(
            f"FinancialAnalystAgent initialized (mode: "
            f"{'FMP' if self.use_fmp else 'live' if self.live_mode else 'sample'})"
        )

    @trace_agent("financial_analyst", {"version": "1.0"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze financial signals for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Industry sector

        Returns:
            AgentOutput with financial metrics and sentiment
        """
        logger.info(f"Analyzing financial signals for {ticker}")

        # MODIFIED: Fetch financial data with FMP support
        # Priority: FMP (if enabled) → Live → Sample
        if self.use_fmp:
            # NEW: Try FMP first
            data = await self._fetch_fmp_data(ticker)
        elif self.live_mode:
            # EXISTING: Live mode fallback
            data = await self._fetch_live_data(ticker)
        else:
            # EXISTING: Sample data fallback
            data = self._fetch_sample_data(ticker)

        # Extract metrics
        metrics = self._extract_metrics(data)

        # Calculate sentiment
        sentiment = self._calculate_sentiment(data)

        # Calculate confidence
        confidence = self._calculate_confidence(data)

        # Build evidence
        evidence = self._build_evidence(data)

        # Generate alerts
        alerts = self._generate_alerts(data, ticker)

        # Broadcast observation
        observation = Observation(
            agent_id=self.agent_id,
            ticker=ticker,
            observation=self._generate_summary(data, metrics),
            data={"metrics": metrics, "sentiment": sentiment},
            confidence=confidence
        )
        self.agent_bus.broadcast_observation(observation)

        # Broadcast alerts
        for alert in alerts:
            self.agent_bus.broadcast_alert(alert)

        output = AgentOutput(
            signal=SignalType.FINANCIAL,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,
            sentiment=sentiment,
            confidence=confidence,
            evidence=evidence,
            alerts=[a.title for a in alerts],
            metadata={
                "sector": sector,
                # MODIFIED: Track data source including FMP
                "data_source": "fmp" if self.use_fmp else "live" if self.live_mode else "sample"
            }
        )

        logger.info(
            f"Financial analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    # NEW: FMP data fetching method added 2025-01-06
    async def _fetch_fmp_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch real financial data from Financial Modeling Prep API.

        Falls back to sample data if FMP fails.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with financial metrics
        """
        try:
            logger.info(f"Fetching FMP data for {ticker}")
            data = await self.fmp_tool.get_financial_ratios(ticker)
            logger.info(f"✅ FMP data retrieved for {ticker}")
            return data
        except Exception as e:
            logger.warning(f"⚠️ FMP fetch failed for {ticker}: {e}")
            logger.info(f"Falling back to sample data for {ticker}")
            # NEW: Automatic fallback to sample data on error
            return self._fetch_sample_data(ticker)

    # EXISTING: Original sample data method (unchanged)
    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample financial data."""
        sample_file = self.samples_dir / f"{ticker.lower()}_financial.json"

        if sample_file.exists():
            with open(sample_file) as f:
                return json.load(f)

        # Default sample data
        return {
            "ticker": ticker,
            "revenue_growth": 0.15,
            "gross_margin": 0.42,
            "operating_margin": 0.25,
            "net_margin": 0.20,
            "debt_to_equity": 0.45,
            "current_ratio": 1.8,
            "roe": 0.22,
            "cash_flow_positive": True
        }

    # EXISTING: Original live data method (unchanged, for future use)
    async def _fetch_live_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch live financial data from APIs."""
        # Would integrate with financial APIs here
        return self._fetch_sample_data(ticker)

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics."""
        return {
            "revenue_growth": data.get("revenue_growth", 0),
            "gross_margin": data.get("gross_margin", 0),
            "operating_margin": data.get("operating_margin", 0),
            "net_margin": data.get("net_margin", 0),
            "debt_to_equity": data.get("debt_to_equity", 0),
            "roe": data.get("roe", 0)
        }

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate financial sentiment score."""
        score = 0.0

        # Revenue growth
        if data.get("revenue_growth", 0) > 0.15:
            score += 0.3
        elif data.get("revenue_growth", 0) > 0:
            score += 0.1

        # Margins
        if data.get("gross_margin", 0) > 0.4:
            score += 0.2
        if data.get("operating_margin", 0) > 0.2:
            score += 0.2

        # Debt
        if data.get("debt_to_equity", 1.0) < 0.5:
            score += 0.2

        # ROE
        if data.get("roe", 0) > 0.15:
            score += 0.1

        return min(score, 1.0)

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis."""
        return 0.85  # High confidence for financial data

    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        """Build evidence list."""
        evidence = []

        if data.get("revenue_growth", 0) > 0.1:
            evidence.append(Evidence(
                source="financial_data",
                value={"revenue_growth": data["revenue_growth"]},
                timestamp=datetime.utcnow(),
                description=f"Strong revenue growth of {data['revenue_growth']*100:.1f}%",
                confidence=0.9
            ))

        if data.get("gross_margin", 0) > 0.35:
            evidence.append(Evidence(
                source="financial_data",
                value={"gross_margin": data["gross_margin"]},
                timestamp=datetime.utcnow(),
                description=f"Healthy gross margin of {data['gross_margin']*100:.1f}%",
                confidence=0.85
            ))

        return evidence

    def _generate_alerts(self, data: Dict[str, Any], ticker: str) -> List[Alert]:
        """Generate alerts based on financial data."""
        alerts = []

        if data.get("debt_to_equity", 0) > 1.5:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="High Debt Levels",
                message=f"Debt-to-equity ratio of {data['debt_to_equity']:.2f} is concerning",
                data={"debt_to_equity": data["debt_to_equity"]}
            ))

        return alerts

    def _generate_summary(self, data: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Generate analysis summary."""
        return (
            f"Financial health shows {metrics['revenue_growth']*100:.1f}% revenue growth "
            f"with {metrics['gross_margin']*100:.1f}% gross margin"
        )
