"""
Context Engine Agent

Analyzes contextual patterns and historical correlations to provide
investment recommendations based on similar past scenarios.
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

logger = logging.getLogger(__name__)


class ContextEngineAgent:
    """
    Analyzes contextual patterns and historical correlations.

    Data sources:
    - Sample mode: data/samples/context/*.json
    - Live mode: Historical market data and pattern databases
    """

    def __init__(self, agent_id: str = "context_engine"):
        self.agent_id = agent_id
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/context")
        self.agent_bus = get_agent_bus()

        logger.info(
            f"ContextEngineAgent initialized (mode: "
            f"{'live' if self.live_mode else 'sample'})"
        )

    @trace_agent("context_engine", {"version": "1.0"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze contextual patterns for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Industry sector

        Returns:
            AgentOutput with context analysis and recommendations
        """
        logger.info(f"Analyzing context patterns for {ticker}")

        # Fetch context data
        if self.live_mode:
            data = await self._fetch_live_data(ticker, company_name, sector)
        else:
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
            signal=SignalType.CONTEXT,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,
            sentiment=sentiment,
            confidence=confidence,
            evidence=evidence,
            alerts=[a.title for a in alerts],
            metadata={
                "sector": sector,
                "data_source": "live" if self.live_mode else "sample",
                "pattern_matches": data.get("pattern_matches", 0)
            }
        )

        logger.info(
            f"Context analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample context data."""
        sample_file = self.samples_dir / f"{ticker.lower()}_context.json"

        if sample_file.exists():
            with open(sample_file) as f:
                return json.load(f)

        # Default sample data
        return {
            "ticker": ticker,
            "scenario_type": "normal",
            "pattern_matches": 0,
            "historical_accuracy": 0.75,
            "similar_situations": [],
            "market_cycle": "expansion",
            "sector_trend": "stable",
            "contrarian_signal": False,
            "pattern_confidence": 0.7
        }

    async def _fetch_live_data(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str]
    ) -> Dict[str, Any]:
        """Fetch live context data from pattern databases."""
        # Would integrate with historical pattern databases here
        return self._fetch_sample_data(ticker)

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics."""
        return {
            "pattern_matches": data.get("pattern_matches", 0),
            "historical_accuracy": data.get("historical_accuracy", 0.75),
            "pattern_confidence": data.get("pattern_confidence", 0.7),
            "similar_count": len(data.get("similar_situations", [])),
            "contrarian_score": 1.0 if data.get("contrarian_signal") else 0.0
        }

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate context-based sentiment score."""
        # Base sentiment from scenario type
        scenario_type = data.get("scenario_type", "normal")
        scenario_sentiment = {
            "contrarian_opportunity": 0.6,
            "strong_momentum": 0.7,
            "risk_warning": -0.6,
            "normal": 0.0,
            "uncertain": 0.0
        }.get(scenario_type, 0.0)

        # Market cycle influence
        market_cycle = data.get("market_cycle", "expansion")
        cycle_boost = {
            "expansion": 0.2,
            "peak": 0.0,
            "contraction": -0.2,
            "trough": -0.1
        }.get(market_cycle, 0.0)

        # Sector trend
        sector_trend = data.get("sector_trend", "stable")
        trend_boost = {
            "bullish": 0.15,
            "stable": 0.0,
            "bearish": -0.15
        }.get(sector_trend, 0.0)

        # Contrarian signal adjustment
        contrarian_adjustment = 0.0
        if data.get("contrarian_signal"):
            contrarian_adjustment = 0.3

        # Weighted combination
        sentiment = (
            0.4 * scenario_sentiment +
            0.2 * cycle_boost +
            0.2 * trend_boost +
            0.2 * contrarian_adjustment
        )

        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis."""
        pattern_matches = data.get("pattern_matches", 0)
        historical_accuracy = data.get("historical_accuracy", 0.75)

        # Base confidence from pattern matches
        if pattern_matches > 10:
            base_confidence = 0.9
        elif pattern_matches > 5:
            base_confidence = 0.8
        elif pattern_matches > 2:
            base_confidence = 0.7
        else:
            base_confidence = 0.6

        # Adjust by historical accuracy
        confidence = base_confidence * historical_accuracy

        return round(confidence, 2)

    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        """Build evidence list."""
        evidence = []

        # Pattern matches
        pattern_matches = data.get("pattern_matches", 0)
        if pattern_matches > 0:
            evidence.append(Evidence(
                source="pattern_analysis",
                value={"pattern_matches": pattern_matches},
                timestamp=datetime.utcnow(),
                description=f"Found {pattern_matches} similar historical patterns",
                confidence=0.85
            ))

        # Historical accuracy
        historical_accuracy = data.get("historical_accuracy", 0.75)
        evidence.append(Evidence(
            source="historical_data",
            value={"accuracy": historical_accuracy},
            timestamp=datetime.utcnow(),
            description=f"Pattern accuracy: {historical_accuracy*100:.0f}%",
            confidence=0.8
        ))

        # Similar situations
        similar_situations = data.get("similar_situations", [])
        if similar_situations:
            evidence.append(Evidence(
                source="similarity_analysis",
                value={"similar_count": len(similar_situations)},
                timestamp=datetime.utcnow(),
                description=f"Identified {len(similar_situations)} similar past scenarios",
                confidence=0.75
            ))

        return evidence

    def _generate_alerts(self, data: Dict[str, Any], ticker: str) -> List[Alert]:
        """Generate alerts based on context analysis."""
        alerts = []

        # Contrarian opportunity
        if data.get("contrarian_signal"):
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Contrarian Opportunity Detected",
                message="Historical patterns suggest market may be overreacting",
                data={"scenario_type": data.get("scenario_type")}
            ))

        # Risk warning scenario
        if data.get("scenario_type") == "risk_warning":
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="Historical Risk Pattern",
                message="Current situation matches past negative scenarios",
                data={"pattern_matches": data.get("pattern_matches", 0)}
            ))

        # Low confidence in patterns
        pattern_confidence = data.get("pattern_confidence", 0.7)
        if pattern_confidence < 0.5:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="low",
                title="Low Pattern Confidence",
                message="Limited historical data for this scenario",
                data={"confidence": pattern_confidence}
            ))

        return alerts

    def _generate_summary(self, data: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Generate analysis summary."""
        scenario = data.get("scenario_type", "normal")
        matches = data.get("pattern_matches", 0)
        accuracy = data.get("historical_accuracy", 0.75)

        return (
            f"Scenario: {scenario}. Found {matches} historical matches "
            f"with {accuracy*100:.0f}% accuracy rate"
        )
