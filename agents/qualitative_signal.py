"""
Qualitative Signal Agent

Analyzes qualitative signals including news sentiment, market perception,
and organizational reputation.
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


class QualitativeSignalAgent:
    """
    Analyzes qualitative signals including sentiment, news, and reputation.

    Data sources:
    - Sample mode: data/samples/qualitative/*.json
    - Live mode: News APIs and sentiment analysis services
    """

    def __init__(self, agent_id: str = "qualitative_signal"):
        self.agent_id = agent_id
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/qualitative")
        self.agent_bus = get_agent_bus()

        logger.info(
            f"QualitativeSignalAgent initialized (mode: "
            f"{'live' if self.live_mode else 'sample'})"
        )

    @trace_agent("qualitative_signal", {"version": "1.0"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze qualitative signals for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Industry sector

        Returns:
            AgentOutput with qualitative metrics and sentiment
        """
        logger.info(f"Analyzing qualitative signals for {ticker}")

        # Fetch qualitative data
        if self.live_mode:
            data = await self._fetch_live_data(ticker, company_name)
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
            signal=SignalType.SENTIMENT,
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
                "news_count": data.get("news_count", 0)
            }
        )

        logger.info(
            f"Qualitative analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample qualitative data."""
        sample_file = self.samples_dir / f"{ticker.lower()}_qualitative.json"

        if sample_file.exists():
            with open(sample_file) as f:
                return json.load(f)

        # Default sample data
        return {
            "ticker": ticker,
            "overall_sentiment": "Neutral",
            "sentiment_score": 0.0,
            "news_count": 0,
            "key_themes": [],
            "sentiment_breakdown": {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33
            },
            "reputation_score": 0.5,
            "market_perception": "neutral",
            "recent_news": []
        }

    async def _fetch_live_data(
        self,
        ticker: str,
        company_name: str
    ) -> Dict[str, Any]:
        """Fetch live qualitative data from APIs."""
        # Would integrate with news and sentiment APIs here
        return self._fetch_sample_data(ticker)

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics."""
        return {
            "sentiment_score": data.get("sentiment_score", 0),
            "reputation_score": data.get("reputation_score", 0.5),
            "news_count": data.get("news_count", 0),
            "positive_ratio": data.get("sentiment_breakdown", {}).get("positive", 0.33),
            "negative_ratio": data.get("sentiment_breakdown", {}).get("negative", 0.33),
            "theme_count": len(data.get("key_themes", []))
        }

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate qualitative sentiment score."""
        # Base sentiment from sentiment score
        sentiment_score = data.get("sentiment_score", 0)

        # Sentiment breakdown
        breakdown = data.get("sentiment_breakdown", {})
        pos = breakdown.get("positive", 0.33)
        neg = breakdown.get("negative", 0.33)
        breakdown_sentiment = pos - neg

        # Reputation score
        reputation = data.get("reputation_score", 0.5)
        reputation_sentiment = (reputation - 0.5) * 2  # Convert to [-1, 1]

        # Market perception
        perception = data.get("market_perception", "neutral")
        perception_boost = 0.0
        if perception == "positive":
            perception_boost = 0.2
        elif perception == "negative":
            perception_boost = -0.2

        # Weighted combination
        sentiment = (
            0.4 * sentiment_score +
            0.3 * breakdown_sentiment +
            0.2 * reputation_sentiment +
            0.1 * perception_boost
        )

        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis."""
        news_count = data.get("news_count", 0)

        # Base confidence from data volume
        if news_count > 50:
            confidence = 0.85
        elif news_count > 20:
            confidence = 0.75
        elif news_count > 5:
            confidence = 0.65
        else:
            confidence = 0.50

        return confidence

    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        """Build evidence list."""
        evidence = []

        # Sentiment evidence
        sentiment_score = data.get("sentiment_score", 0)
        if abs(sentiment_score) > 0.3:
            evidence.append(Evidence(
                source="sentiment_analysis",
                value={"sentiment_score": sentiment_score},
                timestamp=datetime.utcnow(),
                description=f"Sentiment score of {sentiment_score:.2f} indicates "
                           f"{'positive' if sentiment_score > 0 else 'negative'} market perception",
                confidence=0.8
            ))

        # News volume evidence
        news_count = data.get("news_count", 0)
        if news_count > 0:
            evidence.append(Evidence(
                source="news_analysis",
                value={"news_count": news_count},
                timestamp=datetime.utcnow(),
                description=f"Analyzed {news_count} news articles",
                confidence=0.7
            ))

        # Theme evidence
        themes = data.get("key_themes", [])
        if themes:
            evidence.append(Evidence(
                source="theme_analysis",
                value={"themes": themes},
                timestamp=datetime.utcnow(),
                description=f"Key themes: {', '.join(themes)}",
                confidence=0.75
            ))

        return evidence

    def _generate_alerts(self, data: Dict[str, Any], ticker: str) -> List[Alert]:
        """Generate alerts based on qualitative data."""
        alerts = []

        # Negative sentiment alert
        sentiment_score = data.get("sentiment_score", 0)
        if sentiment_score < -0.5:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="Negative Market Sentiment",
                message=f"Sentiment score of {sentiment_score:.2f} indicates strong negative perception",
                data={"sentiment_score": sentiment_score}
            ))

        # Reputation crisis
        reputation = data.get("reputation_score", 0.5)
        if reputation < 0.3:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="Reputation Risk",
                message=f"Low reputation score of {reputation:.2f}",
                data={"reputation_score": reputation}
            ))

        # Theme-based alerts
        themes = data.get("key_themes", [])
        risk_themes = ["scandal", "crisis", "investigation", "lawsuit", "layoff"]
        detected_risks = [t for t in themes if any(r in t.lower() for r in risk_themes)]

        if detected_risks:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Risk Themes Detected",
                message=f"Concerning themes identified: {', '.join(detected_risks)}",
                data={"themes": detected_risks}
            ))

        return alerts

    def _generate_summary(self, data: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Generate analysis summary."""
        sentiment_label = data.get("overall_sentiment", "Neutral")
        news_count = data.get("news_count", 0)
        perception = data.get("market_perception", "neutral")

        return (
            f"Market sentiment: {sentiment_label} based on {news_count} news sources. "
            f"Overall perception: {perception}"
        )
