"""
Workforce Intelligence Agent

Analyzes workforce signals including employee sentiment, hiring trends,
and organizational health metrics.
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


class WorkforceIntelligenceAgent:
    """
    Analyzes workforce and organizational health signals.

    Data sources:
    - Sample mode: data/samples/workforce/*.json
    - Live mode: Compliant public sources only (no scraping)
    """

    def __init__(self, agent_id: str = "workforce_intelligence"):
        self.agent_id = agent_id
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/workforce")
        self.agent_bus = get_agent_bus()

        logger.info(
            f"WorkforceIntelligenceAgent initialized (mode: "
            f"{'live' if self.live_mode else 'sample'})"
        )

    @trace_agent("workforce_intelligence", {"version": "1.0"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None,
        domain: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze workforce signals for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Industry sector (optional)
            domain: Company domain for job searches (optional)

        Returns:
            AgentOutput with workforce metrics and sentiment
        """
        logger.info(f"Analyzing workforce signals for {ticker}")

        # Fetch workforce data
        if self.live_mode:
            data = await self._fetch_live_data(ticker, company_name, domain)
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

        # Broadcast alerts if any
        for alert in alerts:
            self.agent_bus.broadcast_alert(alert)

        output = AgentOutput(
            signal=SignalType.WORKFORCE,
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
                "review_count": data.get("review_count", 0)
            }
        )

        logger.info(
            f"Workforce analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample workforce data from fixture."""
        sample_file = self.samples_dir / f"{ticker}_workforce.json"

        if not sample_file.exists():
            logger.warning(f"No sample data for {ticker}, using default")
            return self._get_default_data(ticker)

        with open(sample_file, 'r') as f:
            return json.load(f)

    async def _fetch_live_data(
        self,
        ticker: str,
        company_name: str,
        domain: Optional[str]
    ) -> Dict[str, Any]:
        """
        Fetch live workforce data from compliant sources.

        Note: Currently not implemented. Would require official API
        partnerships with job boards, review platforms, etc.
        """
        logger.warning(
            "Live workforce data not available. "
            "Falling back to sample data."
        )
        return self._fetch_sample_data(ticker)

    def _get_default_data(self, ticker: str) -> Dict[str, Any]:
        """Return default neutral workforce data."""
        return {
            "company": ticker,
            "company_name": f"{ticker} Inc.",
            "rating": 3.5,
            "rating_trend": "stable",
            "rating_history": [{"date": "2025-01-01", "rating": 3.5}],
            "review_count": 0,
            "topics": {
                "work_life_balance": 0.2,
                "compensation": 0.2,
                "culture": 0.2,
                "management": 0.2,
                "career": 0.2
            },
            "hiring_signal": "unknown",
            "job_postings_count": 0,
            "job_postings_trend": "stable",
            "job_postings_history": [],
            "tenure_metrics": {
                "avg_tenure_years": 0,
                "churn_rate": 0,
                "trend": "unknown"
            },
            "sentiment_breakdown": {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured metrics from workforce data."""
        return {
            "employee_rating": data.get("rating", 0),
            "rating_trend": data.get("rating_trend", "unknown"),
            "review_count": data.get("review_count", 0),
            "hiring_signal": data.get("hiring_signal", "unknown"),
            "job_postings_count": data.get("job_postings_count", 0),
            "job_postings_trend": data.get("job_postings_trend", "unknown"),
            "avg_tenure_years": data.get("tenure_metrics", {}).get("avg_tenure_years", 0),
            "churn_rate": data.get("tenure_metrics", {}).get("churn_rate", 0),
            "churn_trend": data.get("tenure_metrics", {}).get("trend", "unknown"),
            "top_topic": self._get_top_topic(data.get("topics", {})),
            "topic_distribution": data.get("topics", {})
        }

    def _get_top_topic(self, topics: Dict[str, float]) -> str:
        """Get most discussed topic from distribution."""
        if not topics:
            return "unknown"
        return max(topics.items(), key=lambda x: x[1])[0]

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """
        Calculate overall workforce sentiment [-1, 1].

        Combines employee rating, sentiment breakdown, and trends.
        """
        # Normalize rating to [-1, 1] (assuming 1-5 scale)
        rating = data.get("rating", 3.0)
        rating_sentiment = (rating - 3.0) / 2.0  # 5→1.0, 3→0, 1→-1.0

        # Sentiment breakdown
        breakdown = data.get("sentiment_breakdown", {})
        pos = breakdown.get("positive", 0.33)
        neg = breakdown.get("negative", 0.33)
        breakdown_sentiment = pos - neg  # [-1, 1]

        # Trend adjustments
        trend_boost = 0.0
        rating_trend = data.get("rating_trend", "stable")
        if rating_trend == "increasing":
            trend_boost = 0.1
        elif rating_trend == "declining":
            trend_boost = -0.1

        hiring_trend = data.get("job_postings_trend", "stable")
        if hiring_trend == "increasing":
            trend_boost += 0.05
        elif hiring_trend == "decreasing":
            trend_boost -= 0.05

        # Weighted combination
        sentiment = (
            0.5 * rating_sentiment +
            0.3 * breakdown_sentiment +
            0.2 * trend_boost
        )

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """
        Calculate confidence in workforce assessment.

        Higher confidence with more reviews and stable trends.
        """
        review_count = data.get("review_count", 0)

        # Base confidence from sample size
        if review_count > 10000:
            base_confidence = 0.9
        elif review_count > 5000:
            base_confidence = 0.8
        elif review_count > 1000:
            base_confidence = 0.7
        elif review_count > 100:
            base_confidence = 0.6
        else:
            base_confidence = 0.4

        # Reduce confidence for volatile trends
        rating_history = data.get("rating_history", [])
        if len(rating_history) >= 3:
            ratings = [h["rating"] for h in rating_history[-3:]]
            volatility = max(ratings) - min(ratings)
            if volatility > 0.5:
                base_confidence *= 0.85

        return round(base_confidence, 2)

    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        """Build evidence list from workforce data."""
        evidence = []

        # Employee rating
        evidence.append(Evidence(
            source="employee_reviews",
            value=data.get("rating", 0),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "")
            ),
            confidence=0.8,
            description=f"Employee rating: {data.get('rating', 0)}/5.0 "
                       f"({data.get('review_count', 0)} reviews)"
        ))

        # Hiring signal
        hiring_signal = data.get("hiring_signal", "unknown")
        evidence.append(Evidence(
            source="job_postings",
            value=hiring_signal,
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "")
            ),
            confidence=0.7,
            description=f"Hiring signal: {hiring_signal} "
                       f"({data.get('job_postings_count', 0)} active postings)"
        ))

        # Churn rate
        churn_rate = data.get("tenure_metrics", {}).get("churn_rate", 0)
        if churn_rate > 0:
            evidence.append(Evidence(
                source="tenure_metrics",
                value=churn_rate,
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "")
                ),
                confidence=0.6,
                description=f"Churn rate: {churn_rate:.1%}"
            ))

        return evidence

    def _generate_alerts(
        self,
        data: Dict[str, Any],
        ticker: str
    ) -> List[Alert]:
        """Generate alerts for significant workforce signals."""
        alerts = []

        # Low rating alert
        rating = data.get("rating", 3.0)
        if rating < 3.0:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high" if rating < 2.5 else "medium",
                title="Low Employee Satisfaction",
                description=f"Employee rating at {rating}/5.0, "
                           f"below industry average",
                recommended_action="Investigate recent organizational changes"
            ))

        # Hiring freeze alert
        hiring_signal = data.get("hiring_signal", "unknown")
        if hiring_signal == "freeze":
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Hiring Freeze Detected",
                description="Job postings declining significantly",
                recommended_action="Monitor for restructuring announcements"
            ))

        # High churn alert
        churn_rate = data.get("tenure_metrics", {}).get("churn_rate", 0)
        if churn_rate > 0.20:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high" if churn_rate > 0.30 else "medium",
                title="Elevated Employee Churn",
                description=f"Churn rate at {churn_rate:.1%}, "
                           f"above healthy threshold",
                recommended_action="Review retention strategies and culture"
            ))

        # Declining trend alert
        rating_trend = data.get("rating_trend", "stable")
        if rating_trend == "declining":
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Declining Employee Sentiment",
                description="Employee rating trending downward",
                recommended_action="Monitor for impact on productivity"
            ))

        return alerts

    def _generate_summary(
        self,
        data: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of workforce analysis."""
        rating = data.get("rating", 0)
        trend = data.get("rating_trend", "unknown")
        hiring = data.get("hiring_signal", "unknown")
        top_topic = metrics.get("top_topic", "unknown")

        return (
            f"Employee rating: {rating}/5.0 ({trend}). "
            f"Hiring: {hiring}. "
            f"Top concern: {top_topic.replace('_', ' ')}"
        )
