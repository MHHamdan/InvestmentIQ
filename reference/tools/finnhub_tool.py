"""
Finnhub Tool

MCP-compliant tool for accessing analyst ratings, price targets, and executive changes.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FinnhubTool:
    """
    Finnhub market data access tool.

    Retrieves analyst ratings, price targets, and executive changes.

    Data modes:
    - Sample mode (default): Uses data/samples/finnhub/*.json
    - Live mode: Connects to Finnhub API with rate limiting

    Finnhub Compliance:
    - Rate limit: 60 calls/minute (free tier)
    - Free for personal/research use
    - Commercial use requires paid plan
    """

    def __init__(self):
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        self.samples_dir = Path("data/samples/finnhub")
        self.base_url = "https://finnhub.io/api/v1/"

        logger.info(
            f"FinnhubTool initialized (mode: {'live' if self.live_mode else 'sample'})"
        )

    async def get_analyst_ratings(
        self,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Get analyst ratings and price targets for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ratings data and consensus
        """
        if self.live_mode and self.api_key:
            return await self._fetch_live_ratings(ticker)
        else:
            return self._fetch_sample_ratings(ticker)

    async def get_executive_changes(
        self,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Get executive leadership changes for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with executive changes and current leadership
        """
        if self.live_mode and self.api_key:
            return await self._fetch_live_execs(ticker)
        else:
            return self._fetch_sample_execs(ticker)

    def _fetch_sample_ratings(self, ticker: str) -> Dict[str, Any]:
        """Load sample ratings data from fixture."""
        sample_file = self.samples_dir / f"{ticker}_ratings.json"

        if not sample_file.exists():
            logger.warning(f"No sample ratings for {ticker}, returning empty")
            return self._get_empty_ratings(ticker)

        with open(sample_file, 'r') as f:
            data = json.load(f)

        logger.info(
            f"Loaded {len(data.get('ratings', []))} sample ratings for {ticker}"
        )
        return data

    def _fetch_sample_execs(self, ticker: str) -> Dict[str, Any]:
        """Load sample executive data from fixture."""
        sample_file = self.samples_dir / f"{ticker}_execs.json"

        if not sample_file.exists():
            logger.warning(f"No sample exec data for {ticker}, returning empty")
            return self._get_empty_execs(ticker)

        with open(sample_file, 'r') as f:
            data = json.load(f)

        logger.info(
            f"Loaded {len(data.get('executive_changes', []))} exec changes for {ticker}"
        )
        return data

    async def _fetch_live_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch live analyst ratings from Finnhub.

        Implementation would use:
        - GET /stock/recommendation with symbol parameter
        - Rate limiting (60 calls/min)
        - API key authentication
        """
        logger.warning(
            "Live Finnhub ratings fetching not implemented. "
            "Falling back to sample data."
        )
        return self._fetch_sample_ratings(ticker)

    async def _fetch_live_execs(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch live executive changes from Finnhub.

        Implementation would use:
        - GET /company-executive with symbol parameter
        - Rate limiting (60 calls/min)
        - API key authentication
        """
        logger.warning(
            "Live Finnhub exec data fetching not implemented. "
            "Falling back to sample data."
        )
        return self._fetch_sample_execs(ticker)

    def _get_empty_ratings(self, ticker: str) -> Dict[str, Any]:
        """Return empty ratings structure."""
        return {
            "company": ticker,
            "ratings": [],
            "consensus": {
                "avg_target": 0.0,
                "high_target": 0.0,
                "low_target": 0.0,
                "buy_count": 0,
                "hold_count": 0,
                "sell_count": 0,
                "total_analysts": 0
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _get_empty_execs(self, ticker: str) -> Dict[str, Any]:
        """Return empty executive data structure."""
        return {
            "company": ticker,
            "executive_changes": [],
            "current_leadership": [],
            "stability_score": 0.5,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def calculate_analyst_sentiment(
        self,
        ratings_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate sentiment from analyst ratings.

        Returns:
            Dictionary with sentiment scores and metrics
        """
        consensus = ratings_data.get("consensus", {})
        total_analysts = consensus.get("total_analysts", 0)

        if total_analysts == 0:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0
            }

        buy_count = consensus.get("buy_count", 0)
        hold_count = consensus.get("hold_count", 0)
        sell_count = consensus.get("sell_count", 0)

        # Calculate sentiment [-1, 1]
        # buy=1, hold=0, sell=-1
        sentiment = (buy_count - sell_count) / total_analysts

        # Confidence based on agreement
        max_count = max(buy_count, hold_count, sell_count)
        confidence = max_count / total_analysts

        return {
            "sentiment": round(sentiment, 3),
            "confidence": round(confidence, 3),
            "bullish_ratio": buy_count / total_analysts,
            "bearish_ratio": sell_count / total_analysts,
            "neutral_ratio": hold_count / total_analysts,
            "total_analysts": total_analysts
        }

    def analyze_rating_changes(
        self,
        ratings_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze recent rating changes (upgrades/downgrades).

        Returns:
            Dictionary with upgrade/downgrade counts and momentum
        """
        ratings = ratings_data.get("ratings", [])

        upgrades = 0
        downgrades = 0
        target_raises = 0
        target_cuts = 0

        rating_hierarchy = {
            "sell": 1,
            "underweight": 2,
            "hold": 3,
            "neutral": 3,
            "buy": 4,
            "overweight": 4,
            "outperform": 4
        }

        for rating in ratings:
            current = rating.get("rating", "").lower()
            previous = rating.get("rating_previous", "").lower()

            current_score = rating_hierarchy.get(current, 3)
            previous_score = rating_hierarchy.get(previous, 3)

            if current_score > previous_score:
                upgrades += 1
            elif current_score < previous_score:
                downgrades += 1

            current_target = rating.get("target", 0)
            previous_target = rating.get("target_previous", 0)

            if current_target > previous_target:
                target_raises += 1
            elif current_target < previous_target:
                target_cuts += 1

        # Momentum: positive if more upgrades, negative if more downgrades
        if upgrades + downgrades > 0:
            momentum = (upgrades - downgrades) / (upgrades + downgrades)
        else:
            momentum = 0.0

        return {
            "upgrades": upgrades,
            "downgrades": downgrades,
            "target_raises": target_raises,
            "target_cuts": target_cuts,
            "momentum": round(momentum, 3),
            "net_changes": upgrades - downgrades
        }

    def calculate_executive_stability(
        self,
        exec_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate executive stability metrics.

        Returns:
            Dictionary with stability score and change summary
        """
        changes = exec_data.get("executive_changes", [])
        current_leadership = exec_data.get("current_leadership", [])

        # Count departures and appointments
        departures = sum(
            1 for c in changes
            if c.get("type") in ["departure", "transition"]
        )

        # Calculate sentiment from changes
        change_sentiments = []
        for change in changes:
            sent = change.get("sentiment", "neutral")
            if sent == "positive":
                change_sentiments.append(1)
            elif sent == "negative":
                change_sentiments.append(-1)
            else:
                change_sentiments.append(0)

        avg_change_sentiment = (
            sum(change_sentiments) / len(change_sentiments)
            if change_sentiments else 0.0
        )

        # Stability score from fixture or calculate
        stability_score = exec_data.get("stability_score", 0.5)

        return {
            "stability_score": stability_score,
            "total_changes": len(changes),
            "departures": departures,
            "avg_change_sentiment": round(avg_change_sentiment, 3),
            "leadership_count": len(current_leadership),
            "recent_turnover": len(changes) > 0
        }

    def get_market_intelligence_summary(
        self,
        ticker: str,
        ratings_data: Dict[str, Any],
        exec_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive market intelligence summary.

        Returns:
            Summary combining analyst ratings and executive analysis
        """
        analyst_sentiment = self.calculate_analyst_sentiment(ratings_data)
        rating_changes = self.analyze_rating_changes(ratings_data)
        exec_stability = self.calculate_executive_stability(exec_data)

        consensus = ratings_data.get("consensus", {})

        return {
            "ticker": ticker,
            "analyst_metrics": {
                "consensus_sentiment": analyst_sentiment["sentiment"],
                "avg_price_target": consensus.get("avg_target", 0.0),
                "target_range": {
                    "high": consensus.get("high_target", 0.0),
                    "low": consensus.get("low_target", 0.0)
                },
                "total_analysts": consensus.get("total_analysts", 0),
                "bullish_ratio": analyst_sentiment["bullish_ratio"],
                "rating_momentum": rating_changes["momentum"],
                "recent_upgrades": rating_changes["upgrades"],
                "recent_downgrades": rating_changes["downgrades"]
            },
            "executive_metrics": {
                "stability_score": exec_stability["stability_score"],
                "recent_turnover": exec_stability["recent_turnover"],
                "total_changes": exec_stability["total_changes"],
                "leadership_count": exec_stability["leadership_count"]
            },
            "overall_sentiment": round(
                0.7 * analyst_sentiment["sentiment"] +
                0.3 * exec_stability["avg_change_sentiment"],
                3
            ),
            "timestamp": ratings_data.get("timestamp", datetime.utcnow().isoformat())
        }
