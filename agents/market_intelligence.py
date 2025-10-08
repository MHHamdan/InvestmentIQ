"""
Market Intelligence Agent

Analyzes market signals including analyst ratings, news sentiment,
SEC filings, and executive changes.

MODIFIED: 2025-10-06 - MURTHY - Added FMP integration for analyst ratings and price targets
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.agent_contracts import (
    AgentOutput,
    SignalType,
    Evidence,
    Observation,
    Alert
)
from core.agent_bus import get_agent_bus
from tools.edgar_tool import EdgarTool
from tools.news_api_tool import NewsAPITool
from tools.finnhub_tool import FinnhubTool
# NEW: Import FMP tool for real analyst ratings and price targets
from tools.fmp_tool import FMPTool
from utils.observability import trace_agent

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent:
    """
    Analyzes market intelligence signals.

    Data sources:
    - SEC EDGAR: Material event filings (8-K)
    - NewsAPI: Company news and sentiment
    - Finnhub: Analyst ratings, targets, executive changes
    """

    def __init__(self, agent_id: str = "market_intelligence"):
        # EXISTING: Original initialization
        self.agent_id = agent_id
        self.edgar_tool = EdgarTool()
        self.news_tool = NewsAPITool()
        self.finnhub_tool = FinnhubTool()
        self.agent_bus = get_agent_bus()

        # NEW: Initialize FMP tool for real analyst data
        self.fmp_tool = FMPTool()
        self.use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"

        mode = "FMP" if self.use_fmp else "sample"
        logger.info(f"MarketIntelligenceAgent initialized (mode: {mode})")

    @trace_agent("market_intelligence", {"version": "1.0"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze market intelligence signals for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Industry sector (optional)

        Returns:
            AgentOutput with market intelligence metrics and sentiment
        """
        logger.info(f"Analyzing market intelligence for {ticker}")

        # MODIFIED: Fetch data from all sources with FMP support
        filings_data = await self.edgar_tool.get_recent_filings(ticker)
        news_data = await self.news_tool.get_company_news(ticker, company_name)

        # NEW: Use FMP for analyst ratings if enabled, otherwise use Finnhub
        if self.use_fmp:
            ratings_data = await self._fetch_fmp_ratings(ticker)
        else:
            ratings_data = await self.finnhub_tool.get_analyst_ratings(ticker)

        exec_data = await self.finnhub_tool.get_executive_changes(ticker)

        # Analyze each data source
        filing_analysis = self._analyze_filings(filings_data)
        news_analysis = self._analyze_news(news_data)
        analyst_analysis = self._analyze_analysts(ratings_data)
        exec_analysis = self._analyze_executives(exec_data)

        # Combine metrics
        metrics = {
            "sec_filings": filing_analysis["metrics"],
            "news_sentiment": news_analysis["metrics"],
            "analyst_ratings": analyst_analysis["metrics"],
            "executive_stability": exec_analysis["metrics"]
        }

        # Calculate aggregate sentiment
        sentiment = self._calculate_aggregate_sentiment(
            filing_analysis, news_analysis, analyst_analysis, exec_analysis
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            filing_analysis, news_analysis, analyst_analysis, exec_analysis
        )

        # Build evidence
        evidence = self._build_evidence(
            filing_analysis, news_analysis, analyst_analysis, exec_analysis
        )

        # Generate alerts
        alerts = self._generate_alerts(
            ticker,
            filing_analysis,
            news_analysis,
            analyst_analysis,
            exec_analysis
        )

        # Broadcast observation
        observation = Observation(
            agent_id=self.agent_id,
            ticker=ticker,
            observation=self._generate_summary(metrics, sentiment),
            data={"metrics": metrics, "sentiment": sentiment},
            confidence=confidence
        )
        self.agent_bus.broadcast_observation(observation)

        # Broadcast alerts
        for alert in alerts:
            self.agent_bus.broadcast_alert(alert)

        # MODIFIED: Update metadata to track data source
        sources = ["edgar", "news_api"]
        if self.use_fmp:
            sources.append("fmp")
        else:
            sources.append("finnhub")

        output = AgentOutput(
            signal=SignalType.MARKET_INTELLIGENCE,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,
            sentiment=sentiment,
            confidence=confidence,
            evidence=evidence,
            alerts=[a.title for a in alerts],
            metadata={
                "sector": sector,
                "sources": sources,
                "data_source": "fmp" if self.use_fmp else "sample"
            }
        )

        logger.info(
            f"Market intelligence analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    async def _fetch_fmp_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        NEW: Fetch analyst ratings and price targets from FMP.

        Returns data in Finnhub-compatible format for existing analysis methods.
        """
        try:
            logger.info(f"Fetching FMP analyst data for {ticker}")

            # MURTHY ADDED 2025-10-07 - Fetch current price quote along with ratings and targets
            # Fetch analyst ratings, price target consensus, and current price
            ratings = await self.fmp_tool.get_analyst_ratings(ticker)
            price_targets = await self.fmp_tool.get_price_target_consensus(ticker)
            quote = await self.fmp_tool.get_quote(ticker)

            # Convert FMP format to Finnhub-compatible format
            # FMP returns: ratings (list), upgrades, downgrades, momentum
            # Price targets: avg_target, high_target, low_target, median_target

            fmp_ratings = ratings.get("ratings", [])

            # Calculate consensus from FMP ratings
            if fmp_ratings:
                # Count rating types
                buy_count = sum(1 for r in fmp_ratings if "buy" in r.get("newGrade", "").lower())
                sell_count = sum(1 for r in fmp_ratings if "sell" in r.get("newGrade", "").lower())
                hold_count = len(fmp_ratings) - buy_count - sell_count
                total_analysts = len(fmp_ratings)

                consensus = {
                    "buy": buy_count,
                    "hold": hold_count,
                    "sell": sell_count,
                    "buy_count": buy_count,
                    "hold_count": hold_count,
                    "sell_count": sell_count,
                    "total_analysts": total_analysts,
                    "strongBuy": sum(1 for r in fmp_ratings if "strong buy" in r.get("newGrade", "").lower()),
                    "strongSell": sum(1 for r in fmp_ratings if "strong sell" in r.get("newGrade", "").lower()),
                    "avg_target": price_targets.get("avg_target", 0.0),
                    "high_target": price_targets.get("high_target", 0.0),
                    "low_target": price_targets.get("low_target", 0.0),
                    "current_price": quote.get("price", 0.0)  # MURTHY ADDED 2025-10-07
                }
            else:
                # No ratings available
                consensus = {
                    "buy": 0, "hold": 0, "sell": 0,
                    "buy_count": 0, "hold_count": 0, "sell_count": 0,
                    "total_analysts": 0,
                    "strongBuy": 0, "strongSell": 0,
                    "avg_target": price_targets.get("avg_target", 0.0),
                    "high_target": price_targets.get("high_target", 0.0),
                    "low_target": price_targets.get("low_target", 0.0),
                    "current_price": quote.get("price", 0.0)  # MURTHY ADDED 2025-10-07
                }

            result = {
                "ratings": fmp_ratings,
                "consensus": consensus,
                "upgrades": ratings.get("upgrades", 0),
                "downgrades": ratings.get("downgrades", 0),
                "momentum": ratings.get("momentum", 0.0)
            }

            logger.info(f"✅ FMP analyst data retrieved for {ticker}")
            return result

        except Exception as e:
            logger.warning(f"⚠️ FMP analyst fetch failed for {ticker}: {e}, using fallback")
            # Fallback to Finnhub
            return await self.finnhub_tool.get_analyst_ratings(ticker)

    def _analyze_filings(self, filings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SEC filings."""
        events = self.edgar_tool.extract_material_events(filings_data)
        categorized = self.edgar_tool.categorize_events(events)
        sentiment = self.edgar_tool.calculate_event_sentiment(events)

        return {
            "metrics": {
                "total_filings": len(filings_data.get("filings", [])),
                "total_events": len(events),
                "financial_events": len(categorized["financial_results"]),
                "leadership_events": len(categorized["leadership_changes"]),
                "strategic_events": len(categorized["strategic_actions"])
            },
            "sentiment": sentiment,
            "events": events[:5],  # Top 5 recent
            "data_quality": 0.8 if len(events) > 0 else 0.3
        }

    def _analyze_news(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze news articles."""
        sentiment_metrics = self.news_tool.calculate_sentiment_metrics(news_data)
        categorized = self.news_tool.categorize_articles(news_data)
        top_stories = self.news_tool.get_top_stories(news_data, limit=3)

        return {
            "metrics": {
                "total_articles": sentiment_metrics["article_count"],
                "avg_sentiment": sentiment_metrics["avg_sentiment"],
                "sentiment_trend": sentiment_metrics["sentiment_trend"],
                "positive_ratio": sentiment_metrics["positive_ratio"],
                "negative_ratio": sentiment_metrics["negative_ratio"]
            },
            "sentiment": sentiment_metrics["avg_sentiment"],
            "top_stories": top_stories,
            "data_quality": min(0.9, 0.5 + sentiment_metrics["article_count"] * 0.05)
        }

    def _analyze_analysts(self, ratings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze analyst ratings."""
        analyst_sentiment = self.finnhub_tool.calculate_analyst_sentiment(ratings_data)
        rating_changes = self.finnhub_tool.analyze_rating_changes(ratings_data)
        consensus = ratings_data.get("consensus", {})

        return {
            "metrics": {
                "consensus_sentiment": analyst_sentiment["sentiment"],
                "avg_price_target": consensus.get("avg_target", 0.0),
                "high_target": consensus.get("high_target", 0.0),  # MURTHY ADDED 2025-10-07
                "low_target": consensus.get("low_target", 0.0),  # MURTHY ADDED 2025-10-07
                "current_price": consensus.get("current_price", 0.0),  # MURTHY ADDED 2025-10-07
                "total_analysts": analyst_sentiment["total_analysts"],
                "bullish_ratio": analyst_sentiment["bullish_ratio"],
                "bearish_ratio": analyst_sentiment["bearish_ratio"],
                "upgrades": rating_changes["upgrades"],
                "downgrades": rating_changes["downgrades"],
                "momentum": rating_changes["momentum"]
            },
            "sentiment": analyst_sentiment["sentiment"],
            "confidence": analyst_sentiment["confidence"],
            "data_quality": (
                0.9 if analyst_sentiment["total_analysts"] >= 3 else
                0.6 if analyst_sentiment["total_analysts"] >= 1 else 0.2
            )
        }

    def _analyze_executives(self, exec_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze executive changes."""
        exec_stability = self.finnhub_tool.calculate_executive_stability(exec_data)

        # Sentiment based on stability and changes
        if exec_stability["recent_turnover"]:
            sentiment = exec_stability["avg_change_sentiment"]
        else:
            # High stability = slightly positive
            sentiment = (exec_stability["stability_score"] - 0.5) * 0.4

        return {
            "metrics": {
                "stability_score": exec_stability["stability_score"],
                "recent_changes": exec_stability["total_changes"],
                "departures": exec_stability["departures"],
                "leadership_count": exec_stability["leadership_count"]
            },
            "sentiment": sentiment,
            "data_quality": 0.7
        }

    def _calculate_aggregate_sentiment(
        self,
        filing_analysis: Dict,
        news_analysis: Dict,
        analyst_analysis: Dict,
        exec_analysis: Dict
    ) -> float:
        """
        Calculate weighted aggregate sentiment from all sources.

        Weights:
        - Analyst ratings: 35% (most predictive)
        - News sentiment: 30% (timely, broad coverage)
        - SEC filings: 25% (material events)
        - Executive stability: 10% (longer-term indicator)
        """
        weights = {
            "analyst": 0.35,
            "news": 0.30,
            "filing": 0.25,
            "exec": 0.10
        }

        # Weight by data quality
        filing_weight = weights["filing"] * filing_analysis["data_quality"]
        news_weight = weights["news"] * news_analysis["data_quality"]
        analyst_weight = weights["analyst"] * analyst_analysis["data_quality"]
        exec_weight = weights["exec"] * exec_analysis["data_quality"]

        total_weight = filing_weight + news_weight + analyst_weight + exec_weight

        if total_weight == 0:
            return 0.0

        sentiment = (
            filing_analysis["sentiment"] * filing_weight +
            news_analysis["sentiment"] * news_weight +
            analyst_analysis["sentiment"] * analyst_weight +
            exec_analysis["sentiment"] * exec_weight
        ) / total_weight

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(
        self,
        filing_analysis: Dict,
        news_analysis: Dict,
        analyst_analysis: Dict,
        exec_analysis: Dict
    ) -> float:
        """
        Calculate confidence based on data quality and agreement.
        """
        # Average data quality
        avg_quality = (
            filing_analysis["data_quality"] +
            news_analysis["data_quality"] +
            analyst_analysis["data_quality"] +
            exec_analysis["data_quality"]
        ) / 4

        # Signal agreement (low variance = high agreement = high confidence)
        sentiments = [
            filing_analysis["sentiment"],
            news_analysis["sentiment"],
            analyst_analysis["sentiment"],
            exec_analysis["sentiment"]
        ]

        # Calculate variance
        mean_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)

        # Agreement score (inverse of variance, normalized)
        agreement = 1.0 - min(1.0, variance / 0.5)

        # Combine quality and agreement
        confidence = 0.6 * avg_quality + 0.4 * agreement

        return round(confidence, 2)

    def _build_evidence(
        self,
        filing_analysis: Dict,
        news_analysis: Dict,
        analyst_analysis: Dict,
        exec_analysis: Dict
    ) -> List[Evidence]:
        """Build evidence list from all sources."""
        evidence = []

        # SEC filings evidence
        if filing_analysis["metrics"]["total_events"] > 0:
            evidence.append(Evidence(
                source="sec_edgar",
                value=filing_analysis["sentiment"],
                timestamp=datetime.utcnow(),
                confidence=filing_analysis["data_quality"],
                description=f"{filing_analysis['metrics']['total_events']} material events, "
                           f"sentiment: {filing_analysis['sentiment']:.2f}"
            ))

        # News evidence
        if news_analysis["metrics"]["total_articles"] > 0:
            evidence.append(Evidence(
                source="news_api",
                value=news_analysis["sentiment"],
                timestamp=datetime.utcnow(),
                confidence=news_analysis["data_quality"],
                description=f"{news_analysis['metrics']['total_articles']} articles, "
                           f"avg sentiment: {news_analysis['sentiment']:.2f}, "
                           f"trend: {news_analysis['metrics']['sentiment_trend']}"
            ))

        # Analyst ratings evidence
        if analyst_analysis["metrics"]["total_analysts"] > 0:
            evidence.append(Evidence(
                source="finnhub_ratings",
                value=analyst_analysis["sentiment"],
                timestamp=datetime.utcnow(),
                confidence=analyst_analysis["confidence"],
                description=f"{analyst_analysis['metrics']['total_analysts']} analysts, "
                           f"consensus: {analyst_analysis['sentiment']:.2f}, "
                           f"target: ${analyst_analysis['metrics']['avg_price_target']:.2f}"
            ))

        # Executive stability evidence
        evidence.append(Evidence(
            source="finnhub_executives",
            value=exec_analysis["sentiment"],
            timestamp=datetime.utcnow(),
            confidence=exec_analysis["data_quality"],
            description=f"Stability score: {exec_analysis['metrics']['stability_score']:.2f}, "
                       f"recent changes: {exec_analysis['metrics']['recent_changes']}"
        ))

        return evidence

    def _generate_alerts(
        self,
        ticker: str,
        filing_analysis: Dict,
        news_analysis: Dict,
        analyst_analysis: Dict,
        exec_analysis: Dict
    ) -> List[Alert]:
        """Generate alerts for significant market intelligence signals."""
        alerts = []

        # Multiple downgrades
        if analyst_analysis["metrics"]["downgrades"] >= 2:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="Multiple Analyst Downgrades",
                description=f"{analyst_analysis['metrics']['downgrades']} recent downgrades detected",
                recommended_action="Review analyst concerns and company guidance"
            ))

        # Negative news surge
        if (news_analysis["metrics"]["total_articles"] >= 3 and
            news_analysis["metrics"]["negative_ratio"] > 0.6):
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Negative News Sentiment Spike",
                description=f"{news_analysis['metrics']['negative_ratio']:.0%} of recent articles negative",
                recommended_action="Investigate source of negative coverage"
            ))

        # Material SEC event
        if filing_analysis["metrics"]["strategic_events"] > 0:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Material Strategic Event Filed",
                description="SEC 8-K filing indicates strategic action",
                recommended_action="Review filing details for M&A or partnership announcements"
            ))

        # Executive departures
        if exec_analysis["metrics"]["departures"] > 0:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="medium",
                title="Executive Departure Detected",
                description=f"{exec_analysis['metrics']['departures']} recent leadership departure(s)",
                recommended_action="Assess impact on operations and strategy"
            ))

        return alerts

    def _generate_summary(
        self,
        metrics: Dict[str, Any],
        sentiment: float
    ) -> str:
        """Generate human-readable summary of market intelligence."""
        analyst_metrics = metrics["analyst_ratings"]
        news_metrics = metrics["news_sentiment"]

        return (
            f"Market sentiment: {sentiment:.2f}. "
            f"{analyst_metrics['total_analysts']} analysts "
            f"({analyst_metrics['bullish_ratio']:.0%} bullish). "
            f"News sentiment: {news_metrics['avg_sentiment']:.2f} "
            f"({news_metrics['sentiment_trend']})."
        )
