"""
SEC EDGAR Tool

MCP-compliant tool for accessing SEC 8-K filings and material events.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class EdgarTool:
    """
    SEC EDGAR data access tool.

    Retrieves 8-K filings and material corporate events.

    Data modes:
    - Sample mode (default): Uses data/samples/edgar/*.json
    - Live mode: Connects to SEC EDGAR API with rate limiting

    SEC EDGAR Compliance:
    - Rate limit: 10 requests/second maximum
    - User-Agent header required with contact email
    - No automated scraping without throttling
    """

    def __init__(self):
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/edgar")
        self.user_agent = "InvestmentIQ/1.0 (contact@investmentiq.example)"

        logger.info(
            f"EdgarTool initialized (mode: {'live' if self.live_mode else 'sample'})"
        )

    async def get_recent_filings(
        self,
        ticker: str,
        form_type: str = "8-K",
        days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Get recent SEC filings for a company.

        Args:
            ticker: Stock ticker symbol
            form_type: Form type (default: 8-K for material events)
            days_back: How many days back to search

        Returns:
            Dictionary with filings data
        """
        if self.live_mode:
            return await self._fetch_live_filings(ticker, form_type, days_back)
        else:
            return self._fetch_sample_filings(ticker)

    def _fetch_sample_filings(self, ticker: str) -> Dict[str, Any]:
        """Load sample filing data from fixture."""
        sample_file = self.samples_dir / f"{ticker}_8K_sample.json"

        if not sample_file.exists():
            logger.warning(f"No sample filings for {ticker}, returning empty")
            return self._get_empty_filings(ticker)

        with open(sample_file, 'r') as f:
            data = json.load(f)

        logger.info(
            f"Loaded {len(data.get('filings', []))} sample filings for {ticker}"
        )
        return data

    async def _fetch_live_filings(
        self,
        ticker: str,
        form_type: str,
        days_back: int
    ) -> Dict[str, Any]:
        """
        Fetch live filings from SEC EDGAR.

        Implementation would use:
        - SEC EDGAR RSS feeds for real-time filings
        - EDGAR search API with CIK lookup
        - Rate limiting (10 req/sec max)
        - User-Agent header with contact email
        """
        logger.warning(
            "Live EDGAR fetching not implemented. "
            "Falling back to sample data."
        )
        return self._fetch_sample_filings(ticker)

    def _get_empty_filings(self, ticker: str) -> Dict[str, Any]:
        """Return empty filings structure."""
        return {
            "company": ticker,
            "company_name": f"{ticker} Inc.",
            "cik": "0000000000",
            "filings": [],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def extract_material_events(
        self,
        filings_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract material events from filings.

        Returns list of events with:
        - event_type: Type of material event
        - date: Filing date
        - summary: Brief description
        - sentiment: positive/neutral/negative
        - key_points: List of important details
        """
        events = []

        for filing in filings_data.get("filings", []):
            for extracted_event in filing.get("extracted_events", []):
                events.append({
                    "event_type": extracted_event.get("title", "Unknown Event"),
                    "item": extracted_event.get("item", ""),
                    "date": filing.get("filing_date", ""),
                    "summary": extracted_event.get("summary", ""),
                    "sentiment": extracted_event.get("sentiment", "neutral"),
                    "key_points": extracted_event.get("key_points", []),
                    "url": filing.get("url", ""),
                    "form_type": filing.get("form_type", "8-K")
                })

        return events

    def categorize_events(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize events by type.

        Categories:
        - financial_results: Earnings, revenue announcements
        - leadership_changes: Executive departures/appointments
        - strategic_actions: M&A, partnerships, major contracts
        - regulatory: Compliance, legal matters
        - other: Misc material events
        """
        categories = {
            "financial_results": [],
            "leadership_changes": [],
            "strategic_actions": [],
            "regulatory": [],
            "other": []
        }

        for event in events:
            item = event.get("item", "")
            event_type = event.get("event_type", "").lower()

            # Categorization logic based on 8-K item numbers
            if item.startswith("2."):
                categories["financial_results"].append(event)
            elif item.startswith("5."):
                categories["leadership_changes"].append(event)
            elif item in ["1.01", "1.02", "8.01"]:
                categories["strategic_actions"].append(event)
            elif item.startswith("3.") or item.startswith("4."):
                categories["regulatory"].append(event)
            else:
                categories["other"].append(event)

        return categories

    def calculate_event_sentiment(
        self,
        events: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate aggregate sentiment from events.

        Returns:
            Sentiment score [-1.0, 1.0]
        """
        if not events:
            return 0.0

        sentiment_map = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }

        total_sentiment = sum(
            sentiment_map.get(event.get("sentiment", "neutral"), 0.0)
            for event in events
        )

        return total_sentiment / len(events)

    def get_filing_summary(
        self,
        ticker: str,
        filings_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate summary of filings for a ticker.

        Returns:
            Summary with event counts, sentiment, key events
        """
        events = self.extract_material_events(filings_data)
        categorized = self.categorize_events(events)
        sentiment = self.calculate_event_sentiment(events)

        return {
            "ticker": ticker,
            "total_filings": len(filings_data.get("filings", [])),
            "total_events": len(events),
            "event_categories": {
                cat: len(events_list)
                for cat, events_list in categorized.items()
            },
            "aggregate_sentiment": sentiment,
            "recent_events": events[:5],  # Top 5 most recent
            "categorized_events": categorized,
            "timestamp": filings_data.get("timestamp", datetime.utcnow().isoformat())
        }
