"""
Financial Modeling Prep (FMP) Tool

Comprehensive API integration for real-time financial data.
Provides financial ratios, analyst ratings, news, SEC filings, and historical data.
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

import requests
import aiohttp

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 250):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]

        if len(self.calls) >= self.calls_per_minute:
            # Wait until oldest call expires
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self.calls.append(now)


class FMPTool:
    """
    Financial Modeling Prep API integration.

    Provides access to:
    - Financial ratios and metrics
    - Analyst ratings and price targets
    - Company news and sentiment
    - SEC filings
    - Historical data for pattern matching

    Features:
    - Automatic rate limiting (250 calls/min free tier)
    - Error handling with graceful degradation
    - Sample data fallback
    - Response caching
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self):
        """Initialize FMP tool with API key and configuration."""
        self.api_key = os.getenv("FMP_API_KEY", "")
        self.enabled = os.getenv("USE_FMP_DATA", "false").lower() == "true"
        self.rate_limiter = RateLimiter(calls_per_minute=250)
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

        if not self.api_key and self.enabled:
            logger.warning("FMP_API_KEY not found but USE_FMP_DATA=true")

        logger.info(
            f"FMPTool initialized (enabled: {self.enabled}, "
            f"key: {'***' + self.api_key[-4:] if self.api_key else 'not set'})"
        )

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to FMP API with rate limiting and error handling.

        Args:
            endpoint: API endpoint (e.g., "/ratios/AAPL")
            params: Optional query parameters

        Returns:
            API response as dictionary

        Raises:
            Exception: On API errors
        """
        if not self.enabled:
            raise Exception("FMP is disabled (USE_FMP_DATA=false)")

        if not self.api_key:
            raise Exception("FMP_API_KEY not set")

        # Rate limiting
        await self.rate_limiter.acquire()

        # Build URL
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apikey"] = self.api_key

        # Check cache
        cache_key = f"{url}?{json.dumps(params, sort_keys=True)}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data

        # Make request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Cache response
                    self.cache[cache_key] = (data, time.time())

                    logger.debug(f"FMP API call successful: {endpoint}")
                    return data

        except aiohttp.ClientError as e:
            logger.error(f"FMP API error for {endpoint}: {e}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"FMP API timeout for {endpoint}")
            raise Exception("FMP API timeout")

    async def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """
        NEW: Get company profile information including name and sector.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with company name, sector, industry, etc.
        """
        try:
            endpoint = "/profile"
            params = {"symbol": ticker.upper()}
            data = await self._make_request(endpoint, params)

            if not data or len(data) == 0:
                raise Exception(f"No profile data for {ticker}")

            profile = data[0]
            return {
                "ticker": ticker.upper(),
                "company_name": profile.get("companyName", ""),
                "sector": profile.get("sector", ""),
                "industry": profile.get("industry", ""),
                "description": profile.get("description", ""),
                "website": profile.get("website", ""),
                "ceo": profile.get("ceo", ""),
                "exchange": profile.get("exchangeShortName", ""),
                "country": profile.get("country", ""),
            }

        except Exception as e:
            logger.error(f"Failed to get profile for {ticker}: {e}")
            raise

    async def get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """
        Get financial ratios for a company.

        Returns all 6 required metrics in ONE API call:
        - revenue_growth
        - gross_margin
        - operating_margin
        - net_margin
        - debt_to_equity
        - roe (Return on Equity)

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with financial ratios
        """
        try:
            endpoint = "/ratios"
            params = {"symbol": ticker.upper(), "limit": 2}  # Get 2 periods to calc revenue growth
            data = await self._make_request(endpoint, params)

            if not data or len(data) == 0:
                raise Exception(f"No data returned for {ticker}")

            # Get most recent period
            latest = data[0]

            # Calculate revenue growth from revenuePerShare (YoY change)
            revenue_growth = 0
            if len(data) >= 2:
                current_rev = latest.get("revenuePerShare", 0)
                prior_rev = data[1].get("revenuePerShare", 0)
                if current_rev and prior_rev and prior_rev != 0:
                    revenue_growth = (current_rev - prior_rev) / prior_rev

            # Calculate ROE using DuPont formula: ROE = Net Margin * Asset Turnover * Financial Leverage
            net_margin = latest.get("netProfitMargin", 0)
            asset_turnover = latest.get("assetTurnover", 0)
            financial_leverage = latest.get("financialLeverageRatio", 0)
            roe = net_margin * asset_turnover * financial_leverage if all([net_margin, asset_turnover, financial_leverage]) else 0

            return {
                "ticker": ticker.upper(),
                "revenue_growth": revenue_growth,  # Calculated from YoY revenuePerShare
                "gross_margin": latest.get("grossProfitMargin", 0),
                "operating_margin": latest.get("operatingProfitMargin", 0),
                "net_margin": latest.get("netProfitMargin", 0),
                "debt_to_equity": latest.get("debtToEquityRatio", 0),  # Fixed: was debtEquityRatio
                "roe": roe,  # Calculated using DuPont formula
                "current_ratio": latest.get("currentRatio", 0),
                "quick_ratio": latest.get("quickRatio", 0),
                "date": latest.get("date", ""),
                "period": latest.get("period", ""),
            }

        except Exception as e:
            logger.error(f"Failed to get financial ratios for {ticker}: {e}")
            raise

    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings and upgrades/downgrades.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with analyst ratings data
        """
        try:
            endpoint = "/grades"
            params = {
                "symbol": ticker.upper(),
                "limit": 20
            }  # Get last 20 ratings
            data = await self._make_request(endpoint, params)

            if not data:
                return {"ratings": [], "consensus": None}

            # Count rating actions
            upgrades = sum(1 for r in data if "upgrade" in r.get("newGrade", "").lower())
            downgrades = sum(1 for r in data if "downgrade" in r.get("newGrade", "").lower())

            # Calculate momentum
            recent_ratings = data[:5]  # Last 5 ratings
            bullish_count = sum(
                1 for r in recent_ratings
                if r.get("newGrade", "").lower() in ["buy", "outperform", "strong buy"]
            )
            bearish_count = sum(
                1 for r in recent_ratings
                if r.get("newGrade", "").lower() in ["sell", "underperform", "strong sell"]
            )

            return {
                "ratings": data,
                "upgrades": upgrades,
                "downgrades": downgrades,
                "momentum": "positive" if bullish_count > bearish_count else "negative" if bearish_count > bullish_count else "neutral",
                "total_ratings": len(data),
                "recent_ratings": recent_ratings
            }

        except Exception as e:
            logger.error(f"Failed to get analyst ratings for {ticker}: {e}")
            raise

    async def get_price_target_consensus(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst price target consensus.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with price targets
        """
        try:
            endpoint = "/price-target-consensus"
            params = {"symbol": ticker.upper()}
            data = await self._make_request(endpoint, params)

            if not data or len(data) == 0:
                return {
                    "avg_target": 0,
                    "high_target": 0,
                    "low_target": 0,
                    "median_target": 0
                }

            latest = data[0] if isinstance(data, list) else data

            return {
                "avg_target": latest.get("targetConsensus", 0) or latest.get("avgPriceTarget", 0),
                "high_target": latest.get("targetHigh", 0),
                "low_target": latest.get("targetLow", 0),
                "median_target": latest.get("targetMedian", 0),
                "total_analysts": latest.get("numberOfAnalysts", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get price targets for {ticker}: {e}")
            raise

    async def get_company_news(
        self,
        ticker: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get company news articles.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of articles (default: 50)

        Returns:
            List of news articles
        """
        try:
            endpoint = "/news/stock"
            params = {
                "symbols": ticker.upper(),
                "limit": limit
            }
            data = await self._make_request(endpoint, params)

            if not data:
                return []

            # Format articles
            articles = []
            for article in data:
                articles.append({
                    "title": article.get("title", ""),
                    "text": article.get("text", ""),
                    "site": article.get("site", ""),
                    "publishedDate": article.get("publishedDate", ""),
                    "url": article.get("url", ""),
                    "symbol": article.get("symbol", ticker.upper())
                })

            return articles

        except Exception as e:
            logger.error(f"Failed to get news for {ticker}: {e}")
            raise

    async def get_sec_filings(
        self,
        ticker: str,
        filing_type: str = "8-K",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get SEC filings for a company.

        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (e.g., "8-K", "10-K", "10-Q")
            limit: Maximum number of filings

        Returns:
            List of SEC filings
        """
        try:
            endpoint = "/sec-filings"
            params = {
                "symbol": ticker.upper(),
                "type": filing_type,
                "limit": limit
            }
            data = await self._make_request(endpoint, params)

            if not data:
                return []

            return data

        except Exception as e:
            logger.error(f"Failed to get SEC filings for {ticker}: {e}")
            raise

    async def get_historical_prices(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            to_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            List of historical prices
        """
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")

            endpoint = "/historical-chart"
            params = {
                "symbol": ticker.upper(),
                "from": from_date,
                "to": to_date
            }
            data = await self._make_request(endpoint, params)

            if not data or "historical" not in data:
                return []

            return data["historical"]

        except Exception as e:
            logger.error(f"Failed to get historical prices for {ticker}: {e}")
            raise

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        MURTHY ADDED 2025-10-07 - Get real-time stock quote with current price

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with current price, change, volume, etc.
        """
        try:
            endpoint = "/quote"
            params = {"symbol": ticker.upper()}
            data = await self._make_request(endpoint, params)

            if not data or len(data) == 0:
                raise Exception(f"No quote data for {ticker}")

            quote = data[0]
            return {
                "ticker": ticker.upper(),
                "price": quote.get("price", 0.0),
                "change": quote.get("change", 0.0),
                "change_percent": quote.get("changesPercentage", 0.0),
                "volume": quote.get("volume", 0),
                "avg_volume": quote.get("avgVolume", 0),
                "market_cap": quote.get("marketCap", 0),
                "pe_ratio": quote.get("pe", 0.0),
                "day_low": quote.get("dayLow", 0.0),
                "day_high": quote.get("dayHigh", 0.0),
                "year_low": quote.get("yearLow", 0.0),
                "year_high": quote.get("yearHigh", 0.0),
                "previous_close": quote.get("previousClose", 0.0),
            }

        except Exception as e:
            logger.error(f"Failed to get quote for {ticker}: {e}")
            raise

    async def get_historical_ratios(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical financial ratios.

        Args:
            ticker: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods to retrieve

        Returns:
            List of historical ratios
        """
        try:
            endpoint = "/ratios"
            params = {
                "symbol": ticker.upper(),
                "period": period,
                "limit": limit
            }
            data = await self._make_request(endpoint, params)

            if not data:
                return []

            return data

        except Exception as e:
            logger.error(f"Failed to get historical ratios for {ticker}: {e}")
            raise


# Standalone testing
async def main():
    """Test FMP tool functionality."""
    print("🧪 Testing FMP Tool")
    print("=" * 50)

    tool = FMPTool()

    if not tool.enabled:
        print("❌ FMP is disabled. Set USE_FMP_DATA=true to test.")
        return

    ticker = "AAPL"
    print(f"\n📊 Testing with ticker: {ticker}")

    # Test 1: Financial Ratios
    print("\n1️⃣ Testing get_financial_ratios()...")
    try:
        ratios = await tool.get_financial_ratios(ticker)
        print(f"✅ Success! Revenue Growth: {ratios['revenue_growth']:.2%}")
        print(f"   Gross Margin: {ratios['gross_margin']:.2%}")
        print(f"   ROE: {ratios['roe']:.2%}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 2: Analyst Ratings
    print("\n2️⃣ Testing get_analyst_ratings()...")
    try:
        ratings = await tool.get_analyst_ratings(ticker)
        print(f"✅ Success! Total Ratings: {ratings['total_ratings']}")
        print(f"   Momentum: {ratings['momentum']}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 3: Price Targets
    print("\n3️⃣ Testing get_price_target_consensus()...")
    try:
        targets = await tool.get_price_target_consensus(ticker)
        print(f"✅ Success! Consensus Target: ${targets['avg_target']:.2f}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Test 4: Company News
    print("\n4️⃣ Testing get_company_news()...")
    try:
        news = await tool.get_company_news(ticker, limit=5)
        print(f"✅ Success! Found {len(news)} articles")
        if news:
            print(f"   Latest: {news[0]['title'][:60]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")

    print("\n" + "=" * 50)
    print("✅ FMP Tool testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
