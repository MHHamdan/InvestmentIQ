"""
News API Tool

MCP-compliant tool for accessing company news and sentiment analysis.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NewsAPITool:
    """
    News API data access tool.

    Retrieves company-specific news articles with metadata and sentiment.

    Data modes:
    - Sample mode (default): Uses data/samples/news/*.json
    - Live mode: Connects to NewsAPI with rate limiting

    NewsAPI Compliance:
    - Rate limit: 100-1000 requests/day depending on tier
    - Attribution required for articles
    - Cannot cache results beyond 24 hours
    - Results limited to 30 days back on free tier
    """

    def __init__(self):
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.api_key = os.getenv("NEWS_API_KEY", "")
        self.samples_dir = Path("data/samples/news")
        self.base_url = "https://newsapi.org/v2/"

        logger.info(
            f"NewsAPITool initialized (mode: {'live' if self.live_mode else 'sample'})"
        )

    async def get_company_news(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 7,
        max_articles: int = 20
    ) -> Dict[str, Any]:
        """
        Get recent news articles for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name for search
            days_back: How many days back to search (max 30 on free tier)
            max_articles: Maximum articles to return

        Returns:
            Dictionary with articles and summary
        """
        if self.live_mode and self.api_key:
            return await self._fetch_live_news(
                ticker, company_name, days_back, max_articles
            )
        else:
            return self._fetch_sample_news(ticker)

    def _fetch_sample_news(self, ticker: str) -> Dict[str, Any]:
        """Load sample news data from fixture."""
        sample_file = self.samples_dir / f"{ticker}_news.json"

        if not sample_file.exists():
            logger.warning(f"No sample news for {ticker}, returning empty")
            return self._get_empty_news(ticker)

        with open(sample_file, 'r') as f:
            data = json.load(f)

        logger.info(
            f"Loaded {len(data.get('articles', []))} sample articles for {ticker}"
        )
        return data

    async def _fetch_live_news(
        self,
        ticker: str,
        company_name: str,
        days_back: int,
        max_articles: int
    ) -> Dict[str, Any]:
        """
        Fetch live news from NewsAPI.

        Implementation would use:
        - NewsAPI /everything endpoint with company name query
        - Date range filtering
        - Sort by relevancy or publishedAt
        - Rate limiting based on tier
        """
        logger.warning(
            "Live NewsAPI fetching not implemented. "
            "Falling back to sample data."
        )
        return self._fetch_sample_news(ticker)

    def _get_empty_news(self, ticker: str) -> Dict[str, Any]:
        """Return empty news structure."""
        return {
            "company": ticker,
            "articles": [],
            "summary": {
                "total_articles": 0,
                "avg_sentiment": 0.0,
                "sentiment_distribution": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0
                },
                "top_categories": [],
                "date_range": {
                    "start": datetime.utcnow().isoformat() + "Z",
                    "end": datetime.utcnow().isoformat() + "Z"
                }
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def calculate_sentiment_metrics(
        self,
        news_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate sentiment metrics from articles.

        Returns:
            Dictionary with sentiment scores and distribution
        """
        articles = news_data.get("articles", [])

        if not articles:
            return {
                "avg_sentiment": 0.0,
                "sentiment_trend": "neutral",
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "article_count": 0
            }

        sentiments = [art.get("sentiment", 0.0) for art in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)

        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count

        # Determine trend based on recent vs. older articles
        if len(articles) >= 3:
            recent_sentiment = sum(sentiments[:3]) / 3
            if recent_sentiment > avg_sentiment + 0.1:
                trend = "improving"
            elif recent_sentiment < avg_sentiment - 0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "neutral"

        return {
            "avg_sentiment": round(avg_sentiment, 3),
            "sentiment_trend": trend,
            "positive_ratio": positive_count / len(articles),
            "negative_ratio": negative_count / len(articles),
            "neutral_ratio": neutral_count / len(articles),
            "article_count": len(articles),
            "sentiment_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count
            }
        }

    def categorize_articles(
        self,
        news_data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize articles by topic.

        Categories:
        - earnings: Financial results, revenue announcements
        - product: Product launches, updates
        - strategic: M&A, partnerships, major deals
        - regulatory: Legal, compliance, regulatory matters
        - leadership: Executive changes, org announcements
        - analyst_rating: Analyst upgrades/downgrades
        - operations: Supply chain, manufacturing, ops
        - other: Misc news
        """
        categories = {
            "earnings": [],
            "product": [],
            "strategic": [],
            "regulatory": [],
            "leadership": [],
            "analyst_rating": [],
            "operations": [],
            "other": []
        }

        for article in news_data.get("articles", []):
            category = article.get("category", "other")
            if category in categories:
                categories[category].append(article)
            else:
                categories["other"].append(article)

        return categories

    def get_top_stories(
        self,
        news_data: Dict[str, Any],
        limit: int = 5,
        min_sentiment_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top news stories, optionally filtered by sentiment.

        Args:
            news_data: News data dictionary
            limit: Maximum stories to return
            min_sentiment_threshold: Only return stories with sentiment >= threshold

        Returns:
            List of top article dictionaries
        """
        articles = news_data.get("articles", [])

        if min_sentiment_threshold is not None:
            articles = [
                art for art in articles
                if art.get("sentiment", 0.0) >= min_sentiment_threshold
            ]

        # Sort by publishedAt (most recent first)
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get("publishedAt", ""),
            reverse=True
        )

        return sorted_articles[:limit]

    def extract_key_themes(
        self,
        news_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract key themes from article titles and descriptions.

        Returns:
            List of themes with frequency counts
        """
        # Simple keyword-based theme extraction
        theme_keywords = {
            "AI": ["ai", "artificial intelligence", "machine learning"],
            "Growth": ["growth", "revenue up", "increase", "expansion"],
            "Challenges": ["miss", "decline", "concern", "challenge", "down"],
            "Innovation": ["new", "launch", "innovation", "breakthrough"],
            "Competition": ["competition", "competitor", "market share"],
            "Regulatory": ["regulatory", "compliance", "legal", "investigation"]
        }

        articles = news_data.get("articles", [])
        theme_counts = {theme: 0 for theme in theme_keywords}

        for article in articles:
            text = (
                article.get("title", "") + " " +
                article.get("description", "")
            ).lower()

            for theme, keywords in theme_keywords.items():
                if any(keyword in text for keyword in keywords):
                    theme_counts[theme] += 1

        # Return themes sorted by frequency
        themes = [
            {"theme": theme, "frequency": count}
            for theme, count in theme_counts.items()
            if count > 0
        ]

        return sorted(themes, key=lambda x: x["frequency"], reverse=True)

    def get_news_summary(
        self,
        ticker: str,
        news_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive news summary for a ticker.

        Returns:
            Summary with metrics, categories, top stories, themes
        """
        sentiment_metrics = self.calculate_sentiment_metrics(news_data)
        categorized = self.categorize_articles(news_data)
        top_stories = self.get_top_stories(news_data, limit=3)
        themes = self.extract_key_themes(news_data)

        return {
            "ticker": ticker,
            "sentiment_metrics": sentiment_metrics,
            "category_breakdown": {
                cat: len(articles)
                for cat, articles in categorized.items()
                if len(articles) > 0
            },
            "top_stories": top_stories,
            "key_themes": themes,
            "total_articles": len(news_data.get("articles", [])),
            "timestamp": news_data.get("timestamp", datetime.utcnow().isoformat())
        }
