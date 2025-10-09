"""
Qualitative Signal Agent

Evaluates market sentiment using recent news articles and social signals.
Gemini analyzes narrative trends and explains its reasoning transparently.

Business Logic:
- Fetches: Recent news articles from EODHD API (last 10 articles)
- Analyzes: Sentiment trends, key themes, narrative momentum
- Returns: Sentiment score (-1 to +1), confidence, and step-by-step reasoning
- Output: Used by fusion engine (25% weight) for final recommendation
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from google import genai
import httpx

from core.agent_contracts import AgentOutput, SignalType, Evidence
from datetime import datetime
from utils.langsmith_tracer import trace_agent, trace_step, trace_llm_call, log_metrics, log_api_call, log_error

logger = logging.getLogger(__name__)


class QualitativeAnalysis(BaseModel):
    """Gemini's structured response - ensures transparency in AI decision-making."""
    sentiment_score: float = Field(description="Score between -1.0 and +1.0")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    summary: str = Field(description="2-3 sentence summary of news sentiment")
    reasoning: str = Field(description="Step-by-step explanation of how you calculated the sentiment score")
    key_factors: list[str] = Field(description="Top 3 themes that influenced your score (e.g., 'Positive earnings coverage', 'Regulatory concerns')")
    positive_ratio: float = Field(description="Ratio of positive news (0.0 to 1.0)")
    negative_ratio: float = Field(description="Ratio of negative news (0.0 to 1.0)")


class ADKQualitativeSignal:
    """
    Evaluates market narrative and sentiment.

    Think: Media analyst reading headlines and gauging public perception.
    """

    def __init__(self, agent_id: str = "qualitative_signal"):
        self.agent_id = agent_id

        # Check if EODHD news API is available
        eodha_live = os.getenv("EODHD_LIVE_CONNECTORS", "false").strip('"').strip("'").lower() == "true"
        google_key = os.getenv("GOOGLE_API_KEY", "").strip('"').strip("'")
        self.live_mode = eodha_live and google_key and google_key != "your_google_api_key_here"

        # Direct Gemini client (simpler than ADK, more reliable)
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

        logger.info(f"ADKQualitativeSignal initialized (mode: {'live' if self.live_mode else 'sample'})")

    @trace_agent("qualitative_signal")


    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Main analysis workflow.

        Steps: Fetch news → Extract themes → Gemini analyzes → Return transparent output
        """
        logger.info(f"Analyzing qualitative signals for {ticker}")

        # Step 1: Fetch recent news articles from EODHD
        if self.live_mode:
            news_articles = await self._fetch_news(ticker, company_name)
        else:
            news_articles = self._get_sample_news(ticker)

        if not news_articles or len(news_articles) == 0:
            logger.warning(f"No news for {ticker}, using fallback")
            return self._create_fallback_output(ticker, company_name, sector)

        # Step 2: Gemini analyzes news sentiment and explains reasoning
        analysis = await self._analyze_with_gemini(ticker, company_name, news_articles)

        # Step 3: Extract metrics for fusion engine
        metrics = {
            "positive_ratio": analysis.positive_ratio,
            "negative_ratio": analysis.negative_ratio,
            "news_count": float(len(news_articles)),
            "sentiment_score": analysis.sentiment_score
        }

        # Step 4: Package for fusion engine with full transparency
        evidence = [
            Evidence(
                source="eodhd_news",
                value={"article_count": len(news_articles)},
                timestamp=datetime.utcnow(),
                confidence=0.75,
                description=analysis.summary
            )
        ]

        return AgentOutput(
            signal=SignalType.SENTIMENT,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics=metrics,  # Raw numbers for fusion math
            sentiment=analysis.sentiment_score,  # Gemini's overall assessment
            confidence=analysis.confidence,
            evidence=evidence,
            metadata={
                "sector": sector,
                "data_source": "EODHD" if self.live_mode else "sample",
                "reasoning": analysis.reasoning,  # WHY this score? (transparency)
                "key_factors": analysis.key_factors,  # WHAT drove the decision?
                "news_count": len(news_articles)
            }
        )

    @trace_llm_call("gemini-2.0-flash")


    async def _analyze_with_gemini(
        self,
        ticker: str,
        company_name: str,
        news_articles: List[str]
    ) -> QualitativeAnalysis:
        """
        Gemini analyzes news sentiment and explains its thinking.

        Returns structured output with score + reasoning (no black box).
        """

        news_text = "\n".join([f"{i+1}. {article[:200]}..." for i, article in enumerate(news_articles[:10])])

        prompt = f"""Analyze the sentiment of recent news for {company_name} ({ticker}):

{news_text}

Provide:
1. Sentiment score (-1.0 to +1.0): Based on overall tone of articles (negative news = negative score, positive news = positive score)
2. Confidence (0.0 to 1.0): How confident you are based on article quality and consistency
3. Summary: 2-3 sentence explanation of overall sentiment
4. Reasoning: Explain step-by-step how you arrived at the sentiment score. Which articles were positive? Which were negative? How did you weight them?
5. Key factors: List the top 3 themes from the news that most influenced your score (e.g., "Positive earnings beat", "Product launch concerns", "Leadership changes")
6. Positive ratio: What % of articles were positive? (0.0 to 1.0)
7. Negative ratio: What % of articles were negative? (0.0 to 1.0)"""

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=QualitativeAnalysis
                )
            )

            result = json.loads(response.text)
            return QualitativeAnalysis(**result)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback to neutral
            return QualitativeAnalysis(
                sentiment_score=0.0,
                confidence=0.5,
                summary="Analysis unavailable",
                reasoning="Gemini API call failed",
                key_factors=["API error", "No analysis available", "Using neutral score"],
                positive_ratio=0.33,
                negative_ratio=0.33
            )

    async def _fetch_news(self, ticker: str, company_name: str) -> List[str]:
        """Fetch real news from EODHD API."""
        try:
            api_key = os.getenv("EODHD_API_KEY", "").strip('"').strip("'")
            if not api_key:
                logger.warning("EODHD API key not found")
                return self._get_sample_news(ticker)

            url = "https://eodhd.com/api/news"
            params = {
                "s": f"{ticker}.US",
                "api_token": api_key,
                "limit": 10,
                "offset": 0
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                news_data = response.json()

                if not news_data:
                    logger.warning(f"No news data returned for {ticker}")
                    return self._get_sample_news(ticker)

                # Extract article titles and content
                articles = [
                    f"{item.get('title', '')}: {item.get('content', '')[:200]}"
                    for item in news_data[:10]
                ]

                logger.info(f"✅ Fetched {len(articles)} news articles for {ticker}")
                return articles

        except Exception as e:
            logger.warning(f"EODHD fetch failed: {e}, using sample data")
            return self._get_sample_news(ticker)

    def _get_sample_news(self, ticker: str) -> List[str]:
        """Sample news articles."""
        return [
            f"{ticker} reports strong quarterly earnings, beating analyst expectations",
            f"{ticker} announces new product launch in growing market segment",
            f"Analysts upgrade {ticker} citing improved fundamentals",
            f"{ticker} faces regulatory scrutiny over recent practices",
            f"Industry outlook positive for {ticker}'s sector"
        ]

    def _create_fallback_output(self, ticker: str, company_name: str, sector: Optional[str]) -> AgentOutput:
        """Fallback output when no news available."""
        return AgentOutput(
            signal=SignalType.SENTIMENT,
            agent_id=self.agent_id,
            ticker=ticker,
            metrics={
                "positive_ratio": 0.5,
                "negative_ratio": 0.5,
                "news_count": 0.0,
                "sentiment_score": 0.0
            },
            sentiment=0.0,
            confidence=0.3,
            evidence=[],
            metadata={
                "sector": sector,
                "data_source": "none",
                "reasoning": "No news articles available",
                "key_factors": ["No news data", "Using neutral fallback", "Low confidence"],
                "news_count": 0
            }
        )
