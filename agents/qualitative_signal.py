import httpx 
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Environment Variable Loading ---
from dotenv import load_dotenv
load_dotenv()  # Load variables from the .env file
# ------------------------------------

# --- LLM/LangChain Imports ---
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
# -----------------------------

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

# --- Define Structured Output for the LLM (Pydantic Model) ---
class LLMQualitativeAnalysis(BaseModel):
    """Structured output for the LLM's qualitative analysis."""
    overall_sentiment: str = Field(description="The final aggregated sentiment, one of 'Strongly Positive', 'Positive', 'Neutral', 'Negative', 'Strongly Negative'.")
    sentiment_score: float = Field(description="A numeric score between -1.0 (most negative) and 1.0 (most positive), representing the News Sentiment.")
    reputation_score: float = Field(description="A score between 0.0 (poor) and 1.0 (excellent) for Organizational Reputation, based on news of leadership, ethics, and sustainability.")
    market_perception: str = Field(description="A concise summary of how the company is generally viewed in the market (e.g., 'Innovation leader,' 'Struggling with debt,' 'Strong operational execution').")
    key_themes: List[str] = Field(description="A list of 3-5 main recurring themes or narratives from the analyzed news articles.")
    positive_ratio: float = Field(description="The calculated ratio of positive-leaning news statements (0.0 to 1.0).")
    negative_ratio: float = Field(description="The calculated ratio of negative-leaning news statements (0.0 to 1.0).")
    news_count: int = Field(description="The total number of news articles provided for analysis.")


class QualitativeSignalAgent:
    """
    Analyzes qualitative signals using an LLM (GPT-4o) including sentiment, news, and reputation.
    """

    def __init__(self, agent_id: str = "qualitative_signal"):
        self.agent_id = agent_id
        self.live_mode = os.getenv("LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/qualitative")
        self.agent_bus = get_agent_bus()

        # --- Initialize LLM and LangChain Runnable ---
        # ChatOpenAI automatically detects OPENAI_API_KEY from the environment
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.analysis_chain = self._setup_analysis_chain()
        # -----------------------------------------------

        logger.info(
            f"QualitativeSignalAgent initialized (mode: "
            f"{'live/LLM' if self.live_mode else 'sample'})"
        )

    # --- New Method to set up the LLM Chain ---
    def _setup_analysis_chain(self):
        """Sets up the LangChain runnable for structured LLM analysis."""
        
        # Prompt Template
        system_template = (
            "You are an expert financial analyst. Your task is to analyze raw news articles "
            "and provide a structured, objective qualitative assessment of the specified company. "
            "You must strictly adhere to the provided JSON format for your output. "
            "Focus specifically on News Sentiment, Market Perception, and Organizational Reputation."
        )

        human_template = (
            "Analyze the following raw news articles for {company_name} (Ticker: {ticker}).\n\n"
            "Raw News Content:\n---\n{raw_news_content}\n---\n\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        # LCEL Chain: Prompt -> Model -> Structured Output (Pydantic)
        chain = prompt | self.llm.with_structured_output(LLMQualitativeAnalysis)
        return chain

    @trace_agent("qualitative_signal", {"version": "2.0-llm"})
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
            # This calls the LLM
            data = await self._fetch_live_data(ticker, company_name)
        else:
            # Fallback to sample data
            data = self._fetch_sample_data(ticker)

        # Extract metrics
        metrics = self._extract_metrics(data)

        # Calculate sentiment (uses LLM output but applies blending/adjustment)
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
                "data_source": "LLM-generated" if self.live_mode else "sample",
                "news_count": data.get("news_count", 0)
            }
        )

        logger.info(
            f"Qualitative analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output


    # # --- Placeholder for Raw News Fetching ---
    # async def _fetch_raw_news(self, ticker: str, company_name: str) -> List[str]:
    #     """SIMULATED: Fetch raw news content for the company."""
    #     # **NOTE:** In a real application, this method would call external News APIs.
        
    #     logger.warning("Using SIMULATED raw news for LLM analysis.")
        
    #     # Example Simulation Logic
    #     if ticker.lower() == "aapl":
    #         return [
    #             "Apple (AAPL) stock hits all-time high after unveiling revolutionary M4 chip.",
    #             "Supply chain issues slightly disrupt iPhone 16 production outlook.",
    #             "Tim Cook's leadership earns high marks in recent ethical survey, boosting reputation.",
    #             "Analysts are bullish on Apple's services revenue growth for the next quarter."
    #         ]
        
    #     # Default fallback
    #     return [
    #         f"News for {company_name} is mostly neutral today.",
    #         f"The company's reputation remains stable in its sector with no major events.",
    #         f"Minor themes of market consolidation are present in recent reports."
    #     ]

    # --- Updated Method for Real News Data ---
    async def _fetch_raw_news(self, ticker: str, company_name: str) -> List[str]:
        """Fetch raw news content for the company using a demo API."""
        
        # 1. Configuration (use a placeholder or load from .env if you register)
        # The EODHD API is free for certain tickers using the 'DEMO' key.
        API_KEY = os.getenv("EODHD_API_KEY", "DEMO") 
        
        # EODHD requires the exchange suffix, e.g., AAPL.US
        # For a real application, you'd look this up, but for the demo, we'll assume .US
        full_ticker = f"{ticker.upper()}.US"
        
        # EODHD Financial News API Endpoint
        URL = f"https://eodhd.com/api/news"
        
        params = {
            "api_token": API_KEY,
            "s": full_ticker,  # Ticker symbol
            "limit": 10,       # Get 10 most recent articles
            "from": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), # Last 7 days
        }
        
        # Use httpx.AsyncClient for asynchronous requests
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(URL, params=params)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                
                news_data = response.json()
                
                if not news_data:
                    logger.warning(f"No news found for {full_ticker}. Returning empty list.")
                    return []
                
                # The LLM needs the content. We extract the full article text (or a summary/snippet).
                raw_articles = []
                for item in news_data:
                    # Prioritize content/snippet if available, otherwise use title
                    content = item.get('content') or item.get('snippet') or item.get('title', '')
                    if content:
                        raw_articles.append(f"TITLE: {item.get('title', '')}\nCONTENT: {content[:500]}...") # Truncate content for token efficiency
                        
                logger.info(f"Successfully fetched {len(raw_articles)} news articles for {full_ticker}.")
                return raw_articles

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error fetching news for {full_ticker}: {e}")
        except httpx.RequestError as e:
            logger.error(f"Network Error fetching news for {full_ticker}: {e}")
        except Exception as e:
            logger.error(f"Unexpected Error in _fetch_raw_news: {e}")
            
        # Fallback to empty list on error
        return []

    async def _fetch_live_data(
        self,
        ticker: str,
        company_name: str
    ) -> Dict[str, Any]:
        """Fetch raw data and invoke the LLM for structured qualitative analysis."""
        
        # 1. Fetch raw data (simulated)
        raw_news = await self._fetch_raw_news(ticker, company_name)
        
        # 2. Prepare input for the LLM Chain
        llm_input = {
            "company_name": company_name,
            "ticker": ticker,
            "raw_news_content": "\n".join([f"- {n}" for n in raw_news]), 
        }
        
        news_count_override = len(raw_news)

        # 3. Invoke the LLM asynchronously
        logger.info(f"Invoking GPT-4o for qualitative analysis for {ticker}...")
        try:
            # The .ainvoke() call returns the Pydantic model instance
            print("*******Call LLM*******")
            structured_analysis_model: LLMQualitativeAnalysis = await self.analysis_chain.ainvoke(llm_input)
            
            # 4. Convert Pydantic model to a standard dictionary 
            analysis_dict = structured_analysis_model.model_dump()
            
            # The agent's existing code expects the news_count to be accurate
            analysis_dict['news_count'] = news_count_override

            return analysis_dict
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}. Falling back to sample data.", exc_info=True)
            return self._fetch_sample_data(ticker)


    # --- Original methods for sample data and processing remain below ---

    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample qualitative data (fallback)."""
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

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics."""
        # LLM output is designed to be consumed here
        return {
            "sentiment_score": data.get("sentiment_score", 0),
            "reputation_score": data.get("reputation_score", 0.5),
            "news_count": data.get("news_count", 0),
            # Note: We simulate sentiment_breakdown from LLM's positive/negative ratio
            "positive_ratio": data.get("positive_ratio", 0.33),
            "negative_ratio": data.get("negative_ratio", 0.33),
            "theme_count": len(data.get("key_themes", []))
        }

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate qualitative sentiment score (uses LLM output for weighting)."""
        # Base sentiment from LLM's sentiment score
        sentiment_score = data.get("sentiment_score", 0)

        # Sentiment breakdown (using LLM's ratios)
        pos = data.get("positive_ratio", 0.33)
        neg = data.get("negative_ratio", 0.33)
        breakdown_sentiment = pos - neg

        # Reputation score
        reputation = data.get("reputation_score", 0.5)
        reputation_sentiment = (reputation - 0.5) * 2  # Convert to [-1, 1]

        # Market perception - This is simplified as the LLM provides a string description
        perception_desc = data.get("market_perception", "").lower()
        perception_boost = 0.0
        if "leader" in perception_desc or "strong" in perception_desc:
            perception_boost = 0.2
        elif "struggling" in perception_desc or "poor" in perception_desc:
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

    # ... (rest of _build_evidence, _generate_alerts, and _generate_summary remain the same)
    
    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        # ... (implementation from original code)
        evidence = []

        # Sentiment evidence
        sentiment_score = data.get("sentiment_score", 0)
        if abs(sentiment_score) > 0.3:
            evidence.append(Evidence(
                source="LLM_sentiment_analysis",
                value={"sentiment_score": sentiment_score},
                timestamp=datetime.utcnow(),
                description=f"LLM score of {sentiment_score:.2f} indicates "
                           f"{'positive' if sentiment_score > 0 else 'negative'} market sentiment",
                confidence=0.9
            ))

        # News volume evidence
        news_count = data.get("news_count", 0)
        if news_count > 0:
            evidence.append(Evidence(
                source="LLM_input",
                value={"news_count": news_count},
                timestamp=datetime.utcnow(),
                description=f"LLM analyzed {news_count} raw news articles",
                confidence=0.8
            ))

        # Theme evidence
        themes = data.get("key_themes", [])
        if themes:
            evidence.append(Evidence(
                source="LLM_theme_analysis",
                value={"themes": themes},
                timestamp=datetime.utcnow(),
                description=f"Key themes identified by LLM: {', '.join(themes)}",
                confidence=0.85
            ))

        return evidence
    
    def _generate_alerts(self, data: Dict[str, Any], ticker: str) -> List[Alert]:
        # ... (implementation from original code)
        alerts = []

        # Negative sentiment alert
        sentiment_score = data.get("sentiment_score", 0)
        if sentiment_score < -0.5:
            alerts.append(Alert(
                agent_id=self.agent_id,
                ticker=ticker,
                severity="high",
                title="Negative Market Sentiment",
                message=f"LLM Sentiment score of {sentiment_score:.2f} indicates strong negative perception",
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
                message=f"LLM Reputation score of {reputation:.2f}",
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
                message=f"Concerning themes identified by LLM: {', '.join(detected_risks)}",
                data={"themes": detected_risks}
            ))

        return alerts
    
    def _generate_summary(self, data: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Generate analysis summary."""
        sentiment_label = data.get("overall_sentiment", "Neutral")
        news_count = data.get("news_count", 0)
        perception = data.get("market_perception", "neutral")

        return (
            f"LLM-based Market sentiment: {sentiment_label} (Score: {metrics.get('sentiment_score'):.2f}) based on {news_count} articles. "
            f"Overall perception: {perception}"
        )