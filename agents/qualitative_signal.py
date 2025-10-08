import httpx
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from google.genai.types import Content, Part

# --- Environment Variable Loading ---
from dotenv import load_dotenv
load_dotenv()  # Load variables from the .env file
# ------------------------------------

# --- Pydantic Imports (Still used for structured output definition) ---
from pydantic import BaseModel, Field
# --------------------------------------------------------------------

# --- NEW: Google ADK Imports ---
# Agent is an alias for LlmAgent, the core reasoning agent in ADK.
from google.adk.agents import Agent
# InMemoryRunner is a simple, local runner for executing agents.
from google.adk.runners import InMemoryRunner
# genai_types are used for constructing content messages if needed.
from google.genai import types as genai_types
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

# --- The Pydantic model for structured output remains UNCHANGED ---
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
    Analyzes qualitative signals using Google ADK with a Gemini model.
    """

    def __init__(self, agent_id: str = "qualitative_signal"):
        self.agent_id = agent_id
        self.live_mode = os.getenv("EODHA_LIVE_CONNECTORS", "false").lower() == "true"
        self.samples_dir = Path("data/samples/qualitative")
        self.agent_bus = get_agent_bus()

        # --- REPLACED: LangChain setup is removed ---
        # --- NEW: Initialize ADK Agent and Runner ---
        self.analysis_agent = self._setup_adk_agent()
        # InMemoryRunner is self-contained and includes an InMemorySessionService
        self.runner = InMemoryRunner(
            agent=self.analysis_agent,
            app_name="qualitative_signal_app"
        )
        # -------------------------------------------

        logger.info(
            f"QualitativeSignalAgent initialized (mode: "
            f"{'live/ADK-Gemini' if self.live_mode else 'sample'})"
        )

# --- NEW: Method to configure the ADK Agent ---
    def _setup_adk_agent(self) -> Agent:
        """Sets up the ADK Agent for structured qualitative analysis."""
        # --- FIX: Make the instruction more explicit ---
        system_template = (
            "You are an expert financial analyst. Your task is to analyze the raw news articles "
            "provided in the session state key 'raw_news_content' for the company {company_name} (Ticker: {ticker}).\n"
            "Your response MUST be ONLY a single, valid JSON object that conforms to the required schema. "
            "Do not include any introductory text, explanations, or markdown formatting like ```json."
        )
        # --- END FIX ---

        agent = Agent(
            name="financial_analyzer",
            model="gemini-2.5-flash", # Using a specific model version is good practice
            instruction=system_template,
            output_schema=LLMQualitativeAnalysis,
            output_key="llm_analysis_result"
        )
        return agent

    @trace_agent("qualitative_signal", {"version": "3.0-adk"})
    async def analyze(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str] = None
    ) -> AgentOutput:
        """
        Analyze qualitative signals for a company.
        """
        logger.info(f"Analyzing qualitative signals for {ticker}")

        if self.live_mode:
            # This now calls the ADK-powered method
            data = await self._fetch_live_data(ticker, company_name)
        else:
            data = self._fetch_sample_data(ticker)

        # --- The rest of this method remains UNCHANGED ---
        metrics = self._extract_metrics(data)
        sentiment = self._calculate_sentiment(data)
        confidence = self._calculate_confidence(data)
        evidence = self._build_evidence(data)
        alerts = self._generate_alerts(data, ticker)

        observation = Observation(
            agent_id=self.agent_id,
            ticker=ticker,
            observation=self._generate_summary(data, metrics),
            data={"metrics": metrics, "sentiment": sentiment},
            confidence=confidence
        )
        self.agent_bus.broadcast_observation(observation)

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
                "data_source": "ADK-Gemini-generated" if self.live_mode else "sample",
                "news_count": data.get("news_count", 0)
            }
        )

        logger.info(
            f"Qualitative analysis complete for {ticker}: "
            f"sentiment={sentiment:.2f}, confidence={confidence:.2f}"
        )

        return output

    # --- _fetch_raw_news method remains UNCHANGED ---
    async def _fetch_raw_news(self, ticker: str, company_name: str) -> List[str]:
        """Fetch raw news content for the company using a demo API."""
        API_KEY = os.getenv("EODHD_API_KEY", "DEMO")
        full_ticker = f"{ticker.upper()}.US"
        URL = "https://eodhd.com/api/news"
        params = {
            "api_token": API_KEY, "s": full_ticker, "limit": 10,
            "from": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(URL, params=params)
                response.raise_for_status()
                news_data = response.json()
                if not news_data:
                    return []
                raw_articles = [
                    f"TITLE: {item.get('title', '')}\nCONTENT: {(item.get('content') or item.get('snippet') or item.get('title', ''))[:500]}..."
                    for item in news_data if item.get('content') or item.get('snippet') or item.get('title')
                ]
                logger.info(f"Successfully fetched {len(raw_articles)} news articles for {full_ticker}.")
                print(f"***************Fetched {len(raw_articles)} articles for {full_ticker}.")
                return raw_articles
        except Exception as e:
            logger.error(f"Error in _fetch_raw_news for {full_ticker}: {e}")
        return []

# --- REWRITTEN with FINAL FIX ---
    async def _fetch_live_data(
        self,
        ticker: str,
        company_name: str
    ) -> Dict[str, Any]:
        """Fetch raw data and invoke the ADK Agent for structured analysis."""
        # 1. Fetch raw news content
        raw_news = await self._fetch_raw_news(ticker, company_name)
        news_count_override = len(raw_news)
        if not raw_news:
            logger.warning(f"No news found for {ticker}. Falling back to sample data.")
            return self._fetch_sample_data(ticker)

        # 2. Prepare the initial state for the ADK session.
        initial_state = {
            "company_name": company_name,
            "ticker": ticker,
            "raw_news_content": "\n".join([f"- {n}" for n in raw_news]),
        }

        # 3. Create a unique session for this specific analysis run.
        session_id = f"analysis-{ticker}-{datetime.utcnow().isoformat()}"
        user_id = "analyst_user"
        await self.runner.session_service.create_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )

        # 4. Invoke the ADK agent via the runner.
        logger.info(f"Invoking ADK/Gemini for qualitative analysis for {ticker}...")
        try:
            trigger_message = Content(
                role="user",
                parts=[Part(text="Perform the qualitative financial analysis based on the provided news content.")]
            )

            async for _ in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=trigger_message
            ):
                pass

            # 5. Retrieve the final session state.
            final_session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name, user_id=user_id, session_id=session_id
            )

            # --- FIX: Expect a dictionary and validate it directly ---
            dict_result = final_session.state.get("llm_analysis_result")

            logger.debug(f"Received from state['llm_analysis_result']: {dict_result}")
            if not isinstance(dict_result, dict):
                raise ValueError(f"Expected a dictionary from session state, but got {type(dict_result)}.")

            # Validate the dictionary directly into the Pydantic model.
            structured_analysis_model = LLMQualitativeAnalysis.model_validate(dict_result)
            # --- END FIX ---

            # 6. Convert the Pydantic model back to a dictionary.
            analysis_dict = structured_analysis_model.model_dump()
            analysis_dict['news_count'] = news_count_override
            return analysis_dict

        except Exception as e:
            logger.error(f"ADK/Gemini analysis failed: {e}. Falling back to sample data.", exc_info=True)
            return self._fetch_sample_data(ticker)

    # --- All original methods below this point remain UNCHANGED ---
    # They correctly consume the dictionary produced by the LLM step.

    def _fetch_sample_data(self, ticker: str) -> Dict[str, Any]:
        """Load sample qualitative data (fallback)."""
        sample_file = self.samples_dir / f"{ticker.lower()}_qualitative.json"
        if sample_file.exists():
            with open(sample_file) as f:
                return json.load(f)
        return {
            "ticker": ticker, "overall_sentiment": "Neutral", "sentiment_score": 0.0,
            "news_count": 0, "key_themes": [], "positive_ratio": 0.33,
            "negative_ratio": 0.33, "reputation_score": 0.5, "market_perception": "neutral",
        }

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics."""
        return {
            "sentiment_score": data.get("sentiment_score", 0),
            "reputation_score": data.get("reputation_score", 0.5),
            "news_count": data.get("news_count", 0),
            "positive_ratio": data.get("positive_ratio", 0.33),
            "negative_ratio": data.get("negative_ratio", 0.33),
            "theme_count": len(data.get("key_themes", []))
        }

    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate qualitative sentiment score."""
        sentiment_score = data.get("sentiment_score", 0)
        pos = data.get("positive_ratio", 0.33)
        neg = data.get("negative_ratio", 0.33)
        breakdown_sentiment = pos - neg
        reputation = data.get("reputation_score", 0.5)
        reputation_sentiment = (reputation - 0.5) * 2
        perception_desc = data.get("market_perception", "").lower()
        perception_boost = 0.2 if "leader" in perception_desc or "strong" in perception_desc else -0.2 if "struggling" in perception_desc or "poor" in perception_desc else 0.0
        sentiment = (0.4 * sentiment_score + 0.3 * breakdown_sentiment + 0.2 * reputation_sentiment + 0.1 * perception_boost)
        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis."""
        news_count = data.get("news_count", 0)
        if news_count > 50: return 0.85
        if news_count > 20: return 0.75
        if news_count > 5: return 0.65
        return 0.50

    def _build_evidence(self, data: Dict[str, Any]) -> List[Evidence]:
        evidence = []
        sentiment_score = data.get("sentiment_score", 0)
        if abs(sentiment_score) > 0.3:
            evidence.append(Evidence(
                source="ADK_Gemini_sentiment", value={"sentiment_score": sentiment_score},
                timestamp=datetime.utcnow(),
                description=f"Gemini score of {sentiment_score:.2f} indicates {'positive' if sentiment_score > 0 else 'negative'} sentiment",
                confidence=0.9
            ))
        # ... other evidence generation ...
        return evidence

    def _generate_alerts(self, data: Dict[str, Any], ticker: str) -> List[Alert]:
        alerts = []
        sentiment_score = data.get("sentiment_score", 0)
        if sentiment_score < -0.5:
            alerts.append(Alert(
                agent_id=self.agent_id, ticker=ticker, severity="high", title="Negative Market Sentiment",
                message=f"Gemini Sentiment score of {sentiment_score:.2f} indicates strong negative perception",
                data={"sentiment_score": sentiment_score}
            ))
        # ... other alert generation ...
        return alerts

    def _generate_summary(self, data: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Generate analysis summary."""
        sentiment_label = data.get("overall_sentiment", "Neutral")
        news_count = data.get("news_count", 0)
        perception = data.get("market_perception", "neutral")
        return (
            f"ADK/Gemini-based sentiment: {sentiment_label} (Score: {metrics.get('sentiment_score'):.2f}) based on {news_count} articles. "
            f"Perception: {perception}"
        )

