"""
ADK Orchestrator - Coordinates 4 specialist agents with A2A collaboration

Simple orchestration pattern:
1. Run all 4 agents in parallel
2. Collect outputs
3. Fuse signals using custom fusion engine
4. Return unified recommendation
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.langsmith_tracer import trace_agent, trace_step, trace_llm_call, log_metrics, log_api_call, log_error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.adk_financial_analyst import ADKFinancialAnalyst
from agents.adk_market_intelligence import ADKMarketIntelligence
from agents.adk_qualitative_signal import ADKQualitativeSignal
from agents.adk_context_engine import ADKContextEngine
from core.signal_fusion import SignalFusion
from core.agent_contracts import AgentOutput, FusedSignal

logger = logging.getLogger(__name__)


class ADKOrchestrator:
    """
    Orchestrates 4 ADK agents with parallel execution and signal fusion.

    Agents:
    1. Financial Analyst - FMP financial ratios
    2. Market Intelligence - FMP analyst ratings & price targets
    3. Qualitative Signal - EODHD news sentiment
    4. Context Engine - FMP sector performance + FRED macroeconomics
    """

    def __init__(self):
        """Initialize orchestrator and all agents."""
        logger.info("Initializing ADK Orchestrator...")

        # Initialize 4 specialist agents
        self.financial_analyst = ADKFinancialAnalyst()
        self.market_intelligence = ADKMarketIntelligence()
        self.qualitative_signal = ADKQualitativeSignal()
        self.context_engine = ADKContextEngine()

        # Initialize custom fusion engine
        self.fusion = SignalFusion(
            method="weighted_average",
            use_llm_summary=False  # Disabled - we have Gemini in agents already
        )

        logger.info("✅ ADK Orchestrator initialized with 4 agents")

    @trace_agent("orchestrator")


    async def analyze(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a stock ticker with all 4 agents.

        Args:
            ticker: Stock ticker (e.g., "AAPL")
            company_name: Company name (optional, will fetch from FMP if not provided)
            sector: Sector (optional, will fetch from FMP if not provided)

        Returns:
            Dictionary with fused signal and individual agent outputs
        """
        logger.info(f"🔍 Starting analysis for {ticker}")
        start_time = datetime.now()

        # Get company profile if not provided
        if not company_name or not sector:
            profile = await self._get_company_profile(ticker)
            company_name = company_name or profile.get("company_name", ticker)
            sector = sector or profile.get("sector", "Unknown")

        logger.info(f"Analyzing {company_name} ({ticker}) in {sector} sector")

        # Run all 4 agents in parallel
        logger.info("Running 4 agents in parallel...")
        agent_outputs = await self._run_agents_parallel(ticker, company_name, sector)

        # Fuse signals using custom fusion engine
        logger.info("Fusing signals...")
        fused_signal = self._fuse_signals(ticker, agent_outputs)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"✅ Analysis complete for {ticker}: "
            f"score={fused_signal.final_score:.3f}, "
            f"confidence={fused_signal.confidence:.3f}, "
            f"time={execution_time:.2f}s"
        )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "fused_signal": fused_signal,
            "agent_outputs": agent_outputs,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }

    async def _run_agents_parallel(
        self,
        ticker: str,
        company_name: str,
        sector: str
    ) -> List[AgentOutput]:
        """
        Run all 4 agents in parallel for maximum speed.

        Returns:
            List of AgentOutput from each agent
        """
        # Create tasks for parallel execution
        tasks = [
            self.financial_analyst.analyze(ticker, company_name, sector),
            self.market_intelligence.analyze(ticker, company_name, sector),
            self.qualitative_signal.analyze(ticker, company_name, sector),
            self.context_engine.analyze(ticker, company_name, sector)
        ]

        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect valid outputs
        agent_outputs = []
        agent_names = ["Financial Analyst", "Market Intelligence", "Qualitative Signal", "Context Engine"]

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ {agent_names[i]} failed: {result}")
            else:
                agent_outputs.append(result)
                logger.info(f"✅ {agent_names[i]}: sentiment={result.sentiment:.2f}, confidence={result.confidence:.2f}")

        if not agent_outputs:
            raise Exception("All agents failed - cannot generate recommendation")

        return agent_outputs

    def _fuse_signals(self, ticker: str, agent_outputs: List[AgentOutput]) -> FusedSignal:
        """
        Fuse agent signals using custom fusion engine.

        Fusion weights (without workforce):
        - Financial: 0.35 (up from 0.30)
        - Market Intelligence: 0.30 (up from 0.25)
        - Qualitative: 0.25 (up from 0.20)
        - Context: 0.10 (same)
        """
        # Custom weights for 4-agent system
        weights = {
            "financial_analyst": 0.35,
            "market_intelligence": 0.30,
            "qualitative_signal": 0.25,
            "context_engine": 0.10
        }

        # Fuse using custom fusion engine
        fused_signal = self.fusion.fuse(
            ticker=ticker,
            agent_outputs=agent_outputs,
            weights=weights
        )

        return fused_signal

    async def _get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """Get company profile from FMP."""
        try:
            # Use financial analyst's FMP tool to get profile
            from tools.fmp_tool import FMPTool
            fmp = FMPTool()
            profile = await fmp.get_company_profile(ticker)
            return profile
        except Exception as e:
            logger.warning(f"Could not fetch company profile for {ticker}: {e}")
            return {
                "company_name": ticker,
                "sector": "Unknown"
            }

    def get_recommendation(self, fused_signal: FusedSignal) -> str:
        """
        Convert fused score to buy/hold/sell recommendation.

        Thresholds:
        - Strong Buy: > 0.5
        - Buy: 0.2 to 0.5
        - Hold: -0.2 to 0.2
        - Sell: -0.5 to -0.2
        - Strong Sell: < -0.5
        """
        score = fused_signal.final_score
        confidence = fused_signal.confidence

        if score > 0.5:
            action = "STRONG BUY"
        elif score > 0.2:
            action = "BUY"
        elif score > -0.2:
            action = "HOLD"
        elif score > -0.5:
            action = "SELL"
        else:
            action = "STRONG SELL"

        # Add confidence qualifier
        if confidence < 0.5:
            qualifier = "Low confidence"
        elif confidence < 0.7:
            qualifier = "Moderate confidence"
        else:
            qualifier = "High confidence"

        return f"{action} ({qualifier})"


# Convenience function for simple usage
async def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Simple convenience function to analyze a stock.

    Usage:
        result = await analyze_stock("AAPL")
        print(result["fused_signal"].final_score)
    """
    orchestrator = ADKOrchestrator()
    return await orchestrator.analyze(ticker)
