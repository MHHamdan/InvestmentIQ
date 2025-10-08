import sys
import os

script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '..'))

"""
InvestmentIQ Dashboard

Multi-agent investment analysis platform powered by AI.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent
from agents.context_engine import ContextEngineAgent
from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent
from agents.strategic_orchestrator import StrategicOrchestratorAgent
from core.signal_fusion import SignalFusion
from tools.fmp_tool import FMPTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="InvestmentIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #ffffff;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Typography */
    .main-title {
        font-size: 2.8rem;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.125rem;
        color: #6B7280;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }

    .rec-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }

    .rec-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    .rec-hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    .rec-action {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .rec-confidence {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Metric cards */
    .metric-container {
        background: #F9FAFB;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }

    /* Evidence cards */
    .evidence-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Alert styling */
    .alert-critical {
        background: #FEE2E2;
        border-left-color: #DC2626;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }

    .alert-warning {
        background: #FEF3C7;
        border-left-color: #D97706;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agents():
    """Initialize agents once and cache them."""
    financial_agent = FinancialAnalystAgent()
    qualitative_agent = QualitativeSignalAgent()
    context_agent = ContextEngineAgent()
    workforce_agent = WorkforceIntelligenceAgent()
    market_agent = MarketIntelligenceAgent()

    orchestrator = StrategicOrchestratorAgent(
        agent_id="orchestrator",
        financial_agent=financial_agent,
        qualitative_agent=qualitative_agent,
        context_agent=context_agent,
        workforce_agent=workforce_agent,
        market_agent=market_agent
    )

    return orchestrator, financial_agent, qualitative_agent, context_agent, workforce_agent, market_agent


def fetch_company_profile(ticker: str) -> Optional[Dict[str, str]]:
    """Fetch company name and sector from FMP API."""
    use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"

    if not use_fmp:
        return None

    try:
        fmp_tool = FMPTool()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        profile = loop.run_until_complete(fmp_tool.get_company_profile(ticker))
        loop.close()

        return {
            "company_name": profile.get("company_name", ""),
            "sector": profile.get("sector", "Technology"),
            "industry": profile.get("industry", "")
        }
    except Exception as e:
        logger.warning(f"Could not fetch profile for {ticker}: {e}")
        return None


def run_analysis(ticker: str, company_name: str, sector: str) -> Optional[Dict[str, Any]]:
    """Run full analysis pipeline."""
    try:
        orchestrator, *_ = initialize_agents()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            orchestrator.orchestrate_analysis(ticker, company_name, sector)
        )
        loop.close()

        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        st.error(f"Analysis failed: {str(e)}")
        return None


def render_recommendation_card(recommendation: Dict[str, Any]):
    """Render clean recommendation card."""
    action = recommendation.get("action", "HOLD").upper()
    confidence = recommendation.get("confidence", 0.0)
    score = recommendation.get("fused_score", 0.0)
    reasoning = recommendation.get("reasoning", "")

    # Determine card style
    card_class = "rec-card"
    if action == "BUY":
        card_class += " rec-buy"
    elif action == "SELL":
        card_class += " rec-sell"
    else:
        card_class += " rec-hold"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="rec-action">{action}</div>
        <div class="rec-confidence">Confidence: {confidence:.1%} | Score: {score:+.3f}</div>
        <div style="margin-top: 1rem; font-size: 0.95rem;">{reasoning}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_overview(ticker: str, agent_outputs: list):
    """Render clean metrics overview."""
    if agent_outputs:
        avg_sentiment = sum(o.get("sentiment", 0) for o in agent_outputs) / len(agent_outputs)
        avg_confidence = sum(o.get("confidence", 0) for o in agent_outputs) / len(agent_outputs)
        num_agents = len(agent_outputs)
        num_alerts = sum(len(o.get("alerts", [])) for o in agent_outputs)
    else:
        avg_sentiment = 0.0
        avg_confidence = 0.0
        num_agents = 0
        num_alerts = 0

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            label="Company",
            value=ticker.upper()
        )

    with cols[1]:
        st.metric(
            label="Sentiment",
            value=f"{avg_sentiment:+.2f}",
            delta=f"{avg_sentiment:.1%}"
        )

    with cols[2]:
        st.metric(
            label="Confidence",
            value=f"{avg_confidence:.1%}"
        )

    with cols[3]:
        st.metric(
            label="Agents",
            value=num_agents,
            delta=f"{num_alerts} alerts" if num_alerts > 0 else "No alerts"
        )


def render_agent_breakdown(agent_outputs: list, signal_contributions: Dict[str, float]):
    """Render agent analysis breakdown."""
    st.subheader("Analysis Breakdown")

    for output in agent_outputs:
        agent_name = output.get("agent_id", "Unknown").replace("_", " ").title()
        sentiment = output.get("sentiment", 0)
        confidence = output.get("confidence", 0)
        weight = signal_contributions.get(output.get("agent_id", ""), 0)

        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**{agent_name}**")

            with col2:
                color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                st.markdown(f":{color}[{sentiment:+.2f}]")

            with col3:
                st.caption(f"{confidence:.0%} confidence")

            if weight > 0:
                st.progress(weight, text=f"{weight:.1%} weight")

            st.divider()


def render_evidence(evidence: list):
    """Render supporting evidence."""
    st.subheader("Key Evidence")

    if not evidence:
        st.info("No evidence available")
        return

    for idx, item in enumerate(evidence[:10], 1):
        source = item.get("source", "Unknown")
        description = item.get("description", "")
        confidence = item.get("confidence", 0)

        st.markdown(f"""
        <div class="evidence-card">
            <strong>{idx}. {source}</strong> (confidence: {confidence:.0%})<br/>
            {description}
        </div>
        """, unsafe_allow_html=True)


def render_alerts(recommendation: Dict[str, Any]):
    """Render alerts."""
    alerts = recommendation.get("alerts", [])

    if not alerts:
        st.success("No alerts")
        return

    st.subheader("Alerts")

    for alert in alerts:
        severity = alert.get("severity", "info")
        message = alert.get("message", "")

        if severity == "high":
            st.error(message)
        elif severity == "medium":
            st.warning(message)
        else:
            st.info(message)


def main():
    """Main dashboard application."""

    # Clean header
    st.markdown('<h1 class="main-title">InvestmentIQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered investment analysis</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Analysis")

        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            max_chars=5,
            help="Enter stock ticker (e.g., AAPL, MSFT)"
        ).upper()

        if st.button("Lookup Company", width="stretch"):
            if ticker:
                with st.spinner("Fetching..."):
                    profile = fetch_company_profile(ticker)
                    if profile:
                        st.session_state["company_name"] = profile["company_name"]
                        st.session_state["sector"] = profile["sector"]
                        st.success(f"Found: {profile['company_name']}")
                        st.rerun()
                    else:
                        st.warning("Company info not available")
            else:
                st.error("Enter a ticker first")

        company_name = st.text_input(
            "Company Name",
            value=st.session_state.get("company_name", "Apple Inc.")
        )

        sector_options = [
            "Technology", "Healthcare", "Financial Services",
            "Consumer Cyclical", "Consumer Defensive", "Industrials",
            "Energy", "Basic Materials", "Real Estate",
            "Utilities", "Communication Services", "Other"
        ]

        default_sector = st.session_state.get("sector", "Technology")
        try:
            sector_index = sector_options.index(default_sector)
        except ValueError:
            sector_index = 0

        sector = st.selectbox(
            "Sector",
            options=sector_options,
            index=sector_index
        )

        analyze_button = st.button(
            "Run Analysis",
            type="primary",
            width="stretch"
        )

        st.divider()

        st.markdown("### Status")
        use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"
        data_source = "Live Data" if use_fmp else "Sample Data"
        st.caption(f"Source: {data_source}")
        st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")

        st.divider()

        st.caption(
            "Educational purposes only. Not financial advice."
        )

    # Main content
    if analyze_button:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return

        with st.spinner(f"Analyzing {ticker}..."):
            result = run_analysis(ticker, company_name, sector)

        if not result:
            return

        recommendation = result.get("recommendation", {})
        agent_outputs = result.get("agent_outputs", [])

        # Metrics
        render_metrics_overview(ticker, agent_outputs)

        st.divider()

        # Recommendation
        render_recommendation_card(recommendation)

        # Two-column layout
        col1, col2 = st.columns([1, 1])

        with col1:
            signal_contributions = recommendation.get("signal_contributions", {})
            render_agent_breakdown(agent_outputs, signal_contributions)

        with col2:
            evidence = recommendation.get("supporting_evidence", [])
            render_evidence(evidence)

        st.divider()

        # Alerts
        render_alerts(recommendation)

        # Details (expandable)
        with st.expander("Technical Details"):
            st.json(result)

    else:
        # Welcome screen
        st.info("Enter a ticker symbol and run analysis to begin")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### How it works

            1. Enter a stock ticker symbol
            2. Optionally lookup company details
            3. Run the analysis
            4. Review AI-powered insights
            """)

        with col2:
            st.markdown("""
            #### What you get

            - Multi-agent analysis
            - Actionable recommendations
            - Supporting evidence
            - Risk alerts
            """)


if __name__ == "__main__":
    main()
