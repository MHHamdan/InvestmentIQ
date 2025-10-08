"""
InvestmentIQ Dashboard

Streamlit web interface for multi-agent investment analysis.

MODIFIED: 2025-10-06 - MURTHY - Added FMP company profile auto-population
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
# NEW: Import FMP tool for company profile lookup
from tools.fmp_tool import FMPTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="InvestmentIQ MVAS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-left: 4px solid #3b82f6;
    }
    .recommendation-buy {
        color: #10b981;
        font-weight: 700;
        font-size: 2rem;
    }
    .recommendation-sell {
        color: #ef4444;
        font-weight: 700;
        font-size: 2rem;
    }
    .recommendation-hold {
        color: #f59e0b;
        font-weight: 700;
        font-size: 2rem;
    }
    .evidence-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #f9fafb;
        border-radius: 0.375rem;
        border-left: 3px solid #3b82f6;
    }
    .alert-high {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #fee2e2;
        border-radius: 0.375rem;
        border-left: 3px solid #ef4444;
    }
    .alert-medium {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #fef3c7;
        border-radius: 0.375rem;
        border-left: 3px solid #f59e0b;
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


# NEW: Function to fetch company profile from FMP
def fetch_company_profile(ticker: str) -> Optional[Dict[str, str]]:
    """
    Fetch company name and sector from FMP API.

    Returns:
        Dictionary with company_name and sector, or None if failed
    """
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
            "sector": profile.get("sector", "Technology"),  # Default to Technology
            "industry": profile.get("industry", "")
        }
    except Exception as e:
        logger.warning(f"Could not fetch profile for {ticker}: {e}")
        return None


def run_analysis(ticker: str, company_name: str, sector: str) -> Optional[Dict[str, Any]]:
    """
    Run full analysis pipeline.

    Returns:
        Analysis results or None if error
    """
    try:
        orchestrator, _, _, _, _, _ = initialize_agents()

        # Run async analysis in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            orchestrator.process({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector
            })
        )

        loop.close()

        if result.status == "success":
            return result.data
        else:
            st.error(f"Analysis failed: {result.data.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        st.error(f"Analysis failed: {str(e)}")
        return None


def render_recommendation_card(recommendation: Dict[str, Any]):
    """Render main recommendation card."""
    action = recommendation.get("action", "HOLD")
    confidence = recommendation.get("confidence", 0.0)
    fused_score = recommendation.get("fused_score", 0.0)

    # MURTHY ADDED 2025-10-07 - Extract price data for display
    price_data = recommendation.get("price_data", {})
    current_price = price_data.get("current_price", 0.0)
    avg_target = price_data.get("avg_price_target", 0.0)
    high_target = price_data.get("high_price_target", 0.0)
    low_target = price_data.get("low_price_target", 0.0)
    upside = price_data.get("upside_potential", 0.0)

    # Determine CSS class based on action
    if action in ["BUY", "ACCUMULATE"]:
        action_class = "recommendation-buy"
        action_emoji = "📈"
    elif action in ["SELL", "REDUCE"]:
        action_class = "recommendation-sell"
        action_emoji = "📉"
    else:
        action_class = "recommendation-hold"
        action_emoji = "➡️"

    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin-top: 0;">{action_emoji} Investment Recommendation</h2>
        <div class="{action_class}">{action}</div>
        <div style="margin-top: 1rem;">
            <strong>Confidence:</strong> {confidence:.1%}<br>
            <strong>Fused Score:</strong> {fused_score:+.3f}<br>
            <strong>Reasoning:</strong> {recommendation.get('reasoning', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MURTHY ADDED 2025-10-07 - Display price info in separate card to avoid HTML escaping
    if current_price > 0 and avg_target > 0:
        upside_color = "#10b981" if upside > 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="margin-top: -1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <strong>Current Price:</strong> ${current_price:.2f}<br>
            <strong>Analyst Target:</strong> ${avg_target:.2f} (Range: ${low_target:.2f} - ${high_target:.2f})<br>
            <strong>Upside Potential:</strong> <span style="color: {upside_color}; font-weight: bold;">{upside:+.1f}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_agent_breakdown(agent_outputs: list, signal_contributions: Dict[str, float]):
    """Render agent breakdown table."""
    st.subheader("Agent Analysis Breakdown")

    if not agent_outputs:
        st.info("No agent outputs available. Using sample mode with limited agents.")
        return

    # Build table data
    table_data = []
    for output in agent_outputs:
        agent_id = output.get("agent_id", "unknown")
        sentiment = output.get("sentiment", 0.0)
        confidence = output.get("confidence", 0.0)
        contribution = signal_contributions.get(agent_id, 0.0)

        # Determine status
        if sentiment > 0.2:
            status = "Bullish"
            status_color = "#10b981"
        elif sentiment < -0.2:
            status = "Bearish"
            status_color = "#ef4444"
        else:
            status = "Neutral"
            status_color = "#f59e0b"

        table_data.append({
            "Agent": agent_id.replace("_", " ").title(),
            "Sentiment": f"{sentiment:+.3f}",
            "Confidence": f"{confidence:.1%}",
            "Weight": f"{contribution:.1%}",
            "Status": status
        })

    df = pd.DataFrame(table_data)

    # Custom styling for dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def render_signal_fusion_charts(signal_contributions: Dict[str, float], explanations: list):
    """Render signal fusion visualization."""
    st.subheader("Signal Fusion Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Agent Contribution Weights**")

        # Pie chart of weights
        if signal_contributions:
            labels = [k.replace("_", " ").title() for k in signal_contributions.keys()]
            values = list(signal_contributions.values())

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])

            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=30, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No contribution data available")

    with col2:
        st.markdown("**Fusion Explanations**")

        if explanations:
            for explanation in explanations[:5]:  # Show top 5
                st.markdown(f"""
                <div class="evidence-item">
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explanations available")


def render_evidence(evidence_list: list):
    """Render supporting evidence."""
    st.subheader("Supporting Evidence")

    if not evidence_list:
        st.info("No evidence available")
        return

    for i, evidence in enumerate(evidence_list[:8], 1):
        source = evidence.get("source", "Unknown")
        description = evidence.get("description", "No description")
        confidence = evidence.get("confidence", 0.0)

        st.markdown(f"""
        <div class="evidence-item">
            <strong>{i}. {source}</strong> (confidence: {confidence:.1%})<br>
            {description}
        </div>
        """, unsafe_allow_html=True)


def render_alerts(recommendation: Dict[str, Any]):
    """Render alerts and warnings."""
    st.subheader("Alerts & Warnings")

    conflicting_points = recommendation.get("conflicting_points", [])
    alerts = recommendation.get("alerts", [])

    if not conflicting_points and not alerts:
        st.success("No alerts. All signals aligned.")
        return

    # Render conflicts
    if conflicting_points:
        st.markdown("**Signal Conflicts:**")
        for conflict in conflicting_points:
            st.markdown(f"""
            <div class="alert-high">
                {conflict}
            </div>
            """, unsafe_allow_html=True)

    # Render alerts
    if alerts:
        st.markdown("**Agent Alerts:**")
        for alert in alerts:
            st.markdown(f"""
            <div class="alert-medium">
                {alert}
            </div>
            """, unsafe_allow_html=True)


def render_metrics_overview(ticker: str, agent_outputs: list):
    """Render key metrics overview."""
    st.subheader("Quick Metrics Overview")

    cols = st.columns(4)

    # Calculate aggregate metrics
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

    with cols[0]:
        st.metric(
            label="Ticker",
            value=ticker.upper()
        )

    with cols[1]:
        sentiment_delta = f"{avg_sentiment:+.2f}"
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.2f}",
            delta=sentiment_delta
        )

    with cols[2]:
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.1%}"
        )

    with cols[3]:
        st.metric(
            label="Active Agents",
            value=num_agents,
            delta=f"{num_alerts} alerts" if num_alerts > 0 else None
        )


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">InvestmentIQ MVAS</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Agent Investment Analysis System</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("Analysis Configuration")

        # MURTHY MODIFIED 2025-10-07 - Simplified ticker input, auto-lookup on analysis
        ticker = st.text_input(
            "Stock Ticker *",
            value="AAPL",
            max_chars=5,
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
        ).upper()

        # Analysis button
        analyze_button = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True
        )

        st.divider()

        # MODIFIED: System info with FMP status
        st.markdown("**System Status**")
        use_fmp = os.getenv("USE_FMP_DATA", "false").lower() == "true"
        fmp_status = "🟢 FMP Enabled" if use_fmp else "🔴 Sample Data"
        st.caption(f"Data: {fmp_status}")
        st.caption(f"Agents: 5 active")
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        st.divider()

        # Disclaimer
        st.markdown("**Disclaimer**")
        st.caption(
            "InvestmentIQ is for educational purposes only. "
            "Not financial advice. Consult a professional advisor before investing."
        )

    # Main content area
    if analyze_button:
        if not ticker:
            st.error("Please enter a valid ticker symbol")
            return

        # MURTHY ADDED 2025-10-07 - Auto-fetch company info before analysis
        with st.spinner(f"Looking up {ticker} company info..."):
            profile = fetch_company_profile(ticker)
            if profile:
                company_name = profile["company_name"]
                sector = profile["sector"]
                st.session_state["last_company_name"] = company_name
                st.session_state["last_sector"] = sector
                st.success(f"✅ {company_name} ({sector})")
            else:
                # Fallback to defaults if lookup fails
                company_name = f"{ticker} Inc."
                sector = "Technology"
                st.warning("⚠️ Could not fetch company info. Using defaults.")

        # Show progress
        with st.spinner(f"Analyzing {ticker}..."):
            result = run_analysis(ticker, company_name, sector)

        if result is None:
            st.error("Analysis failed. Please try again.")
            return

        # Cache results in session state
        st.session_state["last_result"] = result
        st.session_state["last_ticker"] = ticker

    # Display cached results if available
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        ticker = st.session_state["last_ticker"]
        company_name = st.session_state.get("last_company_name", ticker)
        sector = st.session_state.get("last_sector", "Technology")

        recommendation = result.get("recommendation", {})

        # MURTHY ADDED 2025-10-07 - Display company info in header
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <h2 style="margin: 0; color: #1f2937;">{ticker}</h2>
            <p style="margin: 0.25rem 0 0 0; color: #6b7280; font-size: 1.1rem;">{company_name} • {sector}</p>
        </div>
        """, unsafe_allow_html=True)
        agent_outputs = result.get("agent_outputs", [])

        # Metrics overview
        render_metrics_overview(ticker, agent_outputs)

        st.divider()

        # Main recommendation
        render_recommendation_card(recommendation)

        st.divider()

        # Two-column layout for details
        col1, col2 = st.columns([1, 1])

        with col1:
            # Agent breakdown
            signal_contributions = recommendation.get("signal_contributions", {})
            render_agent_breakdown(agent_outputs, signal_contributions)

            st.divider()

            # Evidence
            evidence = recommendation.get("supporting_evidence", [])
            render_evidence(evidence)

        with col2:
            # Signal fusion
            explanations = recommendation.get("explanations", [])
            render_signal_fusion_charts(signal_contributions, explanations)

            st.divider()

            # Alerts
            render_alerts(recommendation)

        st.divider()

        # Workflow summary (expandable)
        with st.expander("View Workflow Details"):
            workflow = result.get("workflow_summary", [])
            if workflow:
                workflow_df = pd.DataFrame(workflow)
                st.dataframe(workflow_df, use_container_width=True)
            else:
                st.info("No workflow data available")

        # Raw data (expandable, for debugging)
        with st.expander("View Raw Data"):
            st.json(result)

    else:
        # Welcome screen
        st.info("Enter a ticker symbol and click 'Run Analysis' to begin")

        st.markdown("""
        ### How It Works

        1. **Enter Ticker**: Type a stock symbol (e.g., AAPL, MSFT, TSLA)
        2. **Run Analysis**: Click the button to start multi-agent analysis
        3. **Review Results**: See recommendation, agent breakdown, and supporting evidence

        ### Available Agents

        - **Workforce Intelligence**: Employee sentiment, hiring trends
        - **Market Intelligence**: Analyst ratings, news sentiment, SEC filings
        - More agents can be added by integrating legacy components

        ### Sample Data

        Currently running in sample mode with pre-loaded data for:
        - AAPL (Apple Inc.)
        - MSFT (Microsoft Corporation)
        - TSLA (Tesla Inc.)

        For live data, set `LIVE_CONNECTORS=true` in `.env` and add API keys.
        """)


if __name__ == "__main__":
    main()
