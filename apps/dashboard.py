"""
InvestmentIQ Dashboard

Streamlit web interface for multi-agent investment analysis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent
from agents.strategic_orchestrator import StrategicOrchestratorAgent
from core.signal_fusion import SignalFusion

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
    workforce_agent = WorkforceIntelligenceAgent()
    market_agent = MarketIntelligenceAgent()

    # Note: Legacy agents (financial, qualitative, context) can be added here
    # For now, using only new agents
    orchestrator = StrategicOrchestratorAgent(
        agent_id="orchestrator",
        financial_agent=None,  # Legacy support
        qualitative_agent=None,
        context_agent=None,
        workforce_agent=workforce_agent,
        market_agent=market_agent
    )

    return orchestrator, workforce_agent, market_agent


def run_analysis(ticker: str, company_name: str, sector: str) -> Optional[Dict[str, Any]]:
    """
    Run full analysis pipeline.

    Returns:
        Analysis results or None if error
    """
    try:
        orchestrator, _, _ = initialize_agents()

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

        # Ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            max_chars=5,
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
        ).upper()

        # Company name
        company_name = st.text_input(
            "Company Name",
            value="Apple Inc.",
            help="Full company name"
        )

        # Sector
        sector = st.selectbox(
            "Sector",
            options=[
                "Technology",
                "Healthcare",
                "Financial",
                "Consumer",
                "Industrial",
                "Energy",
                "Other"
            ],
            index=0
        )

        # Analysis button
        analyze_button = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True
        )

        st.divider()

        # System info
        st.markdown("**System Status**")
        st.caption(f"Mode: Sample Data")
        st.caption(f"Agents: 2 active")
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

        recommendation = result.get("recommendation", {})
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
