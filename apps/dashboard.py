"""
InvestmentIQ - Transparent AI Investment Analysis Dashboard

Modern, clean interface inspired by Apple/Google design principles.
Shows complete transparency: data sources, agent reasoning, and decision flow.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.adk_orchestrator import ADKOrchestrator


# ============================================================================
# Page Configuration - Clean, Modern Theme
# ============================================================================

st.set_page_config(
    page_title="InvestmentIQ | Group 2 Capstone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS - Apple/Google Inspired Design
# ============================================================================

st.markdown("""
<style>
    /* Clean, modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit footer only */
    footer {visibility: hidden;}

    /* Main container */
    .main {
        padding: 2rem 4rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }

    /* Agent reasoning card */
    .agent-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .agent-card h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem 0;
    }

    .score-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .score-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }

    .score-neutral {
        background: linear-gradient(135deg, #bdc3c7 0%, #95a5a6 100%);
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def get_score_color(score):
    """Get color based on sentiment score."""
    if score > 0.2:
        return "#38ef7d"  # Green
    elif score < -0.2:
        return "#f45c43"  # Red
    else:
        return "#95a5a6"  # Gray


def get_recommendation_badge(score):
    """Generate HTML badge for recommendation."""
    if score > 0.5:
        return '<span class="score-badge score-positive">STRONG BUY</span>'
    elif score > 0.2:
        return '<span class="score-badge score-positive">BUY</span>'
    elif score > -0.2:
        return '<span class="score-badge score-neutral">HOLD</span>'
    elif score > -0.5:
        return '<span class="score-badge score-negative">SELL</span>'
    else:
        return '<span class="score-badge score-negative">STRONG SELL</span>'


def create_gauge_chart(score, title="Final Score"):
    """Create a beautiful gauge chart for the score."""
    score_pct = score * 100  # Convert to percentage

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'family': 'Inter'}},
        delta={'reference': 0, 'increasing': {'color': "#38ef7d"}, 'decreasing': {'color': "#f45c43"}},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkgray", 'ticksuffix': '%'},
            'bar': {'color': get_score_color(score), 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-100, -20], 'color': 'rgba(244, 92, 67, 0.2)'},
                {'range': [-20, 20], 'color': 'rgba(149, 165, 166, 0.2)'},
                {'range': [20, 100], 'color': 'rgba(56, 239, 125, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Inter', 'size': 14},
        height=300
    )

    return fig


def create_agent_contribution_chart(agent_outputs, weights):
    """Create waterfall chart showing agent contributions."""
    contributions = []
    labels = []

    for output in agent_outputs:
        agent_id = output.agent_id
        weight = weights.get(agent_id, 0)
        contribution = output.sentiment * weight * 100  # Convert to percentage
        contributions.append(contribution)
        labels.append(agent_id.replace('_', ' ').title())

    # Sort by absolute contribution
    sorted_data = sorted(zip(labels, contributions), key=lambda x: abs(x[1]), reverse=True)
    labels, contributions = zip(*sorted_data)

    colors = [get_score_color(c / 100) for c in contributions]  # Use original scale for color

    fig = go.Figure(go.Bar(
        x=labels,
        y=contributions,
        marker_color=colors,
        text=[f"{c:+.2f}%" for c in contributions],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Agent Contributions to Final Score",
            'font': {'size': 20, 'family': 'Inter'}
        },
        xaxis_title="Agent",
        yaxis_title="Contribution (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': 'Inter', 'size': 12},
        height=500,
        margin=dict(t=100, b=80, l=60, r=40),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            ticksuffix='%'
        )
    )

    return fig


# ============================================================================
# Main App
# ============================================================================

def main():
    # Sidebar - Project Info
    with st.sidebar:
        st.markdown("### 🎓 Group 2: Capstone Project")
        st.markdown("**Team Members:**")
        st.markdown("- Mohammed\n- Rui\n- Ameya\n- Amine\n- Rajesh\n- Murthy")

        st.markdown("---")
        st.markdown("### ⚙️ Architecture")
        st.markdown("**Powered by:**")
        st.markdown("- Google Gemini 2.0 Flash (AI)")
        st.markdown("- Custom Signal Fusion Engine")
        st.markdown("- 4 Specialist ADK Agents")

        st.markdown("---")
        st.markdown("### 📊 Data Sources")
        st.markdown("- FMP (Financials)")
        st.markdown("- EODHD (News)")
        st.markdown("- FRED (Macro)")

        st.markdown("---")
        st.caption(f"📅 Last Updated: Oct 8, 2025 21:29")

    # Header
    st.title("📊 InvestmentIQ - AI Investment Analysis")
    st.markdown("**Complete Transparency:** Data Sources • Agent Reasoning • Decision Flow")

    # Input Section - Centered and Aligned
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        ticker = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, TSLA",
            help="Enter the stock ticker symbol you want to analyze",
            label_visibility="visible"
        ).upper()

        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if analyze_button and ticker:
        test_file = Path(f"tests/test_results_{ticker}.json")

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🚀 Initializing orchestrator...")
        progress_bar.progress(10)

        # Run analysis
        orchestrator = ADKOrchestrator()

        status_text.text("📊 Running 4 specialist agents in parallel...")
        progress_bar.progress(30)

        try:
            # Run async analysis (LIVE)
            result = asyncio.run(orchestrator.analyze(ticker))
            progress_bar.progress(80)

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Display Results
            display_results(result, orchestrator)

        except Exception as e:
            # Fallback to cached result if exists
            if test_file.exists():
                st.warning(f"⚠️ Live analysis failed. Loading cached results...")
                st.error(f"Error: {str(e)}")
                import json
                with open(test_file, 'r') as f:
                    result = json.load(f)
                progress_bar.empty()
                status_text.empty()
                display_results(result, orchestrator)
            else:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)


def display_results(result, orchestrator):
    """Display analysis results with beautiful, transparent UI."""

    # Handle both dict and object formats
    fused = result['fused_signal']
    if isinstance(fused, dict):
        final_score = fused['final_score']
        confidence = fused['confidence']
        signal_weights = fused.get('signal_weights', {})
    else:
        final_score = fused.final_score
        confidence = fused.confidence
        signal_weights = fused.signal_weights

    company = result.get('company_name', result['ticker'])
    sector = result.get('sector', 'Unknown')

    # Handle agent_outputs - could be dict or object
    agent_outputs = result['agent_outputs']
    if agent_outputs and isinstance(agent_outputs[0], dict):
        # Convert dict to simple namespace for easier access
        from types import SimpleNamespace
        agent_outputs = [SimpleNamespace(**agent) for agent in agent_outputs]

    # Company Header
    st.markdown("---")
    col_info1, col_info2 = st.columns([2, 1])

    with col_info1:
        st.markdown(f"## {company} ({result['ticker']})")
        st.markdown(f"**Sector:** {sector}")

    with col_info2:
        st.metric("Analysis Date", datetime.now().strftime("%b %d, %Y %H:%M"))

    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #666; font-size: 0.9rem;">FINAL SCORE</p>
            <h1 style="margin: 0.5rem 0; color: {get_score_color(final_score)}; font-size: 3rem;">
                {final_score:+.3f}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #666; font-size: 0.9rem;">CONFIDENCE</p>
            <h1 style="margin: 0.5rem 0; color: #667eea; font-size: 3rem;">
                {confidence:.1%}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <p style="margin: 0; color: #666; font-size: 0.9rem;">RECOMMENDATION</p>
            <div style="margin-top: 1rem;">
                {get_recommendation_badge(final_score)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Gauge Chart
    st.plotly_chart(create_gauge_chart(final_score), use_container_width=True)

    # Agent Contributions
    st.markdown("### 🤖 Agent Analysis Breakdown")
    st.plotly_chart(
        create_agent_contribution_chart(agent_outputs, signal_weights),
        use_container_width=True
    )

    # Detailed Agent Cards with Transparency
    st.markdown("### 🔍 Transparent Agent Reasoning")
    st.markdown("*Click on each agent to see detailed reasoning and key factors*")

    for agent_output in agent_outputs:
        agent_name = agent_output.agent_id.replace('_', ' ').title()

        with st.expander(f"**{agent_name}** - Sentiment: {agent_output.sentiment:+.3f} | Confidence: {agent_output.confidence:.2%}"):
            # Metrics
            st.markdown("**📊 Metrics:**")
            metrics_col1, metrics_col2 = st.columns(2)

            def format_metric_value(key, value):
                """Format metric values with appropriate units and formatting."""
                if not isinstance(value, (int, float)):
                    return str(value)

                key_lower = key.lower()

                # Integer counts (no decimals, with comma separators for readability)
                if any(term in key_lower for term in ['total_analysts', 'buy_count', 'sell_count', 'hold_count', 'strong_buy', 'strong_sell']):
                    return f"{int(value):,}"

                # Ratios (no $ sign, 2 decimals)
                if any(term in key_lower for term in ['debt_to_equity', 'roe', 'roa', 'current_ratio', 'quick_ratio', 'p/e', 'p/b']):
                    return f"{value:.2f}"

                # Percentage metrics already in percentage form (not 0-1 scale)
                # upside_potential is calculated as ((target - price) / price) * 100, so already a percentage
                if 'upside_potential' in key_lower:
                    return f"{value:.2f}%"

                # Other specific percentage metrics (already in percentage form)
                if any(term in key_lower for term in ['gdp_growth', 'unemployment_rate', 'fed_funds_rate']):
                    return f"{value:.2f}%"

                # General percentage metrics (0-1 scale)
                if any(term in key_lower for term in ['margin', 'growth', 'ratio', 'rate', 'return', 'yield', 'change']):
                    if abs(value) <= 1.5:
                        return f"{value:.2%}"

                # Currency metrics (no debt_to_equity here anymore)
                if any(term in key_lower for term in ['price', 'revenue', 'income', 'earnings', 'profit', 'cash', 'market_cap', 'value']):
                    if abs(value) >= 1_000_000_000:
                        return f"${value/1_000_000_000:,.2f}B"
                    elif abs(value) >= 1_000_000:
                        return f"${value/1_000_000:,.2f}M"
                    elif abs(value) >= 1_000:
                        return f"${value:,.2f}"
                    else:
                        return f"${value:.2f}"

                # Default formatting
                if abs(value) >= 1_000:
                    return f"{value:,.2f}"
                elif abs(value) >= 1:
                    return f"{value:.2f}"
                else:
                    return f"{value:.4f}"

            metrics_items = list(agent_output.metrics.items())
            mid = len(metrics_items) // 2

            with metrics_col1:
                for key, value in metrics_items[:mid]:
                    st.metric(key.replace('_', ' ').title(), format_metric_value(key, value))

            with metrics_col2:
                for key, value in metrics_items[mid:]:
                    st.metric(key.replace('_', ' ').title(), format_metric_value(key, value))

            # Reasoning
            if 'reasoning' in agent_output.metadata:
                st.markdown("**💡 Gemini's Reasoning:**")
                st.info(agent_output.metadata['reasoning'])

            # Key Factors
            if 'key_factors' in agent_output.metadata:
                st.markdown("**📌 Key Factors:**")
                for factor in agent_output.metadata['key_factors']:
                    st.markdown(f"- {factor}")

            # Data Source
            data_source = agent_output.metadata.get('data_source', 'Unknown')
            st.caption(f"Data Source: {data_source}")

    # Fusion Details
    with st.expander("⚙️ Fusion Engine Details"):
        st.markdown("**Weighted Average Calculation:**")

        fusion_data = []
        for agent_id, weight in signal_weights.items():
            agent_output = next((a for a in agent_outputs if a.agent_id == agent_id), None)
            if agent_output:
                contribution = agent_output.sentiment * weight
                fusion_data.append({
                    'Agent': agent_id.replace('_', ' ').title(),
                    'Sentiment': f"{agent_output.sentiment:+.3f}",
                    'Weight': f"{weight:.2%}",
                    'Contribution': f"{contribution*100:+.2f}%"
                })

        st.table(fusion_data)

        st.markdown("**Formula:**")
        st.code(f"Final Score = Σ (Agent Sentiment × Agent Weight)\n            = {final_score:+.3f} ({final_score*100:+.1f}%)", language="text")

    # Export Results
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Download Full Report (JSON)"):
            import json
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"investment_analysis_{result['ticker']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

    with col2:
        st.caption(f"⏱️ Analysis completed in {result.get('execution_time', 0):.2f} seconds")


if __name__ == "__main__":
    main()
