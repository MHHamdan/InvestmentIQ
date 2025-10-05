# InvestmentIQ System Overview

## High-Level Architecture

InvestmentIQ is a **5-agent multi-agent system (MVAS)** that analyzes investment opportunities by fusing signals from different data sources. Here's how all components work together:

```
User Input (Ticker: AAPL)
        ↓
Strategic Orchestrator ← (coordinates everything)
        ↓
    Parallel Agent Execution
    ┌─────┬─────┬─────┬─────┐
    │  1  │  2  │  3  │  4  │  5
    ↓     ↓     ↓     ↓     ↓
Financial│Qual │Context│Workforce│Market
Analyst  │Signal│Engine │Intel    │Intel
    ↓     ↓     ↓     ↓     ↓
    └─────┴─────┴─────┴─────┘
              ↓
      Signal Fusion (weighted ensemble)
              ↓
      Conflict Detection
              ↓
    Debate/Consensus (if conflicts)
              ↓
      Final Recommendation
              ↓
    Streamlit Dashboard (visualization)
```

---

## Step-by-Step Flow

### **Step 1: User Input**
User enters a stock ticker (e.g., "AAPL") via:
- Streamlit dashboard (web UI)
- CLI command
- Python API call

**What happens:**
```python
request = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
}
```

---

### **Step 2: Strategic Orchestrator Receives Request**

**File:** `agents/strategic_orchestrator.py`

**What it does:**
1. Validates input (ticker required)
2. Logs workflow initiation
3. Calls all available agents in parallel

**Code flow:**
```python
orchestrator = StrategicOrchestratorAgent(...)
response = await orchestrator.process(request)
```

---

### **Step 3: Parallel Agent Execution**

The orchestrator calls **5 specialist agents** simultaneously:

#### **Agent 1: Financial Analyst**
- **File:** `agents/financial_analyst.py`
- **Tool:** `tools/financial_data_tool.py`
- **Analyzes:** Revenue, margins, debt ratios, cash flow
- **Output:** Financial health score + sentiment (-1 to +1)

#### **Agent 2: Qualitative Signal**
- **File:** `agents/qualitative_signal.py`
- **Tool:** `tools/qualitative_data_tool.py`
- **Analyzes:** Market sentiment, brand perception
- **Output:** Sentiment score + risk assessment

#### **Agent 3: Context Engine**
- **File:** `agents/context_engine.py`
- **Tools:** `tools/context_rule_tool.py`, `tools/rag_context_tool.py`
- **Analyzes:** Historical patterns, similar cases (RAG)
- **Output:** Context-adjusted recommendation

#### **Agent 4: Workforce Intelligence** 
- **File:** `agents/workforce_intelligence.py`
- **Data:** `data/samples/workforce/*.json` (or live sources)
- **Analyzes:** Employee ratings, hiring trends, churn rates
- **Output:** Workforce sentiment + alerts (e.g., hiring freeze)

#### **Agent 5: Market Intelligence** 
- **File:** `agents/market_intelligence.py`
- **Tools:**
  - `tools/edgar_tool.py` (SEC filings)
  - `tools/news_api_tool.py` (news sentiment)
  - `tools/finnhub_tool.py` (analyst ratings)
- **Analyzes:** Material events, news sentiment, analyst consensus
- **Output:** Market sentiment + alerts (e.g., downgrades)

**Each agent returns:**
```python
AgentOutput(
    signal=SignalType.WORKFORCE,  # or MARKET_INTELLIGENCE, etc.
    agent_id="workforce_intelligence",
    ticker="AAPL",
    sentiment=0.65,  # -1 (bearish) to +1 (bullish)
    confidence=0.82,  # 0 to 1
    metrics={...},  # Agent-specific metrics
    evidence=[...],  # Supporting evidence
    alerts=[...]  # Any warnings
)
```

---

### **Step 4: Agent Communication (A2A)**

**File:** `core/agent_bus.py`

**What happens:**
- Agents broadcast **Observations** to message bus
- Agents broadcast **Alerts** for significant findings
- All messages logged for audit trail

**Example:**
```python
# Agent broadcasts observation
observation = Observation(
    agent_id="workforce_intelligence",
    ticker="AAPL",
    observation="Employee rating: 4.2/5.0 (stable). Hiring: active.",
    confidence=0.82
)
agent_bus.broadcast_observation(observation)
```

---

### **Step 5: Signal Fusion**

**File:** `core/signal_fusion.py`

**What it does:**
1. **Collects** all 5 agent outputs
2. **Assigns weights** (Financial=30%, Market=25%, Sentiment=20%, Workforce=15%, Context=10%)
3. **Calculates** weighted average sentiment
4. **Generates** SHAP-like explanations (contribution of each agent)

**Example output:**
```python
FusedSignal(
    ticker="AAPL",
    final_score=0.58,  # Combined sentiment
    confidence=0.79,
    signal_weights={
        "financial_analyst": 0.30,
        "market_intelligence": 0.25,
        "qualitative_signal": 0.20,
        "workforce_intelligence": 0.15,
        "context_engine": 0.10
    },
    explanations=[
        "Final fused score: 0.58 (bullish)",
        "market_intelligence: strong positive contribution (+0.20)",
        "financial_analyst: moderate positive contribution (+0.16)",
        ...
    ]
)
```

---

### **Step 6: Conflict Detection**

**File:** `core/signal_fusion.py` (method: `detect_conflicts`)

**What it does:**
- Compares sentiment scores between agents
- Flags conflicts when difference ≥ 1.0

**Example:**
```python
conflicts = [
    {
        "agent_1": "financial_analyst",
        "agent_2": "workforce_intelligence",
        "sentiment_1": 0.85,
        "sentiment_2": -0.35,
        "difference": 1.20,
        "description": "financial_analyst (+0.85) conflicts with workforce_intelligence (-0.35)"
    }
]
```

---

### **Step 7: Debate & Consensus** (if conflicts exist)

**File:** `agents/strategic_orchestrator.py` (method: `_orchestrate_debate`)

**What happens:**
1. **Hypothesis** created from Agent 1's view
2. **Counterpoint** added from Agent 2's opposing view
3. **Debate** messages broadcast on agent bus
4. **Consensus** reached via weighted fusion
5. **Final recommendation** includes both perspectives

**Output:**
```python
Consensus(
    ticker="AAPL",
    final_recommendation="ACCUMULATE",
    fused_score=0.58,
    calibrated_confidence=0.79,
    participating_agents=[...],
    conflicting_points=[
        "financial_analyst (+0.85) conflicts with workforce_intelligence (-0.35)"
    ],
    signal_contributions={...},
    debate_rounds=1
)
```

---

### **Step 8: Confidence Calibration** (optional)

**File:** `core/confidence.py`

**What it does:**
- Adjusts raw confidence scores using calibration curves
- Ensures confidence matches empirical accuracy
- Uses isotonic regression or Platt scaling

**When used:**
- After training on historical predictions
- Requires outcome data (did prediction match reality?)

---

### **Step 9: Pattern Matching** (optional)

**File:** `core/pattern_miner.py`

**What it does:**
- Checks current signals against discovered patterns
- Example: "When Glassdoor < 3.2 AND hiring_freeze → earnings_miss (78% confidence)"

**When used:**
- By Context Engine for historical pattern matching
- After training on historical data

---

### **Step 10: Final Recommendation**

**File:** `agents/strategic_orchestrator.py`

**What it generates:**
```python
{
    "ticker": "AAPL",
    "action": "BUY",  # BUY, ACCUMULATE, HOLD, REDUCE, SELL
    "confidence": 0.79,
    "fused_score": 0.58,
    "reasoning": "Consensus from 5 agents",
    "signal_contributions": {
        "financial_analyst": 0.30,
        "market_intelligence": 0.25,
        ...
    },
    "supporting_evidence": [
        {
            "source": "finnhub_ratings",
            "description": "4 analysts, consensus: 0.75, target: $212.50",
            "confidence": 0.90
        },
        ...
    ],
    "conflicting_points": [...],
    "alerts": [
        "Multiple Analyst Downgrades",
        "Negative News Sentiment Spike"
    ]
}
```

**Action determination:**
```python
fused_score >= 0.4  → BUY
fused_score >= 0.1  → ACCUMULATE
fused_score >= -0.1 → HOLD
fused_score >= -0.4 → REDUCE
fused_score < -0.4  → SELL
```

---

### **Step 11: Streamlit Dashboard Display**

**File:** `apps/dashboard.py` (to be implemented)

**What it shows:**

**Page 1: Company Search**
- Input box for ticker
- Search button
- Company info display

**Page 2: Analysis Results**
- **Final Recommendation Card**
  - Action (BUY/SELL/HOLD)
  - Confidence score
  - Fused sentiment score

- **Agent Breakdown**
  - Table showing each agent's sentiment and confidence
  - Contribution percentage
  - Status (agree/conflict)

- **Signal Fusion Visualization**
  - Pie chart of signal weights
  - Bar chart of agent contributions

- **Evidence & Alerts**
  - Top 5 evidence items with sources
  - Any alerts flagged by agents

- **Timeline (optional)**
  - Historical sentiment trend
  - Pattern matches

---

## Data Flow: Live vs Sample Mode

### **Sample Mode** (Default: `LIVE_CONNECTORS=false`)

```
User → Orchestrator → Agents → Tools → data/samples/*.json
                                           ↓
                                    Return sample data
                                           ↓
                              Agents process → Fusion
                                           ↓
                                  Dashboard displays
```

**Advantages:**
- No API keys needed
- No rate limits
- Fast (local file reads)
- Deterministic results for demos

### **Live Mode** (`LIVE_CONNECTORS=true`)

```
User → Orchestrator → Agents → Tools → External APIs
                                   ↓ (with rate limiting)
                              Real-time data
                                   ↓
                         Agents process → Fusion
                                   ↓
                          Dashboard displays
```

**Requires:**
- API keys in `.env`
- Internet connection
- Respects rate limits
- Non-deterministic (real market data)

---

## Component Dependencies

### **Orchestrator depends on:**
- All 5 agents
- Signal fusion module
- Agent bus

### **Agents depend on:**
- Tools (for data fetching)
- Agent bus (for A2A communication)
- LangSmith (for tracing)

### **Tools depend on:**
- Resilience layer (rate limiting, retries)
- Sample data files (in sample mode)
- External APIs (in live mode)

### **Signal Fusion depends on:**
- Agent outputs
- NumPy (for calculations)

### **Dashboard depends on:**
- Orchestrator (for running analysis)
- Streamlit (for UI)
- All underlying components

---

## Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| **Entry Point** | `apps/dashboard.py` | Streamlit UI  |
| **Coordination** | `agents/strategic_orchestrator.py` | Manages entire workflow |
| **Agent 1** | `agents/financial_analyst.py` | Financial metrics |
| **Agent 2** | `agents/qualitative_signal.py` | Sentiment analysis |
| **Agent 3** | `agents/context_engine.py` | Historical patterns |
| **Agent 4** | `agents/workforce_intelligence.py` | Employee signals |
| **Agent 5** | `agents/market_intelligence.py` | Market signals |
| **Communication** | `core/agent_bus.py` | Pub/sub message bus |
| **Fusion** | `core/signal_fusion.py` | Weighted ensemble |
| **Confidence** | `core/confidence.py` | Calibration |
| **Patterns** | `core/pattern_miner.py` | Correlation mining |
| **Resilience** | `core/resilience.py` | Rate limiting, retries |
| **Tools** | `tools/edgar_tool.py` | SEC filings |
| | `tools/news_api_tool.py` | News articles |
| | `tools/finnhub_tool.py` | Analyst ratings |
| **Sample Data** | `data/samples/**/*.json` | Demo fixtures |

---

## Running the System

### **Quick Start (Sample Mode)**

```bash
# 1. Activate environment
source .investment-iq-env/bin/activate

# 2. Ensure .env exists
cp .env.example .env
# Edit .env: LIVE_CONNECTORS=false

# 3. Run analysis (Python)
python -c "
import asyncio
from agents.strategic_orchestrator import StrategicOrchestratorAgent
from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent

async def main():
    workforce = WorkforceIntelligenceAgent()
    market = MarketIntelligenceAgent()

    orchestrator = StrategicOrchestratorAgent(
        agent_id='orchestrator',
        financial_agent=None,  # Legacy, can be None
        qualitative_agent=None,
        context_agent=None,
        workforce_agent=workforce,
        market_agent=market
    )

    result = await orchestrator.process({
        'ticker': 'AAPL',
        'company_name': 'Apple Inc.',
        'sector': 'Technology'
    })

    print(result.data['recommendation'])

asyncio.run(main())
"

# 4. Run dashboard (when implemented)
streamlit run apps/dashboard.py
```

### **Production (Live Mode)**

```bash
# 1. Set up API keys in .env
LIVE_CONNECTORS=true
FINNHUB_API_KEY=your_real_key
NEWS_API_KEY=your_real_key

# 2. Run with live data
streamlit run apps/dashboard.py
```

---

## System Features

### **Observability**
- LangSmith tracing on all agents
- Message history in agent bus
- Workflow logging in orchestrator

### **Resilience**
- Rate limiting per API
- Exponential backoff retries
- Circuit breakers for fault tolerance

### **Compliance**
- No web scraping
- Rate limits enforced
- Sample data fallback
- Legal disclaimers

### **Explainability**
- SHAP-like contribution scores
- Evidence with sources and confidence
- Debate history for conflicts
- Signal fusion explanations

---

## The Streamlit dashboard:

1. **UI Layout**
   - Sidebar for ticker input
   - Main area for results
   - Tabs for different views

2. **Analysis Trigger**
   - Button to run analysis
   - Progress indicator
   - Error handling

3. **Result Display**
   - Recommendation card
   - Agent breakdown table
   - Charts (signal weights, sentiment)
   - Evidence list
   - Alerts

4. **Caching**
   - Cache analysis results
   - Avoid re-running on page refresh


---

Last Updated: 2025-01-15
