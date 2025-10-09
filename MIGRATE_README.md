# ZADK_Capstone - InvestmentIQ ADK Edition

**Clean start for ADK + A2A + Custom Fusion + LangSmith architecture**

---

## What Was Copied from ZX_Capstone

### ✅ Essential Files (Keep & Use)

#### **Configuration**
- `.env` - Your API keys (READY TO USE)
- `.env.example` - Template for team
- `.gitignore` - Git ignore rules
- `requirements.txt` - Will be modified (remove LangGraph, add LangSmith)

#### **Tools (Data Fetching) - USE AS-IS**
All tools remain unchanged - they fetch data from APIs:
- `tools/fmp_tool.py` - Financial Modeling Prep integration
- `tools/finnhub_tool.py` - Finnhub market data
- `tools/edgar_tool.py` - SEC filings
- `tools/news_api_tool.py` - News aggregation
- `tools/data_tools.py` - Data utilities
- `tools/rag_context_tool.py` - RAG context

#### **Core Business Logic - KEEP**
- `core/signal_fusion.py` - **YOUR PROVEN ALGORITHM** (keep!)
- `core/agent_contracts.py` - Type definitions (may simplify)

---

### ⚠️ Reference Files (Extract Logic Only)

These files are for reference to extract business logic when creating ADK agents:

#### **reference/agents/**
- `qualitative_signal.py` - Already ADK-based! Use as template
- `financial_analyst.py` - Extract financial analysis logic
- `market_intelligence.py` - Extract market analysis logic
- `context_engine.py` - Extract context analysis logic
- `workforce_intelligence.py` - Extract workforce analysis logic

#### **reference/apps/**
- `dashboard.py` - UI structure reference

#### **reference/utils/**
- `observability.py` - Tracing patterns for LangSmith

---

## What Was LEFT BEHIND (No Baggage)

❌ **LangGraph orchestration**
- `core/investment_graph.py` - State machine (replaced by ADK)
- `agents/strategic_orchestrator.py` - LangGraph wrapper (replaced by ADK)
- `core/agent_bus.py` - Event bus (ADK handles this)

❌ **Base classes**
- `agents/base_agent.py` - ADK provides its own base

❌ **Unused utilities**
- `utils/hf_client.py` - HuggingFace (not needed with Gemini)
- `utils/llm_factory.py` - ADK initializes LLMs
- `utils/logger.py` - Use standard logging

---

## Next Steps - What to Create

### **1. Update requirements.txt**

Remove:
```
langgraph>=0.5.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
langgraph-checkpoint>=2.1.0
```

Add:
```
langsmith>=0.1.0  # Observability
```

Keep:
```
google-adk
google-adk[mcp]
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
requests>=2.31.0
aiohttp>=3.9.0
# ... all other non-LangGraph deps
```

---

### **2. Create ADK Agents**

Create these new files:

```
agents/
├── adk_financial_analyst.py      # NEW - Based on reference/agents/financial_analyst.py
├── adk_qualitative_signal.py     # NEW - Based on reference/agents/qualitative_signal.py (already ADK!)
├── adk_market_intelligence.py    # NEW - Based on reference/agents/market_intelligence.py
├── adk_context_engine.py         # NEW - Based on reference/agents/context_engine.py
├── adk_workforce_intelligence.py # NEW - Based on reference/agents/workforce_intelligence.py
└── adk_orchestrator.py           # NEW - Main orchestrator with A2A
```

**Template (based on qualitative_signal.py):**

```python
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from pydantic import BaseModel, Field

class FinancialAnalysis(BaseModel):
    sentiment_score: float = Field(description="Score between -1.0 and 1.0")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")
    # ... other fields

financial_agent = Agent(
    name="financial_analyst",
    model="gemini-2.5-flash",
    instruction="""
    Analyze financial statements for {ticker}.
    Focus on revenue growth, profit margins, debt ratios.
    Provide sentiment score and confidence.
    """,
    output_schema=FinancialAnalysis
)
```

---

### **3. Create Orchestrator with A2A**

```python
# agents/adk_orchestrator.py

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from langsmith import traceable

orchestrator = Agent(
    name="investment_orchestrator",
    model="gemini-2.5-flash",
    instruction="""
    You are coordinating 5 analyst agents to analyze {ticker}.

    1. Consult all 5 agents for their analysis
    2. If agents disagree significantly (>0.3 difference), facilitate debate
    3. Return all agent outputs as structured data for fusion
    """,
    transfer_to=[
        financial_agent,
        qualitative_agent,
        context_agent,
        workforce_agent,
        market_agent
    ],
    output_schema=MultiAgentAnalysis
)

@traceable(name="Investment Analysis")
async def analyze_stock(ticker: str, company_name: str):
    runner = InMemoryRunner(agent=orchestrator)
    result = await runner.run(
        user_message=f"Analyze {ticker} ({company_name})",
        session_state={"ticker": ticker, "company_name": company_name}
    )
    return result
```

---

### **4. Integrate Custom Fusion**

```python
# After ADK orchestrator returns agent outputs

from core.signal_fusion import SignalFusion

# Get agent outputs from ADK
agent_outputs = extract_from_adk_result(result)

# Use your proven fusion algorithm
fusion_engine = SignalFusion(method="weighted_average")
fused_signal = fusion_engine.fuse(ticker, agent_outputs)

# Map to recommendation (your thresholds)
if fused_signal.score >= 0.30:
    recommendation = "BUY"
elif fused_signal.score >= 0.15:
    recommendation = "ACCUMULATE"
# ... etc
```

---

### **5. Update Dashboard**

Modify `apps/dashboard.py` (use reference/apps/dashboard.py as starting point):

```python
# OLD (LangGraph)
from agents.strategic_orchestrator import StrategicOrchestratorAgent
orchestrator = StrategicOrchestratorAgent(...)

# NEW (ADK)
from agents.adk_orchestrator import analyze_stock
result = await analyze_stock(ticker, company_name)
```

---

### **6. Add LangSmith Tracing**

```python
import os
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_key"
os.environ["LANGSMITH_PROJECT"] = "InvestmentIQ-ADK"

from langsmith import traceable

@traceable(name="Full Analysis")
async def run_analysis(ticker: str):
    # All nested calls auto-traced
    result = await analyze_stock(ticker)
    fused = signal_fusion.fuse(result.agent_outputs)
    return recommendation
```

---

## Architecture Comparison

### **Old (ZX_Capstone):**
```
User → Dashboard → Strategic Orchestrator → LangGraph → 5 Agents (sequential) → Signal Fusion → Recommendation
```

**Files:** ~1000 lines across 5 files

---

### **New (ZADK_Capstone):**
```
User → Dashboard → ADK Orchestrator → 5 ADK Agents (A2A) → Custom Fusion → Recommendation
                                                ↓
                                          LangSmith Tracing
```

**Files:** ~150 lines in 1-2 files + your proven fusion algorithm

---

## Benefits of New Architecture

✅ **Simpler:** 80% less orchestration code
✅ **Smarter:** LLM-based agent collaboration with A2A
✅ **Reliable:** Your proven fusion math (deterministic)
✅ **Observable:** LangSmith full tracing
✅ **Maintainable:** Single framework (ADK), less complexity

---

## File Structure (After Implementation)

```
ZADK_Capstone/InvestmentIQ/
├── .env                          ✅ Ready to use
├── .env.example                  ✅ Ready to use
├── .gitignore                    ✅ Ready to use
├── requirements.txt              ⚠️  Update (remove LangGraph, add LangSmith)
│
├── tools/                        ✅ Use as-is
│   ├── fmp_tool.py
│   ├── finnhub_tool.py
│   ├── edgar_tool.py
│   └── news_api_tool.py
│
├── core/                         ✅ Keep these!
│   ├── signal_fusion.py          ← Your secret sauce
│   └── agent_contracts.py        ← May simplify
│
├── agents/                       🆕 Create these
│   ├── adk_financial_analyst.py
│   ├── adk_qualitative_signal.py
│   ├── adk_market_intelligence.py
│   ├── adk_context_engine.py
│   ├── adk_workforce_intelligence.py
│   └── adk_orchestrator.py
│
├── apps/                         🆕 Create/modify
│   └── dashboard.py              ← Modify from reference
│
└── reference/                    ℹ️  For logic extraction only
    ├── agents/                   ← Old agent code
    ├── apps/                     ← Old dashboard
    └── utils/                    ← Old utilities
```

---

## Development Workflow

1. ✅ **DONE:** Files copied to clean directory
2. **TODO:** Update requirements.txt
3. **TODO:** Create 5 ADK agents (start with qualitative as template)
4. **TODO:** Create ADK orchestrator with A2A
5. **TODO:** Integrate custom fusion engine
6. **TODO:** Update dashboard
7. **TODO:** Add LangSmith tracing
8. **TODO:** Test with AAPL, BA, AMZN
9. **TODO:** Document new architecture

---

**Created:** October 8, 2025
**Source:** ZX_Capstone/InvestmentIQ (LangGraph version)
**Target:** ZADK_Capstone/InvestmentIQ (ADK + A2A + Custom Fusion + LangSmith)
