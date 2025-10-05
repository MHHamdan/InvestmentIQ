# InvestmentIQ MVAS 

Multi-Agent Value Assessment System for Investment Intelligence

**Powered by LangGraph + HuggingFace**

## Overview

InvestmentIQ is a multi-agent system for investment analysis, combining 5 specialized AI agents with LangGraph orchestration and HuggingFace AI enhancements.

### Key Features

- **LangGraph Orchestration** - Declarative workflow with automatic state management
- **5 Specialist Agents** - Financial, Qualitative, Context, Workforce, Market Intelligence
- **HuggingFace AI** - FinBERT sentiment analysis + BART text generation
- **Signal Fusion** - Weighted ensemble with SHAP-like explanations
- **Debate Mechanism** - Automatic conflict detection and resolution
- **MCP Tools** - Edgar, NewsAPI, Finnhub integrations
- **Free Models** - $0 cost using HuggingFace free tier

## Quick Start

### 1. Install Dependencies

```bash
# Using pip (recommended)
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

**Note:** Requires Python 3.8+. All dependencies including LangGraph are in `requirements.txt`.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required:
```bash
HUGGING_FACE_API_KEY=your_key_here
```

### 3. Run Dashboard

```bash
streamlit run apps/dashboard.py
```

Navigate to `http://localhost:8501` and analyze any ticker!

## Architecture

### LangGraph Workflow

```
START
  ↓
Financial Analysis
  ↓
Qualitative Analysis
  ↓
Context Analysis
  ↓
Workforce Intelligence
  ↓
Market Intelligence
  ↓
Signal Fusion
  ↓
[Conflict Router]
  ├─→ Debate (if conflicts) ─→ Recommendation
  └─→ Recommendation (if aligned)
  ↓
END
```

### Agent Roles

1. **Financial Analyst** - Revenue, margins, cash flow analysis
2. **Qualitative Signal** - News sentiment, social signals
3. **Context Engine** - Historical patterns, sector trends
4. **Workforce Intelligence** - Employee sentiment, hiring trends
5. **Market Intelligence** - SEC filings, analyst ratings, news

### AI Enhancements

- **FinBERT** (`ProsusAI/finbert`) - Financial sentiment classification
- **BART** (`facebook/bart-large-cnn`) - Evidence summarization & reasoning

## Project Structure

```
investment_iq/
├── agents/                      # Agent implementations
│   ├── strategic_orchestrator.py    # LangGraph orchestrator (273 lines)
│   ├── financial_analyst.py
│   ├── qualitative_signal.py
│   ├── context_engine.py
│   ├── workforce_intelligence.py
│   └── market_intelligence.py
├── core/                        # Core components
│   ├── investment_graph.py          # LangGraph workflow (700+ lines)
│   ├── signal_fusion.py             # Signal fusion with BART
│   ├── agent_contracts.py           # Pydantic schemas
│   ├── agent_bus.py                 # Pub/sub messaging
│   └── confidence.py                # Confidence calibration
├── tools/                       # MCP tools
│   ├── edgar_tool.py                # SEC filings
│   ├── news_api_tool.py             # News with FinBERT
│   └── finnhub_tool.py              # Analyst ratings
├── utils/                       # Utilities
│   ├── hf_client.py                 # HuggingFace API client
│   ├── llm_factory.py               # LLM factory
│   └── observability.py             # Tracing
├── apps/                        # Applications
│   └── dashboard.py                 # Streamlit dashboard
├── data/samples/                # Sample data
├── backup_manual/               # Old implementation (backed up)
└── *.md                         # Documentation
```

## Documentation

### Main Documentation

- **`SYSTEM_OVERVIEW.md`** - Complete system architecture and design
- **`LANGGRAPH_INTEGRATION.md`** - LangGraph implementation details
- **`LANGGRAPH_QUICKSTART.md`** - Quick reference guide
- **`FINAL_STATUS.md`** - Current system status and features

### Backup Documentation

See `backup_manual/` for:
- Old manual orchestration code
- Migration documentation
- Test scripts
- HuggingFace setup details

## Usage

### Programmatic API

```python
from agents.strategic_orchestrator import StrategicOrchestratorAgent

# Initialize (agents created automatically)
orchestrator = StrategicOrchestratorAgent(...)

# Analyze ticker
response = await orchestrator.process({
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
})

# Access results
recommendation = response.data["recommendation"]
print(f"Action: {recommendation['action']}")
print(f"Confidence: {recommendation['confidence']:.2%}")
print(f"Reasoning: {recommendation['reasoning']}")
```

### Dashboard

```bash
streamlit run apps/dashboard.py
```

Features:
- Interactive ticker analysis
- Real-time sentiment (FinBERT)
- AI-generated reasoning (BART)
- Signal breakdown visualization
- Workflow tracking

## Configuration

### Environment Variables

```bash
# Required
HUGGING_FACE_API_KEY=your_key_here

# Optional (for live data)
NEWS_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
LIVE_CONNECTORS=false  # Set to true for live data

# LangSmith (optional - for tracing)
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
```

### Data Modes

**Sample Mode (Default):**
- Uses `data/samples/` for demo data
- No API keys required (except HuggingFace)
- Fast and reliable

**Live Mode:**
- Set `LIVE_CONNECTORS=true`
- Add NEWS_API_KEY and FINNHUB_API_KEY
- Real-time data from APIs

## Development

### Running Tests

```bash
# Verify HuggingFace setup
python backup_manual/verify_hf.py

# Test LangGraph workflow
python backup_manual/test_langgraph.py

# Test HuggingFace integration
python backup_manual/test_hf_integration.py
```

### Code Quality

- **Type Hints** - Full type safety with Pydantic
- **Async/Await** - Efficient async execution
- **Error Handling** - Graceful fallbacks
- **Logging** - Comprehensive logging

## Performance

- **Analysis Time:** 5-10 seconds per ticker
- **Memory Usage:** ~500MB (including models)
- **Cost:** $0 (free HuggingFace models)
- **Scalability:** 100s of concurrent workflows

## Technology Stack

- **Python** 3.8+
- **LangGraph** 0.5.0+ (workflow orchestration)
- **LangChain** (observability)
- **HuggingFace** (AI models)
- **Streamlit** (dashboard)
- **Plotly** (visualizations)
- **Pydantic** (data validation)
- **Asyncio** (async execution)



## Acknowledgments

- **LangGraph** - For declarative workflow orchestration
- **HuggingFace** - For free AI models
- **LangChain** - For observability tools
- **Streamlit** - For dashboard framework

---

**InvestmentIQ** -  Multi-Agent Investment Analysis

