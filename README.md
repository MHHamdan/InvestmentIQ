# 📊 InvestmentIQ - AI-Powered Investment Analysis Platform

**Version 2.0** - *ADK Architecture with LangSmith Observability*

**Group 2: Capstone Project**
*Team Members: Mohammed, Rui, Ameya, Amine, Rajesh, Murthy*

---

## 🎯 Project Overview

InvestmentIQ is a transparent AI investment analysis platform that leverages **Google Agent Development Kit (ADK)** and **Gemini 2.0 Flash** to provide explainable, data-driven stock recommendations with complete observability.

### Key Features
- 🤖 **4 Specialist AI Agents** running in parallel
- 🔍 **Complete Transparency** - see reasoning, key factors, and data sources
- 📊 **Real-time Data** from FMP, EODHD, and FRED APIs
- 🧮 **Custom Signal Fusion** with weighted averaging
- 🎨 **Modern Dashboard** built with Streamlit
- 🔭 **LangSmith Observability** - full tracing and debugging

---

## 🏗️ Architecture

### System Design
```
┌─────────────────────────────────────────────────────┐
│                  ADK Orchestrator                    │
│            (Parallel Agent Execution)                │
└─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────┐
        ▼                 ▼                 ▼          ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Financial   │  │   Market     │  │ Qualitative  │  │   Context    │
│   Analyst    │  │ Intelligence │  │    Signal    │  │    Engine    │
│  (35% wt.)   │  │  (30% wt.)   │  │  (25% wt.)   │  │  (10% wt.)   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │                │
        ▼                 ▼                 ▼                ▼
    FMP API          FMP API          EODHD API         FMP + FRED
  (Financials)      (Analysts)          (News)          (Macro)
        │                 │                 │                │
        └─────────────────┴─────────────────┴────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  Gemini 2.0 Flash    │
                      │  (AI Analysis)       │
                      └──────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  Signal Fusion       │
                      │  (Weighted Average)  │
                      └──────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  Final Score         │
                      │  + Recommendation    │
                      └──────────────────────┘
```

### Technology Stack
- **AI**: Google Gemini 2.0 Flash, Google ADK
- **Observability**: LangSmith (tracing, debugging, monitoring)
- **Data Sources**: FMP, EODHD, FRED
- **Backend**: Python 3.12, asyncio
- **Frontend**: Streamlit, Plotly
- **Data Validation**: Pydantic

---

## 📁 Project Structure

```
InvestmentIQ/
├── agents/                          # ADK Agent Implementation
│   ├── adk_financial_analyst.py     # Financial metrics analysis (35% weight)
│   ├── adk_market_intelligence.py   # Analyst consensus analysis (30% weight)
│   ├── adk_qualitative_signal.py    # News sentiment analysis (25% weight)
│   ├── adk_context_engine.py        # Macro/sector analysis (10% weight)
│   └── adk_orchestrator.py          # Parallel agent coordinator
│
├── apps/                            # User Interface
│   └── dashboard.py                 # Streamlit dashboard with transparency
│
├── utils/                           # Utilities & Observability
│   ├── langsmith_tracer.py          # LangSmith tracing decorators & functions
│   └── add_tracing.py               # Batch script to add tracing to agents
│
├── tests/                           # Testing & Evaluation
│   ├── test_adk_orchestrator.py     # Orchestrator test suite
│   ├── test_results_*.json          # Cached analysis results (5 stocks)
│   ├── test_summary.md              # Test results summary
│   ├── evaluate_agents.py           # Evaluation suite with accuracy metrics
│   ├── eval_dataset.json            # Ground truth for 7 stocks
│   ├── eval_results.json            # Evaluation output with metrics
│   └── EVALUATION_REPORT.md         # Comprehensive evaluation analysis
│
├── core/                            # Core contracts & fusion
│   ├── agent_contracts.py           # Pydantic models for agent outputs
│   └── signal_fusion.py             # Custom weighted averaging fusion
│
├── tools/                           # API integration tools
│   └── fmp_tool.py                  # Financial Modeling Prep API client
│
├── .env.example                     # Environment template with API instructions
├── .env                            # Your API keys (not in git)
├── LANGSMITH_INTEGRATION.md         # LangSmith setup & usage guide
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- pip or uv for package management

### 2. Clone & Setup
```bash
# Clone the repository
cd ZADK_Capstone/InvestmentIQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit plotly google-generativeai pydantic python-dotenv aiohttp httpx
```

### 3. Configure API Keys
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use any text editor
```

**Required API Keys:**
1. **FMP** (Financial Modeling Prep) - [Get Free Key](https://site.financialmodelingprep.com/developer/docs)
   - 250 requests/day free tier

2. **EODHD** (News API) - [Get Free Key](https://eodhd.com/register)
   - 20 requests/day free tier

3. **FRED** (Federal Reserve) - [Get Free Key](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Unlimited with registration

4. **Google Gemini** - [Get Free Key](https://aistudio.google.com/apikey)
   - 50 requests/day free tier

5. **LangSmith** (Optional - Observability) - [Get Free Key](https://smith.langchain.com/)
   - 5,000 traces/month free tier
   - Enables full tracing, debugging, and monitoring

### 4. Run the Dashboard
```bash
streamlit run apps/dashboard.py
```

Visit: http://localhost:8501

---

## 📂 File Explanations

### **Agents** (`/agents/`)

#### `adk_financial_analyst.py` (163 lines)
**Purpose**: Analyzes fundamental financial health  
**Data Source**: FMP API  
**Key Metrics**:
- Revenue growth, margins (gross, operating, net)
- Profitability ratios (ROE, ROA)
- Leverage ratios (debt-to-equity, interest coverage)
- Valuation multiples (P/E, P/B, EV/EBITDA)

**Process**:
1. Fetches financial ratios from FMP
2. Calculates key metrics
3. Sends to Gemini for AI analysis
4. Returns sentiment (-1 to +1), confidence, reasoning, and key factors

**Weight in Final Score**: 35%

---

#### `adk_market_intelligence.py` (232 lines)
**Purpose**: Analyzes Wall Street analyst consensus  
**Data Source**: FMP API  
**Key Metrics**:
- Analyst ratings (buy/sell/hold counts)
- Price targets (average, high, low)
- Upside potential (current price vs. target)

**Process**:
1. Fetches analyst recommendations from FMP
2. Calculates consensus metrics
3. Gemini analyzes Wall Street sentiment
4. Returns structured analysis with transparency

**Weight in Final Score**: 30%

---

#### `adk_qualitative_signal.py` (246 lines)
**Purpose**: Analyzes news sentiment and market themes  
**Data Source**: EODHD News API  
**Key Metrics**:
- News sentiment (positive/negative ratio)
- Article count and themes
- Recent developments and catalysts

**Process**:
1. Fetches last 10 news articles from EODHD
2. Gemini analyzes sentiment and extracts themes
3. Returns sentiment score with reasoning

**Weight in Final Score**: 25%

---

#### `adk_context_engine.py` (346 lines)
**Purpose**: Analyzes macroeconomic and sector context  
**Data Sources**: FMP (sector) + FRED (macro)  
**Key Metrics**:
- GDP growth rate
- Unemployment rate
- Federal funds rate (interest rates)
- Sector performance

**Process**:
1. Fetches macro indicators from FRED
2. Fetches sector data from FMP
3. Gemini analyzes economic context
4. Returns macro sentiment

**Notable Fix**: GDP calculation now correctly computes growth rate from quarterly values instead of returning absolute GDP

**Weight in Final Score**: 10%

---

#### `adk_orchestrator.py` (230 lines)
**Purpose**: Coordinates all 4 agents and fuses signals  
**Key Features**:
- Parallel agent execution with `asyncio.gather()`
- Weighted signal fusion (35% + 30% + 25% + 10%)
- Confidence blending
- Error handling and logging

**Process**:
1. Runs all 4 agents in parallel
2. Collects sentiment scores and confidence
3. Applies weighted averaging:
   ```
   Final Score = (0.35 × Financial) + (0.30 × Market) + (0.25 × Qualitative) + (0.10 × Context)
   ```
4. Returns comprehensive result with all agent outputs

---

### **Dashboard** (`/apps/`)

#### `dashboard.py` (600+ lines)
**Purpose**: Interactive Streamlit dashboard with complete transparency  
**Key Features**:
- **Sidebar**: Project info, team members, architecture details
- **Input**: Centered ticker search
- **Results Display**:
  - Company header with sector
  - Final score gauge (percentage scale)
  - Agent contribution waterfall chart
  - Expandable agent cards showing:
    - All metrics (properly formatted with $, %, decimals)
    - Gemini's complete reasoning
    - Key factors that influenced the score
    - Data sources
  - Fusion engine details (formula + calculations)
  - JSON export

**Formatting Intelligence**:
- Percentages: `46.20%` (margins, growth, rates)
- Currency: `$385.00B`, `$1,234.56`
- Ratios: `1.23` (debt-to-equity, ROE)
- Integers: `45` (analyst counts)

**Design**: Clean, modern UI inspired by Apple/Google design principles

---

### **Utilities** (`/utils/`)

#### `langsmith_tracer.py`
**Purpose**: LangSmith observability decorators and logging functions
**Key Features**:
- `@trace_agent()` - Traces entire agent execution with metadata
- `@trace_step()` - Traces individual workflow steps (data fetching, processing)
- `@trace_llm_call()` - Traces Gemini API calls with token usage
- `log_metrics()` - Logs extracted metrics to LangSmith
- `log_api_call()` - Logs external API calls (FMP, EODHD, FRED) with response times
- `log_error()` - Logs errors with context for debugging

**Integration**: Automatically enabled when `LANGSMITH_TRACING=true` in `.env`

**What's Traced**:
- Agent inputs (ticker, company, sector)
- Agent outputs (sentiment, confidence, reasoning, key factors)
- All Gemini prompts and responses
- API response times and status codes
- Metric extraction steps
- Error stack traces with context

**View Traces**: https://smith.langchain.com/o/default/projects/p/investmentiq-adk

#### `add_tracing.py`
**Purpose**: Batch utility to add LangSmith tracing to agents
**Usage**: Automatically adds decorators and import statements to agent files
**When to use**: When adding new agents or updating tracing patterns

---

### **Tests** (`/tests/`)

#### `test_adk_orchestrator.py`
**Purpose**: Integration testing for orchestrator and agents
**Features**:
- Tests full pipeline for multiple tickers
- Displays transparent output (metrics, reasoning, key factors)
- Saves results to JSON files
- Generates summary report

**Test Results** (as of Oct 8, 2025):
| Ticker | Score  | Recommendation |
|--------|--------|----------------|
| MSFT   | +0.485 | STRONG BUY     |
| AMZN   | +0.477 | STRONG BUY     |
| AAPL   | +0.230 | BUY            |
| TSLA   | +0.125 | HOLD           |
| BA     | -0.162 | HOLD (Bearish) |

---

#### `evaluate_agents.py` ⭐ NEW
**Purpose**: Comprehensive agent evaluation suite with LangSmith integration
**Features**:
- Ground truth dataset for 7 stocks (AAPL, MSFT, NVDA, AMZN, META, TSLA, BA)
- Automated accuracy measurement with 3 key metrics:
  - **Sentiment MAE** (Mean Absolute Error) - measures prediction accuracy
  - **Directional Accuracy** - % of sentiment directions matched (positive/negative/neutral)
  - **Recommendation Match** - % of recommendations matched (BUY/HOLD/SELL)
- Pass/fail thresholds: MAE < 0.30, Directional ≥ 70%, Recommendation ≥ 60%
- LangSmith tracing for all evaluation runs
- JSON output with detailed results

**Evaluation Results** (as of Oct 8, 2025):
| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| Sentiment MAE | 0.295 | < 0.30 | ✅ PASS |
| Directional Accuracy | 57.1% | ≥ 70% | ❌ FAIL |
| Recommendation Match | 71.4% | ≥ 60% | ✅ PASS |
| **Overall** | **2/3** | **3/3** | **⚠️ NEEDS IMPROVEMENT** |

**Key Findings**:
- ✅ Strong performance on stocks with full API access (AAPL, MSFT, NVDA - all correct)
- ⚠️ API quota limitations affected 4/7 stocks (defaulted to HOLD/neutral)
- 🎯 Accurate when data available - MAE 0.295 is excellent
- 🔧 Directional accuracy impacted by Gemini free tier limit (10 req/min)

**Usage**:
```bash
python tests/evaluate_agents.py
```

**Full Report**: See [tests/EVALUATION_REPORT.md](tests/EVALUATION_REPORT.md) for comprehensive analysis and recommendations

---

#### `eval_dataset.json`
**Purpose**: Ground truth dataset with expected sentiments for evaluation
**Content**: 7 stocks with analyst-consensus-based expected sentiments, recommendations, and rationales

#### `EVALUATION_REPORT.md`
**Purpose**: Comprehensive evaluation analysis and findings
**Content**:
- Detailed results table with predictions vs. expected
- Strengths and limitations analysis
- API quota impact assessment
- Recommendations for v2.1, v2.2, v3.0 improvements

---

## 🔧 API Configuration

### Free Tier Limits
| API       | Free Tier Limit       | Usage in Project                    |
|-----------|-----------------------|-------------------------------------|
| FMP       | 250 requests/day      | ~4 requests per stock analysis      |
| EODHD     | 20 requests/day       | 1 request per stock                 |
| FRED      | Unlimited             | ~3 requests per analysis            |
| Gemini    | 50 requests/day       | 4 requests per stock (one per agent)|
| LangSmith | 5,000 traces/month    | 5 traces per stock (orchestrator + 4 agents) |

**Note**: With Gemini's 50 req/day limit, you can analyze ~12 stocks per day on free tier. LangSmith allows ~1,000 stocks/month on free tier.

---

## 🧪 Running Tests

### Integration Tests
```bash
# Run orchestrator test on a single ticker
python tests/test_adk_orchestrator.py

# The test will:
# 1. Run all 4 agents in parallel
# 2. Display transparent analysis
# 3. Save results to tests/test_results_TICKER.json
```

### Agent Evaluation ⭐ NEW
```bash
# Run comprehensive evaluation suite
python tests/evaluate_agents.py

# The evaluation will:
# 1. Analyze 7 stocks against ground truth
# 2. Calculate 3 accuracy metrics (MAE, Directional, Recommendation)
# 3. Trace all runs in LangSmith
# 4. Save results to tests/eval_results.json
# 5. Generate pass/fail report

# View detailed report
cat tests/EVALUATION_REPORT.md
```

---

## 🚨 Troubleshooting

### Issue: Gemini API Quota Exhausted
**Error**: `429 RESOURCE_EXHAUSTED`  
**Solution**: Dashboard will automatically load cached results from `tests/` folder if available. Quota resets daily.

### Issue: GDP showing as 23,770% instead of ~3%
**Status**: ✅ Fixed in v2.0  
**Solution**: Context Engine now calculates growth rate from quarterly GDP values

### Issue: Environment variables not loading
**Solution**: Ensure `load_dotenv()` is called at the top of orchestrator before importing agents

---

## 📊 Understanding the Output

### Sentiment Score
- **Range**: -1.0 (very bearish) to +1.0 (very bullish)
- **Scale**:
  - `> +0.5`: STRONG BUY
  - `+0.2 to +0.5`: BUY
  - `-0.2 to +0.2`: HOLD
  - `-0.5 to -0.2`: SELL
  - `< -0.5`: STRONG SELL

### Confidence Score
- **Range**: 0.0 to 1.0
- Higher confidence = more reliable data and consistent signals

### Transparency Features
Every agent provides:
1. **Reasoning**: Step-by-step explanation of how Gemini calculated the score
2. **Key Factors**: Top 3 metrics that influenced the decision
3. **Data Sources**: Where the information came from

---

## 🔭 LangSmith Observability (v2.0 Feature)

### What is LangSmith?
LangSmith provides complete observability for AI agents - see exactly what happens during each analysis.

### What's Traced?
Every analysis creates a hierarchical trace showing:
```
📊 InvestmentIQ Analysis (AAPL)
├── 🔗 Orchestrator (parallel execution)
│   ├── 🔗 Financial Analyst Agent
│   │   ├── 🔧 Fetch FMP Financial Data (0.245s)
│   │   ├── 📊 Extract Metrics (6 metrics)
│   │   └── 🤖 Gemini Analysis (tokens: 150/200)
│   ├── 🔗 Market Intelligence Agent
│   │   ├── 🔧 Fetch FMP Analyst Data (0.312s)
│   │   └── 🤖 Gemini Analysis (tokens: 180/220)
│   ├── 🔗 Qualitative Signal Agent
│   │   ├── 🔧 Fetch EODHD News (0.421s)
│   │   └── 🤖 Gemini Analysis (tokens: 200/250)
│   └── 🔗 Context Engine Agent
│       ├── 🔧 Fetch FRED Macro Data (0.189s)
│       └── 🤖 Gemini Analysis (tokens: 160/190)
└── 🧮 Signal Fusion (weighted average)
```

### Benefits
1. **Debugging** - See exact prompts sent to Gemini and responses received
2. **Performance** - Identify slow API calls and bottlenecks
3. **Cost Tracking** - Monitor Gemini token usage per analysis
4. **Error Analysis** - Full stack traces with context for failures
5. **Transparency** - Audit trail of all agent decisions

### How to Enable
Already configured in `.env`:
```bash
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=investmentiq-adk
LANGSMITH_TRACING=true
```

**View Traces**: https://smith.langchain.com/

### Documentation
See [LANGSMITH_INTEGRATION.md](LANGSMITH_INTEGRATION.md) for complete setup guide and usage examples.

---

## 🔄 Migration from LangGraph

This project evolved from a LangGraph-based architecture to Google ADK:

**Why we migrated:**
1. ❌ LangGraph InMemoryRunner API issues (session management bugs)
2. ✅ Direct Gemini API with structured outputs (more reliable)
3. ✅ Simpler codebase (~200 lines per agent vs. ~500)
4. ✅ Better transparency with Pydantic schemas
5. ✅ **v2.0 Enhancement**: Added LangSmith observability for complete tracing

**Architecture Comparison:**
- **LangGraph (v0.x)**: Complex orchestration, ~500 lines/agent, opaque execution
- **ADK v1.0**: Direct Gemini API, ~200 lines/agent, transparent with reasoning
- **ADK v2.0**: + LangSmith tracing for full observability

---

## 🤝 Contributing

This is a capstone project by Group 2. For questions or feedback:
- Review the code
- Check test results in `/tests/`
- Examine API documentation in `.env.example`

---

## 📝 Version History

### v2.0 (Current) - Oct 8, 2025
- ✨ **NEW**: LangSmith observability integration
- ✨ **NEW**: Complete tracing for all agents and API calls
- ✨ **NEW**: Token usage tracking and performance monitoring
- ✨ **NEW**: Agent evaluation suite with 3 accuracy metrics
- 📊 Evaluation results: 71.4% recommendation accuracy, MAE 0.295
- 📈 Ground truth dataset for 7 major stocks
- 🎨 Modern dashboard with Apple/Google-inspired UI
- 🔧 Fixed GDP calculation bug (now shows growth rate)
- 📊 Intelligent metric formatting with $, %, decimals
- 🔍 Enhanced transparency with reasoning and key factors

### v1.0 - Oct 6, 2025
- 🚀 Initial ADK architecture implementation
- 🤖 4 specialist agents with Gemini 2.0 Flash
- 📊 Custom signal fusion engine
- 📈 Real-time data from FMP, EODHD, FRED APIs

---

## 📝 License

Academic project - Group 2 Capstone
Created: October 2025
Last Updated: Oct 8, 2025 22:30

---

## 🙏 Acknowledgments

- **Google Gemini** for AI capabilities
- **LangSmith** for observability and debugging
- **FMP, EODHD, FRED** for financial data
- **Streamlit** for rapid dashboard development
- **Group 2 Team** for collaborative development
