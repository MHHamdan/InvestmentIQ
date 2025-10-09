# 📊 InvestmentIQ - AI-Powered Investment Analysis Platform

**Group 2: Capstone Project**  
*Team Members: Mohammed, Rui, Ameya, Amine, Rajesh, Murthy*

---

## 🎯 Project Overview

InvestmentIQ is a transparent AI investment analysis platform that leverages **Google Agent Development Kit (ADK)** and **Gemini 2.0 Flash** to provide explainable, data-driven stock recommendations.

### Key Features
- 🤖 **4 Specialist AI Agents** running in parallel
- 🔍 **Complete Transparency** - see reasoning, key factors, and data sources
- 📊 **Real-time Data** from FMP, EODHD, and FRED APIs
- 🧮 **Custom Signal Fusion** with weighted averaging
- 🎨 **Modern Dashboard** built with Streamlit

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
├── tests/                           # Testing & Results
│   ├── test_adk_orchestrator.py     # Orchestrator test suite
│   ├── test_results_*.json          # Cached analysis results
│   └── test_summary.md              # Test results summary
│
├── reference/                       # Legacy LangGraph implementation
│   └── agents/                      # Original agent implementations
│
├── .env.example                     # Environment template with API instructions
├── .env                            # Your API keys (not in git)
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

## 🔧 API Configuration

### Free Tier Limits
| API      | Free Tier Limit       | Usage in Project                    |
|----------|-----------------------|-------------------------------------|
| FMP      | 250 requests/day      | ~4 requests per stock analysis      |
| EODHD    | 20 requests/day       | 1 request per stock                 |
| FRED     | Unlimited             | ~3 requests per analysis            |
| Gemini   | 50 requests/day       | 4 requests per stock (one per agent)|

**Note**: With Gemini's 50 req/day limit, you can analyze ~12 stocks per day on free tier.

---

## 🧪 Running Tests

```bash
# Run orchestrator test on a single ticker
python tests/test_adk_orchestrator.py

# The test will:
# 1. Run all 4 agents in parallel
# 2. Display transparent analysis
# 3. Save results to tests/test_results_TICKER.json
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

## 🔄 Migration from LangGraph

This project evolved from a LangGraph-based architecture to Google ADK:

**Why we migrated:**
1. ❌ LangGraph InMemoryRunner API issues (session management bugs)
2. ✅ Direct Gemini API with structured outputs (more reliable)
3. ✅ Simpler codebase (~200 lines per agent vs. ~500)
4. ✅ Better transparency with Pydantic schemas

**Legacy code**: Available in `/reference/` folder

---

## 🤝 Contributing

This is a capstone project by Group 2. For questions or feedback:
- Review the code
- Check test results in `/tests/`
- Examine API documentation in `.env.example`

---

## 📝 License

Academic project - Group 2 Capstone  
Created: October 2025  
Last Updated: Oct 8, 2025 21:29

---

## 🙏 Acknowledgments

- **Google Gemini** for AI capabilities
- **FMP, EODHD, FRED** for financial data
- **Streamlit** for rapid dashboard development
- **Group 2 Team** for collaborative development
