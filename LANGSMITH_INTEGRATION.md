# LangSmith Integration - InvestmentIQ ADK

## ✅ Completed Integration

LangSmith observability has been successfully integrated into all InvestmentIQ agents for complete tracing and debugging.

### What's Included

1. **Tracer Utility** (`utils/langsmith_tracer.py`)
   - `@trace_agent()` - Traces entire agent execution
   - `@trace_step()` - Traces individual workflow steps
   - `@trace_llm_call()` - Traces Gemini API calls with token usage
   - `log_metrics()` - Logs extracted metrics
   - `log_api_call()` - Logs external API calls (FMP, EODHD, FRED)
   - `log_error()` - Logs errors with context

2. **All Agents Traced**
   - ✅ Financial Analyst
   - ✅ Market Intelligence  
   - ✅ Qualitative Signal
   - ✅ Context Engine
   - ✅ Orchestrator

3. **Tracked Information**
   - Agent inputs/outputs
   - Gemini API calls and prompts
   - Execution times
   - API response times (FMP, EODHD, FRED)
   - Errors and exceptions
   - Metric extraction steps

### Configuration

**Environment Variables** (already in `.env`):
```bash
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=investmentiq-adk
LANGSMITH_TRACING=true
```

### View Traces

Visit: **https://smith.langchain.com/o/default/projects/p/investmentiq-adk**

### What You'll See in LangSmith

Each analysis run creates a trace tree showing:

```
📊 investmentiq-adk Run
├── 🔗 orchestrator (chain)
│   ├── 🔗 financial_analyst_agent (chain)
│   │   ├── 🔧 fetch_fmp_financial_data (tool)
│   │   └── 🤖 gemini_analysis (llm)
│   ├── 🔗 market_intelligence_agent (chain)
│   │   ├── 🔧 fetch_fmp_analyst_data (tool)
│   │   └── 🤖 gemini_analysis (llm)
│   ├── 🔗 qualitative_signal_agent (chain)
│   │   ├── 🔧 fetch_eodhd_news (tool)
│   │   └── 🤖 gemini_analysis (llm)
│   └── 🔗 context_engine_agent (chain)
│       ├── 🔧 fetch_fred_macro_data (tool)
│       └── 🤖 gemini_analysis (llm)
```

### Trace Details Include

- **Inputs**: Ticker, company name, sector
- **Outputs**: Sentiment scores, confidence, reasoning, key factors
- **Metrics**: All extracted financial/market/macro metrics
- **API Calls**: Response times, status codes
- **Errors**: Full stack traces with context
- **Token Usage**: Gemini prompt and completion tokens

### Usage

Tracing is automatic when `LANGSMITH_TRACING=true`. No code changes needed.

**To disable tracing**:
```bash
# In .env
LANGSMITH_TRACING=false
```

### Free Tier Limits

- **5,000 traces/month** on free tier
- Each stock analysis = 5 traces (1 orchestrator + 4 agents)
- Can analyze ~1,000 stocks/month on free tier

### Example Trace Output

```json
{
  "run_type": "chain",
  "name": "financial_analyst_agent",
  "inputs": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
  },
  "outputs": {
    "sentiment": 0.5,
    "confidence": 0.8,
    "reasoning": "Strong margins and ROE indicate healthy profitability...",
    "key_factors": [
      "Gross margin: 46.2%",
      "ROE: 157%",
      "Low debt-to-equity: 1.97"
    ]
  },
  "metadata": {
    "agent": "financial_analyst",
    "ticker": "AAPL",
    "metrics_count": 6
  }
}
```

### Benefits

1. **Debugging** - See exact prompt, response, and timing for each Gemini call
2. **Monitoring** - Track API failures and performance bottlenecks
3. **Transparency** - Audit trail of all agent decisions
4. **Optimization** - Identify slow API calls or inefficient prompts

### Next Steps

1. Run an analysis from the dashboard
2. Visit LangSmith to see the trace
3. Explore the trace tree to understand agent flow
4. Use filters to find specific runs or errors

---

**Integration completed**: Oct 8, 2025  
**Team**: Group 2 (Murthy)
