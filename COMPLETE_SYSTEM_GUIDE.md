# **InvestmentIQ MVAS - Complete System Guide**
## **File-by-File Breakdown with End-to-End Example: Analyzing Apple Inc. (AAPL)**

---

## **📋 Table of Contents**
1. [Project Structure Overview](#project-structure-overview)
2. [Folder-by-Folder Breakdown](#folder-by-folder-breakdown)
3. [Complete End-to-End Example](#complete-end-to-end-example)
4. [Final Output Visualization](#final-output-visualization)

---

## **Project Structure Overview**

```
InvestmentIQ/
├── 📁 agents/          # The 5 specialist agents + orchestrator
├── 📁 apps/            # User interfaces (Streamlit dashboard)
├── 📁 config/          # Configuration & settings
├── 📁 core/            # Core framework (LangGraph, fusion, contracts)
├── 📁 data/            # Sample/mock data for demo mode
├── 📁 evaluation/      # Evaluation framework & metrics
├── 📁 tests/           # Unit & integration tests
├── 📁 tools/           # MCP-like data connectors
├── 📁 utils/           # Utilities (logging, HF client, observability)
├── run.py             # Main launcher script
├── requirements.txt   # Python dependencies
├── pyproject.toml     # Package configuration
└── README.md          # Documentation
```

---

## **Folder-by-Folder Breakdown**

### **1. `/agents/` - The Brain of the System**

Contains all agent implementations. Each agent is an independent analyst.

#### **`agents/__init__.py`** (27 lines)
**Purpose:** Package initialization, exports main agent classes

```python
from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentResponse
from .strategic_orchestrator import StrategicOrchestratorAgent
from .workforce_intelligence import WorkforceIntelligenceAgent
from .market_intelligence import MarketIntelligenceAgent
```

**What it does:** Makes agents importable as `from agents import StrategicOrchestratorAgent`

---

#### **`agents/base_agent.py`** (142 lines)
**Purpose:** Abstract base class for all agents - defines the contract

**Key Components:**

1. **AgentRole Enum:**
   ```python
   class AgentRole(Enum):
       FINANCIAL_ANALYST = "financial_analyst"
       QUALITATIVE_SIGNAL = "qualitative_signal"
       CONTEXT_ENGINE = "context_engine"
       STRATEGIC_ORCHESTRATOR = "strategic_orchestrator"
   ```

2. **AgentMessage Dataclass:**
   ```python
   @dataclass
   class AgentMessage:
       sender: str                    # Who sent it
       receiver: str                  # Who receives it
       message_type: MessageType      # REQUEST, RESPONSE, ERROR, INFO
       payload: Dict[str, Any]        # Actual data
       timestamp: str                 # When sent
       correlation_id: Optional[str]  # Link related messages
   ```

3. **AgentResponse Dataclass:**
   ```python
   @dataclass
   class AgentResponse:
       agent_id: str
       status: str           # "success" or "error"
       data: Dict[str, Any]  # Agent's output
       metadata: Dict        # Extra context
       timestamp: str
   ```

4. **BaseAgent Class:**
   ```python
   class BaseAgent(ABC):
       def __init__(self, agent_id: str, role: AgentRole):
           self.agent_id = agent_id
           self.role = role
           self.message_history: List[AgentMessage] = []

       @abstractmethod
       async def process(self, request: Dict) -> AgentResponse:
           """Every agent MUST implement this"""
           pass
   ```

**Real-world usage in example:**
```python
# When Financial Agent receives request for AAPL:
response = AgentResponse(
    agent_id="financial_analyst",
    status="success",
    data={
        "sentiment": 0.75,
        "metrics": {"revenue_growth": 0.15, "gross_margin": 0.42}
    },
    metadata={"sector": "Technology"},
    timestamp="2025-01-20T10:30:00Z"
)
```

---

#### **`agents/financial_analyst.py`** (236 lines)
**Purpose:** Analyzes hard financial metrics (revenue, margins, debt, cash flow)

**Key Methods:**
1. `analyze(ticker, company_name, sector)` - Main entry point
2. `_fetch_sample_data(ticker)` - Loads from `data/samples/financial/{ticker}_financial.json`
3. `_calculate_sentiment(data)` - Converts metrics to [-1, 1] score
4. `_build_evidence(data)` - Creates Evidence objects
5. `_generate_alerts(data, ticker)` - Flags concerning metrics

**Example execution for AAPL:**
```python
# Input
ticker = "AAPL"
company_name = "Apple Inc."

# Step 1: Fetch data
data = {
    "revenue_growth": 0.15,     # 15% YoY
    "gross_margin": 0.42,        # 42%
    "operating_margin": 0.25,    # 25%
    "debt_to_equity": 0.45,     # 0.45
    "roe": 0.22                 # 22%
}

# Step 2: Calculate sentiment
score = 0.0
if revenue_growth > 0.15: score += 0.3   # ✓ 0.3
if gross_margin > 0.4: score += 0.2      # ✓ 0.2
if operating_margin > 0.2: score += 0.2  # ✓ 0.2
if debt_to_equity < 0.5: score += 0.2    # ✓ 0.2
if roe > 0.15: score += 0.1              # ✓ 0.1
# Total: 1.0 (capped at 1.0) → sentiment = 1.0

# Step 3: Build evidence
evidence = [
    Evidence(
        source="financial_data",
        description="Strong revenue growth of 15.0%",
        confidence=0.9
    ),
    Evidence(
        source="financial_data",
        description="Healthy gross margin of 42.0%",
        confidence=0.85
    )
]

# Step 4: Broadcast observation via agent_bus
observation = Observation(
    agent_id="financial_analyst",
    ticker="AAPL",
    observation="Financial health shows 15.0% revenue growth with 42.0% gross margin",
    confidence=0.85
)
agent_bus.broadcast_observation(observation)  # → Other agents notified

# Step 5: Return output
return AgentOutput(
    signal=SignalType.FINANCIAL,
    agent_id="financial_analyst",
    ticker="AAPL",
    sentiment=1.0,
    confidence=0.85,
    metrics={"revenue_growth": 0.15, ...},
    evidence=[...]
)
```

---

#### **`agents/qualitative_signal.py`** (314 lines)
**Purpose:** Analyzes news sentiment, brand perception, market perception

**Example execution for AAPL:**
```python
# Loads data/samples/qualitative/aapl_qualitative.json
data = {
    "sentiment_score": 0.3,
    "news_count": 45,
    "sentiment_breakdown": {
        "positive": 0.55,
        "neutral": 0.25,
        "negative": 0.20
    },
    "reputation_score": 0.75,
    "market_perception": "positive",
    "key_themes": ["innovation", "growth", "ai"]
}

# Sentiment calculation
sentiment = (
    0.4 * 0.3 +                    # sentiment_score: 0.12
    0.3 * (0.55 - 0.20) +          # breakdown: 0.105
    0.2 * ((0.75 - 0.5) * 2) +     # reputation: 0.10
    0.1 * 0.2                      # perception boost: 0.02
) = 0.345

# Confidence based on news volume
news_count = 45 → confidence = 0.75  # (>20 but <50)

# Returns sentiment=0.345, confidence=0.75
```

---

#### **`agents/context_engine.py`** (318 lines)
**Purpose:** Finds historical patterns, identifies similar past scenarios

**Example execution for AAPL:**
```python
data = {
    "scenario_type": "strong_momentum",
    "pattern_matches": 8,
    "historical_accuracy": 0.75,
    "similar_situations": [
        {"ticker": "MSFT", "year": 2018, "outcome": "positive"},
        {"ticker": "GOOGL", "year": 2020, "outcome": "positive"}
    ],
    "market_cycle": "expansion",
    "sector_trend": "bullish"
}

# Scenario mapping
scenario_sentiment = {
    "strong_momentum": 0.7
}[scenario_type] = 0.7

# Market cycle boost
cycle_boost = {"expansion": 0.2}["expansion"] = 0.2

# Sector trend boost
trend_boost = {"bullish": 0.15}["bullish"] = 0.15

# Final sentiment
sentiment = (
    0.4 * 0.7 +   # scenario: 0.28
    0.2 * 0.2 +   # cycle: 0.04
    0.2 * 0.15    # trend: 0.03
) = 0.35

# Confidence from pattern matches
pattern_matches = 8 → base_confidence = 0.8
confidence = 0.8 * 0.75 (historical_accuracy) = 0.60

# Returns sentiment=0.35, confidence=0.60
```

---

#### **`agents/workforce_intelligence.py`** (401 lines)
**Purpose:** Analyzes employee satisfaction, hiring trends, turnover

**Example execution for AAPL:**
```python
data = {
    "rating": 3.8,              # Glassdoor-style (1-5)
    "rating_trend": "increasing",
    "review_count": 1250,
    "hiring_signal": "aggressive",
    "job_postings_count": 150,
    "churn_rate": 0.15          # 15%
}

# Normalize rating (1-5) to sentiment
rating_sentiment = (3.8 - 3.0) / 2.0 = 0.4

# Trend boosts
trend_boost = 0.1 (increasing) + 0.05 (hiring aggressive) = 0.15

# Final sentiment
sentiment = 0.5 * 0.4 + 0.2 * 0.15 = 0.23

# Confidence from review volume
review_count = 1250 → confidence = 0.7

# Returns sentiment=0.23, confidence=0.7
```

---

#### **`agents/market_intelligence.py`** (444 lines) ⭐ **Most Complex**
**Purpose:** Multi-source aggregator (SEC filings + News + Analyst ratings)

**Example execution for AAPL:**
```python
# Source 1: SEC EDGAR
filings = edgar_tool.get_recent_filings("AAPL")
# Returns 3 8-K filings
filing_sentiment = 0.30

# Source 2: NewsAPI + FinBERT
news = news_tool.get_company_news("AAPL", "Apple Inc.")
# Returns 25 articles
# Each article → FinBERT → sentiment score
# Average sentiment = 0.35

# Source 3: Finnhub Analysts
ratings = finnhub_tool.get_analyst_ratings("AAPL")
# 32 analysts: 25 buy, 5 hold, 2 sell
# Consensus sentiment = 0.60

# Source 4: Executive changes
execs = finnhub_tool.get_executive_changes("AAPL")
# Stability score = 0.85 → sentiment = 0.10

# Weighted aggregation
sentiment = (
    0.35 * 0.60 +  # analyst (highest weight): 0.21
    0.30 * 0.35 +  # news: 0.105
    0.25 * 0.30 +  # filings: 0.075
    0.10 * 0.10    # execs: 0.01
) = 0.40

# Data quality-based confidence = 0.78

# Returns sentiment=0.40, confidence=0.78
```

---

#### **`agents/strategic_orchestrator.py`** (273 lines)
**Purpose:** Coordinates all agents via LangGraph workflow

**Key Methods:**
1. `__init__(...)` - Creates LangGraph workflow by calling `create_investment_graph()`
2. `process(request)` - Main entry point, executes workflow
3. `get_graph()` - Returns compiled LangGraph for inspection
4. `get_workflow_log()` - Returns execution history

**Example execution for AAPL:**
```python
orchestrator = StrategicOrchestratorAgent(
    agent_id="orchestrator_001",
    financial_agent=FinancialAnalystAgent(),
    qualitative_agent=QualitativeSignalAgent(),
    context_agent=ContextEngineAgent(),
    workforce_agent=WorkforceIntelligenceAgent(),
    market_agent=MarketIntelligenceAgent()
)

# Execute workflow
response = await orchestrator.process({
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
})

# Internally calls:
# 1. graph.ainvoke(initial_state, config)
# 2. LangGraph executes all nodes
# 3. Returns final_state with recommendation
```

---

### **2. `/core/` - Framework Infrastructure**

The "engine" that powers the multi-agent system.

#### **`core/agent_contracts.py`** (154 lines)
**Purpose:** Pydantic models for type-safe data exchange

**Key Models:**
```python
class MessageType(str, Enum):
    OBSERVATION = "observation"
    ALERT = "alert"
    QUERY = "query"
    HYPOTHESIS = "hypothesis"
    COUNTERPOINT = "counterpoint"
    CONSENSUS = "consensus"

class SignalType(str, Enum):
    FINANCIAL = "financial"
    SENTIMENT = "sentiment"
    WORKFORCE = "workforce"
    MARKET_INTELLIGENCE = "market_intelligence"
    CONTEXT = "context"

class AgentMessage(BaseModel):
    message_type: MessageType
    sender: str
    receiver: Optional[str] = None  # None = broadcast
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = Field(ge=1, le=5)

class Evidence(BaseModel):
    source: str
    value: Any
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str]

class AgentOutput(BaseModel):
    signal: SignalType
    agent_id: str
    ticker: str
    metrics: Dict[str, Any]
    sentiment: float = Field(ge=-1.0, le=1.0)  # Validated!
    confidence: float = Field(ge=0.0, le=1.0)  # Validated!
    evidence: List[Evidence]

class FusedSignal(BaseModel):
    ticker: str
    final_score: float
    confidence: float
    agent_signals: Dict[str, AgentOutput]
    signal_weights: Dict[str, float]
    explanations: List[str]
    llm_summary: Optional[str]  # BART-generated

class Consensus(BaseModel):
    participating_agents: List[str]
    final_recommendation: str
    fused_score: float
    conflicting_points: List[str]
    debate_rounds: int
```

**Why this matters:** Type safety prevents bugs. If you try to set `sentiment=1.5`, Pydantic raises error.

---

#### **`core/agent_bus.py`** (268 lines)
**Purpose:** Pub/sub message bus for A2A communication

**Key Methods:**
```python
class AgentBus:
    def subscribe(self, message_type: MessageType, handler: Callable):
        """Agent subscribes to message type"""

    async def publish(self, message: AgentMessage):
        """Broadcast message to all subscribers"""

    def broadcast_observation(self, observation: Observation):
        """Convenience method for observations"""

    def start_debate(self, ticker: str, hypothesis: Hypothesis) -> str:
        """Initiate debate when agents conflict"""

    def reach_consensus(self, consensus: Consensus):
        """Broadcast final consensus"""

    def get_message_history(self, limit=100) -> List[AgentMessage]:
        """Audit trail"""
```

**Example in AAPL analysis:**
```python
# Financial agent broadcasts observation
observation = Observation(
    agent_id="financial_analyst",
    ticker="AAPL",
    observation="Strong revenue growth detected",
    confidence=0.85
)
agent_bus.broadcast_observation(observation)

# Internally:
# 1. Creates AgentMessage with MessageType.OBSERVATION
# 2. Stores in message_history
# 3. Calls all subscribed handlers
# 4. Other agents can react (if they subscribed)
```

---

#### **`core/investment_graph.py`** (518 lines) ⭐ **LangGraph Definition**
**Purpose:** Defines the entire workflow as a state graph

**Key Components:**

1. **State Schema:**
   ```python
   class InvestmentAnalysisState(TypedDict):
       ticker: str
       company_name: str
       agent_outputs: Annotated[List[AgentOutput], operator.add]
       fused_signal: Optional[FusedSignal]
       recommendation: Optional[Dict]
       workflow_log: Annotated[List[Dict], operator.add]
   ```

2. **Node Functions:**
   - `financial_node(state)` → Calls financial agent
   - `qualitative_node(state)` → Calls qualitative agent
   - `context_node(state)` → Calls context agent
   - `workforce_node(state)` → Calls workforce agent
   - `market_node(state)` → Calls market agent
   - `fusion_node(state)` → Fuses all signals
   - `debate_router(state)` → Decides debate vs direct recommendation
   - `debate_node(state)` → Resolves conflicts
   - `recommendation_node(state)` → Generates final output

3. **Graph Construction:**
   ```python
   workflow = StateGraph(InvestmentAnalysisState)
   workflow.add_node("financial_analysis", financial_node)
   # ... add all nodes
   workflow.set_entry_point("financial_analysis")
   workflow.add_edge("financial_analysis", "qualitative_analysis")
   # ... sequential edges
   workflow.add_conditional_edges("signal_fusion", debate_router, {...})
   workflow.add_edge("recommendation", END)

   app = workflow.compile(checkpointer=MemorySaver())
   ```

**AAPL Example Flow:**
```
State Init: {ticker="AAPL", agent_outputs=[], ...}
  ↓
financial_node → {agent_outputs=[FinancialOutput(sentiment=1.0)], ...}
  ↓
qualitative_node → {agent_outputs=[Financial, Qualitative(sentiment=0.345)], ...}
  ↓
context_node → {agent_outputs=[Financial, Qualitative, Context(sentiment=0.35)], ...}
  ↓
workforce_node → {agent_outputs=[..., Workforce(sentiment=0.23)], ...}
  ↓
market_node → {agent_outputs=[..., Market(sentiment=0.40)], ...}
  ↓
fusion_node → {fused_signal=FusedSignal(final_score=0.512), requires_debate=False}
  ↓
debate_router → "recommendation" (no conflicts)
  ↓
recommendation_node → {recommendation={"action": "BUY", ...}}
  ↓
END
```

---

#### **`core/signal_fusion.py`** (442 lines)
**Purpose:** Weighted ensemble fusion with SHAP-like explanations

**Key Methods:**
```python
class SignalFusion:
    def fuse(self, ticker, agent_outputs, weights=None) -> FusedSignal:
        """Main fusion logic"""

    def detect_conflicts(self, agent_signals, threshold=1.0) -> List[Dict]:
        """Find disagreements"""

    def _weighted_average_fusion(self, agent_signals, weights) -> float:
        """Calculate fused score"""

    def _generate_explanations(self, ...) -> List[str]:
        """SHAP-like contributions"""

    def _generate_llm_evidence_summary(self, ticker, evidence, final_score) -> str:
        """Call BART for natural language summary"""
```

**AAPL Example:**
```python
# All 5 agent outputs collected
agent_outputs = [
    AgentOutput(agent_id="financial", sentiment=1.0, confidence=0.85),
    AgentOutput(agent_id="qualitative", sentiment=0.345, confidence=0.75),
    AgentOutput(agent_id="context", sentiment=0.35, confidence=0.60),
    AgentOutput(agent_id="workforce", sentiment=0.23, confidence=0.70),
    AgentOutput(agent_id="market", sentiment=0.40, confidence=0.78)
]

# Apply default weights
weights = {
    "financial": 0.30,
    "market": 0.25,
    "qualitative": 0.20,
    "workforce": 0.15,
    "context": 0.10
}

# Calculate fused score
final_score = (
    1.0 * 0.30 +
    0.40 * 0.25 +
    0.345 * 0.20 +
    0.23 * 0.15 +
    0.35 * 0.10
) = 0.512

# Generate explanations
explanations = [
    "Final fused score: 0.512 (bullish)",
    "financial: strong positive (+0.300), sentiment=1.00, confidence=0.85",
    "market: moderate positive (+0.100), sentiment=0.40, confidence=0.78",
    "qualitative: weak positive (+0.069), sentiment=0.345, confidence=0.75",
    "workforce: weak positive (+0.035), sentiment=0.23, confidence=0.70",
    "context: weak positive (+0.035), sentiment=0.35, confidence=0.60"
]

# Call BART for LLM summary
prompt = """Summarize investment evidence for AAPL in 2-3 sentences.
Overall signal is bullish with score of 0.512.

Evidence:
- Strong revenue growth of 15%
- Healthy gross margin of 42%
- 32 analysts, consensus: 0.60
"""
llm_summary = hf_client.generate_text(prompt, model="facebook/bart-large-cnn")
# → "Apple demonstrates strong financial health with robust revenue growth..."

# Detect conflicts
conflicts = []  # No conflicts (all sentiments aligned)
```

---

### **3. `/tools/` - Data Connectors (MCP-like)**

Tools that fetch data from external sources.

#### **`tools/edgar_tool.py`** (7.4 KB)
**Purpose:** Fetch SEC filings (8-K material events)

```python
class EdgarTool:
    async def get_recent_filings(self, ticker: str) -> Dict:
        """Fetch recent 8-K filings"""

    def extract_material_events(self, filings_data: Dict) -> List[Dict]:
        """Extract material events from filings"""

    def categorize_events(self, events: List) -> Dict:
        """Categorize into financial_results, leadership_changes, strategic_actions"""

    def calculate_event_sentiment(self, events: List) -> float:
        """Assign sentiment to events"""
```

**AAPL Example:**
```python
filings = {
    "filings": [
        {
            "type": "8-K",
            "date": "2025-01-15",
            "description": "Item 2.02 - Results of Operations (Q4 Earnings)"
        }
    ]
}

events = [
    {
        "type": "financial_results",
        "description": "Q4 earnings beat expectations",
        "sentiment": 0.5
    }
]

categorized = {
    "financial_results": [events[0]],
    "leadership_changes": [],
    "strategic_actions": []
}

filing_sentiment = 0.30  # Weighted average
```

---

#### **`tools/news_api_tool.py`** (14.8 KB)
**Purpose:** Fetch news + FinBERT sentiment analysis

```python
class NewsAPITool:
    async def get_company_news(self, ticker: str, company_name: str) -> Dict:
        """Fetch recent news articles"""

    def calculate_sentiment_metrics(self, news_data: Dict) -> Dict:
        """Aggregate sentiments from all articles"""

    def categorize_articles(self, news_data: Dict) -> Dict:
        """Group by topic"""
```

**AAPL Example:**
```python
# Fetches 25 articles
articles = [
    {
        "title": "Apple announces record iPhone sales",
        "description": "Apple Inc. reported...",
        "publishedAt": "2025-01-20"
    },
    # ... 24 more
]

# For each article, calls FinBERT
for article in articles:
    text = article["title"] + " " + article["description"]
    sentiment_result = hf_client.classify_sentiment(text, model="ProsusAI/finbert")
    # Returns: [{"label": "positive", "score": 0.89}]

# Aggregates
metrics = {
    "avg_sentiment": 0.35,
    "sentiment_trend": "improving",
    "positive_ratio": 0.60,
    "negative_ratio": 0.15,
    "article_count": 25
}
```

---

#### **`tools/finnhub_tool.py`** (11.6 KB)
**Purpose:** Analyst ratings + executive changes

```python
class FinnhubTool:
    async def get_analyst_ratings(self, ticker: str) -> Dict:
        """Fetch analyst recommendations"""

    def calculate_analyst_sentiment(self, ratings_data: Dict) -> Dict:
        """Convert ratings to sentiment"""

    async def get_executive_changes(self, ticker: str) -> Dict:
        """Track leadership changes"""
```

**AAPL Example:**
```python
ratings = {
    "consensus": {
        "avg_target": 195.50,
        "buy": 25,
        "hold": 5,
        "sell": 2
    }
}

# Sentiment calculation
bullish_ratio = 25 / 32 = 0.78
sentiment = (bullish_ratio - 0.5) * 2 = 0.56

# Executive changes
execs = {
    "recent_changes": [],
    "stability_score": 0.85
}
```

---

#### **`tools/rag_context_tool.py`** (9.7 KB)
**Purpose:** Retrieval-Augmented Generation using Pinecone

```python
class RAGContextTool:
    def __init__(self):
        self.pinecone_client = ...
        self.embeddings = get_llm_factory().create_embeddings()

    async def query_context(self, query: str, top_k=5) -> List[Dict]:
        """Retrieve relevant historical context"""
```

---

#### **`tools/data_tools.py`** (Various helper tools)
**Purpose:** Additional data access utilities for financial and qualitative data processing

---

### **4. `/utils/` - Infrastructure Utilities**

#### **`utils/hf_client.py`** (177 lines)
**Purpose:** Direct HuggingFace API client

```python
class HuggingFaceClient:
    def generate_text(self, prompt, model="facebook/bart-large-cnn", max_tokens=200):
        """Call BART for text generation"""

    def classify_sentiment(self, text, model="ProsusAI/finbert"):
        """Call FinBERT for sentiment"""
```

**AAPL Example:**
```python
# FinBERT sentiment
result = hf_client.classify_sentiment(
    "Apple announces record iPhone sales",
    model="ProsusAI/finbert"
)
# Returns: {"status": "success", "data": [{"label": "positive", "score": 0.89}]}

# BART text generation
summary = hf_client.generate_text(
    prompt="Summarize: Apple shows strong revenue growth...",
    model="facebook/bart-large-cnn"
)
# Returns: "Apple demonstrates strong financial health with..."
```

---

#### **`utils/llm_factory.py`** (250 lines)
**Purpose:** Abstract LLM provider selection

```python
class LLMFactory:
    def create_chat_model(self, provider="huggingface", model_name=None):
        """Create LangChain-compatible chat model"""

    def create_embeddings(self, provider="huggingface"):
        """Create embeddings for RAG"""

    def get_available_providers(self) -> List[str]:
        """Return list of configured providers"""
```

**Key Features:**
- Supports HuggingFace (free), OpenAI, Anthropic
- Automatic provider selection based on API keys
- Default models: `google/flan-t5-large` for chat, `all-MiniLM-L6-v2` for embeddings

---

#### **`utils/observability.py`** (182 lines)
**Purpose:** LangSmith tracing integration

```python
class ObservabilityManager:
    def __init__(self):
        self.enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    def create_feedback(self, run_id, score, comment=None):
        """Collect user feedback for a run"""

    def get_project_metrics(self) -> Dict[str, Any]:
        """Get aggregated project metrics"""

# Decorator for agent tracing
@trace_agent("financial_analyst", {"version": "1.0"})
async def analyze_financials(data):
    # Automatically traced in LangSmith
    pass

# Decorator for A2A message tracing
@trace_a2a_message("request")
def send_request(message):
    # Traced as A2A communication
    pass

# Decorator for tool usage tracing
@trace_tool("financial_data_tool")
def get_financial_data(company_id):
    # Traced as tool usage
    pass
```

**When enabled:** All agent calls, A2A messages, and tool usages are automatically sent to LangSmith for visualization and debugging.

---

#### **`utils/logger.py`**
**Purpose:** Centralized logging configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

---

#### **`utils/ui_components.py`**
**Purpose:** Reusable UI components for Streamlit dashboard

---

### **5. `/apps/` - User Interfaces**

#### **`apps/dashboard.py`** (15.7 KB)
**Purpose:** Streamlit web dashboard

**Key Components:**

1. **Agent Initialization:**
   ```python
   @st.cache_resource
   def initialize_agents():
       financial = FinancialAnalystAgent()
       qualitative = QualitativeSignalAgent()
       context = ContextEngineAgent()
       workforce = WorkforceIntelligenceAgent()
       market = MarketIntelligenceAgent()
       orchestrator = StrategicOrchestratorAgent(
           financial_agent=financial,
           qualitative_agent=qualitative,
           context_agent=context,
           workforce_agent=workforce,
           market_agent=market
       )
       return orchestrator
   ```

2. **User Input:**
   ```python
   ticker = st.text_input("Enter Stock Ticker", "AAPL")
   company_name = st.text_input("Company Name", "Apple Inc.")
   sector = st.selectbox("Sector", ["Technology", "Finance", ...])

   if st.button("Analyze"):
       # Run analysis
   ```

3. **Visualization:**
   - Signal contribution breakdown (plotly bar chart)
   - Sentiment gauges
   - Evidence cards
   - Workflow timeline
   - LLM-generated summary display

---

### **6. `/config/` - Configuration**

#### **`config/settings.py`** (93 lines)
**Purpose:** Centralized configuration

```python
class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "mock"
    LOG_DIR = PROJECT_ROOT / "logs"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # Agent configuration
    AGENT_CONFIG = {
        "financial_analyst": {
            "agent_id": "financial_analyst_001",
            "timeout_seconds": 30,
            "retry_attempts": 3
        },
        # ... configs for each agent
    }

    # Analysis thresholds
    THRESHOLDS = {
        "high_margin": 40.0,
        "moderate_margin": 30.0,
        "high_debt_ratio": 1.0,
        "moderate_debt_ratio": 0.5,
        "strong_sentiment_threshold": 0.6,
        "weak_sentiment_threshold": -0.6,
        "high_confidence": 0.7,
        "moderate_confidence": 0.5
    }

    # Workflow configuration
    WORKFLOW_CONFIG = {
        "enable_parallel_execution": True,
        "conflict_detection_enabled": True,
        "detailed_logging": True,
        "save_workflow_history": True
    }

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

---

### **7. `/data/` - Sample Data**

#### **`data/mock/`**
Contains sample JSON files for demo mode:
- `COMPANY_X_financial.json` - Sample financial metrics
- `COMPANY_X_qualitative.txt` - Sample qualitative data
- `context_rules.json` - Pattern matching rules

**Example `COMPANY_X_financial.json`:**
```json
{
  "ticker": "COMPANY_X",
  "revenue": 1000000000,
  "revenue_growth": 0.15,
  "gross_profit": 420000000,
  "gross_margin": 0.42,
  "operating_income": 250000000,
  "operating_margin": 0.25,
  "net_income": 200000000,
  "net_margin": 0.20,
  "total_debt": 450000000,
  "total_equity": 1000000000,
  "debt_to_equity": 0.45,
  "current_ratio": 1.8,
  "roe": 0.22,
  "cash_flow_positive": true
}
```

---

### **8. `/evaluation/` - Evaluation Framework**

#### **`evaluation/evaluators.py`** (Custom evaluators)
**Purpose:** LangSmith custom evaluators for quality assessment

```python
class InvestmentIQEvaluators:
    def financial_analysis_structure(self, run, example) -> EvaluationResult:
        """
        Check if financial analysis has proper structure.
        Validates presence of: financial_health, key_ratios, risk_level
        """

    def sentiment_detection_accuracy(self, run, example) -> EvaluationResult:
        """
        Validate sentiment score is in valid range [-1, 1]
        Check for sentiment_score, risk_level, theme extraction
        """

    def recommendation_consistency(self, run, example) -> EvaluationResult:
        """
        Check if final recommendation aligns with fused score
        BUY → score > 0.4, SELL → score < -0.4, etc.
        """

    def conflict_resolution_effectiveness(self, run, example) -> EvaluationResult:
        """
        Evaluate debate mechanism when conflicts detected
        Check if consensus is reached, conflicting points addressed
        """
```

**Usage with LangSmith:**
```python
from langsmith import Client
from evaluation.evaluators import InvestmentIQEvaluators

client = Client()
evaluators = InvestmentIQEvaluators()

# Evaluate a dataset
results = client.evaluate(
    dataset_name="investment-test-cases",
    evaluators=[
        evaluators.financial_analysis_structure,
        evaluators.sentiment_detection_accuracy,
        evaluators.recommendation_consistency
    ]
)
```

---

#### **`evaluation/dataset_creator.py`**
**Purpose:** Create test datasets for evaluation

```python
class DatasetCreator:
    def create_test_cases(self) -> List[Dict]:
        """Generate test cases for evaluation"""

    def add_ground_truth(self, ticker: str, expected_action: str):
        """Add expected outcomes for validation"""
```

---

### **9. `/tests/` - Unit Tests**

#### **`tests/test_agents.py`**
**Purpose:** Test agent functionality

```python
import pytest
from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent

@pytest.mark.asyncio
async def test_financial_analyst():
    agent = FinancialAnalystAgent()
    output = await agent.analyze("AAPL", "Apple Inc.")

    # Validate output structure
    assert -1 <= output.sentiment <= 1
    assert 0 <= output.confidence <= 1
    assert output.signal == SignalType.FINANCIAL
    assert len(output.evidence) > 0

@pytest.mark.asyncio
async def test_qualitative_agent():
    agent = QualitativeSignalAgent()
    output = await agent.analyze("AAPL", "Apple Inc.")

    assert output.signal == SignalType.SENTIMENT
    assert "sentiment_score" in output.metrics

@pytest.mark.asyncio
async def test_orchestrator_workflow():
    orchestrator = StrategicOrchestratorAgent(...)
    response = await orchestrator.process({
        "ticker": "AAPL",
        "company_name": "Apple Inc."
    })

    assert response.status == "success"
    assert "recommendation" in response.data
    assert response.data["recommendation"]["action"] in ["BUY", "SELL", "HOLD", "ACCUMULATE", "REDUCE"]
```

---

### **10. Root Files**

#### **`run.py`** (31 lines)
**Purpose:** Main launcher script

```python
"""
InvestmentIQ MVAS - Launcher Script

Launches the Streamlit dashboard.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    dashboard_path = project_root / "apps" / "dashboard.py"

    print("🚀 Launching InvestmentIQ MVAS Dashboard...")
    print(f"📊 Dashboard: {dashboard_path}")
    print("\nPress Ctrl+C to stop the dashboard\n")

    # Launch streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path)
    ])

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python run.py
# Opens browser at http://localhost:8501
```

---

#### **`requirements.txt`**
**Purpose:** Python dependencies

Key dependencies:
- `langgraph>=0.5.0` - Workflow orchestration
- `langchain>=0.1.0` - LangChain framework
- `langchain-huggingface>=0.0.1` - HuggingFace integration
- `huggingface-hub>=0.20.0` - HF API client
- `sentence-transformers>=2.2.0` - Embeddings
- `streamlit>=1.28.0` - Dashboard
- `plotly>=5.17.0` - Visualizations
- `pydantic>=2.7.4` - Data validation
- `pinecone>=5.0.0` - Vector DB

---

#### **`pyproject.toml`**
**Purpose:** Package configuration

```toml
[project]
name = "investment-iq"
version = "1.0.0"
description = "InvestmentIQ MVAS - Multi-Agent Financial Intelligence System"
requires-python = ">=3.8"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.black]
line-length = 88
```

---

## **Complete End-to-End Example**

### **Scenario: Analyzing Apple Inc. (AAPL)**

#### **User Action:**
```python
# User opens dashboard at http://localhost:8501
# Enters: ticker="AAPL", company_name="Apple Inc.", sector="Technology"
# Clicks "Analyze"
```

---

### **Step-by-Step Execution:**

#### **Step 1: Dashboard calls orchestrator** (`apps/dashboard.py`)
```python
orchestrator = initialize_agents()  # Cached
response = await orchestrator.process({
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
})
```

---

#### **Step 2: Orchestrator initializes LangGraph state** (`agents/strategic_orchestrator.py`)
```python
initial_state = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "agent_outputs": [],
    "workflow_log": [],
    "errors": []
}

# Execute graph
config = {"configurable": {"thread_id": "AAPL"}}
final_state = await self.graph.ainvoke(initial_state, config)
```

---

#### **Step 3: LangGraph executes nodes sequentially** (`core/investment_graph.py`)

**Node 1: financial_node**
```python
# Calls financial_agent.analyze()
output = await financial_agent.analyze("AAPL", "Apple Inc.", "Technology")

# agents/financial_analyst.py
# Loads data/samples/financial/aapl_financial.json
data = {
    "revenue_growth": 0.15,
    "gross_margin": 0.42,
    "operating_margin": 0.25,
    "debt_to_equity": 0.45,
    "roe": 0.22
}

# Calculates sentiment = 1.0, confidence = 0.85

# Broadcasts observation via agent_bus
agent_bus.broadcast_observation(Observation(
    agent_id="financial_analyst",
    ticker="AAPL",
    observation="Financial health shows 15.0% revenue growth with 42.0% gross margin",
    confidence=0.85
))

# Returns state update
return {
    "agent_outputs": [AgentOutput(sentiment=1.0, confidence=0.85, ...)],
    "financial_analysis": {"sentiment": 1.0, "confidence": 0.85, ...},
    "workflow_log": [{"node": "financial_analysis", "status": "completed"}]
}
```

---

**Node 2: qualitative_node**
```python
# Calls qualitative_agent.analyze()
output = await qualitative_agent.analyze("AAPL", "Apple Inc.", "Technology")

# agents/qualitative_signal.py
data = {
    "sentiment_score": 0.3,
    "news_count": 45,
    "sentiment_breakdown": {"positive": 0.55, "neutral": 0.25, "negative": 0.20},
    "reputation_score": 0.75,
    "market_perception": "positive"
}

# Sentiment = 0.345, confidence = 0.75

return {
    "agent_outputs": [output],  # Appended to list
    "qualitative_analysis": {...},
    "workflow_log": [{"node": "qualitative_analysis", "status": "completed"}]
}
```

---

**Node 3: context_node**
```python
# agents/context_engine.py
data = {
    "scenario_type": "strong_momentum",
    "pattern_matches": 8,
    "historical_accuracy": 0.75,
    "market_cycle": "expansion",
    "sector_trend": "bullish"
}

# Sentiment = 0.35, confidence = 0.60

return {
    "agent_outputs": [output],
    "context_analysis": {...},
    "workflow_log": [...]
}
```

---

**Node 4: workforce_node**
```python
# agents/workforce_intelligence.py
data = {
    "rating": 3.8,
    "rating_trend": "increasing",
    "review_count": 1250,
    "hiring_signal": "aggressive",
    "churn_rate": 0.15
}

# Sentiment = 0.23, confidence = 0.70

return {
    "agent_outputs": [output],
    "workforce_analysis": {...},
    "workflow_log": [...]
}
```

---

**Node 5: market_node**
```python
# agents/market_intelligence.py
# Aggregates from 3 sources:

# 1. SEC EDGAR
filings = await edgar_tool.get_recent_filings("AAPL")
filing_sentiment = 0.30

# 2. NewsAPI + FinBERT
news = await news_tool.get_company_news("AAPL", "Apple Inc.")
# 25 articles → FinBERT → avg sentiment = 0.35

# 3. Finnhub
ratings = await finnhub_tool.get_analyst_ratings("AAPL")
# 32 analysts: 25 buy, 5 hold, 2 sell → sentiment = 0.60

# 4. Executives
execs = await finnhub_tool.get_executive_changes("AAPL")
# Stability = 0.85 → sentiment = 0.10

# Weighted aggregation
sentiment = (0.35 * 0.60) + (0.30 * 0.35) + (0.25 * 0.30) + (0.10 * 0.10) = 0.40
confidence = 0.78

return {
    "agent_outputs": [output],
    "market_intelligence": {...},
    "workflow_log": [...]
}
```

**After all 5 nodes:**
```python
state["agent_outputs"] = [
    FinancialOutput(sentiment=1.0, confidence=0.85),
    QualitativeOutput(sentiment=0.345, confidence=0.75),
    ContextOutput(sentiment=0.35, confidence=0.60),
    WorkforceOutput(sentiment=0.23, confidence=0.70),
    MarketOutput(sentiment=0.40, confidence=0.78)
]
```

---

**Node 6: fusion_node** (`core/signal_fusion.py`)
```python
# Collect all agent outputs
agent_outputs = state["agent_outputs"]  # All 5

# Apply default weights
weights = {
    "financial_analyst": 0.30,
    "market_intelligence": 0.25,
    "qualitative_signal": 0.20,
    "workforce_intelligence": 0.15,
    "context_engine": 0.10
}

# Calculate fused score
final_score = (
    1.0 * 0.30 +      # financial: 0.300
    0.40 * 0.25 +     # market: 0.100
    0.345 * 0.20 +    # qualitative: 0.069
    0.23 * 0.15 +     # workforce: 0.035
    0.35 * 0.10       # context: 0.035
) = 0.539

# Calculate confidence (weighted + agreement)
avg_confidence = 0.74
agreement = 0.934  # Low variance
confidence = 0.7 * 0.74 + 0.3 * 0.934 = 0.798

# Generate SHAP explanations
explanations = [
    "Final fused score: 0.539 (bullish)",
    "financial_analyst: strong positive (+0.300), sentiment=1.00, confidence=0.85",
    "market_intelligence: moderate positive (+0.100), sentiment=0.40, confidence=0.78",
    "qualitative_signal: weak positive (+0.069), sentiment=0.345, confidence=0.75",
    "workforce_intelligence: weak positive (+0.035), sentiment=0.23, confidence=0.70",
    "context_engine: weak positive (+0.035), sentiment=0.35, confidence=0.60"
]

# Collect top evidence
top_evidence = [
    Evidence(source="financial_data", description="Strong revenue growth of 15%", confidence=0.9),
    Evidence(source="financial_data", description="Healthy gross margin of 42%", confidence=0.85),
    Evidence(source="finnhub_ratings", description="32 analysts consensus BUY, target $195.50", confidence=0.78),
    Evidence(source="news_api", description="25 articles, avg sentiment 0.35, trending up", confidence=0.75),
    Evidence(source="employee_reviews", description="Employee rating 3.8/5.0, improving", confidence=0.70)
]

# Call BART for LLM summary
prompt = f"""Summarize the following investment evidence for AAPL in 2-3 concise sentences.
The overall signal is bullish with a score of 0.539.

Evidence:
- Strong revenue growth of 15%
- Healthy gross margin of 42%
- 32 analysts consensus BUY, target $195.50
- 25 articles, avg sentiment 0.35, trending up
- Employee rating 3.8/5.0, improving trend

Provide a professional summary focusing on the key factors driving the bullish outlook."""

llm_summary = hf_client.generate_text(
    prompt=prompt,
    model="facebook/bart-large-cnn",
    max_tokens=200,
    temperature=0.2
)
# Returns: "Apple demonstrates strong financial health with 15% revenue growth and
#          healthy 42% gross margin. Analyst consensus shows bullish sentiment with
#          32 analysts rating it a buy, targeting $195.50. Recent news coverage has
#          been predominantly positive, highlighting innovation and market leadership."

# Detect conflicts
conflicts = []  # No conflicts (all sentiments positive, difference < 1.0)

return {
    "fused_signal": FusedSignal(
        ticker="AAPL",
        final_score=0.539,
        confidence=0.798,
        agent_signals={...},
        signal_weights=weights,
        explanations=explanations,
        top_evidence=top_evidence,
        llm_summary=llm_summary
    ),
    "conflicts": [],
    "requires_debate": False,
    "workflow_log": [{"node": "signal_fusion", "status": "completed"}]
}
```

---

**Router: debate_router** (`core/investment_graph.py`)
```python
def debate_router(state: InvestmentAnalysisState) -> str:
    if state.get("requires_debate", False):  # False
        return "debate"
    else:
        return "recommendation"  # ← Takes this path
```

---

**Node 7: recommendation_node** (`core/investment_graph.py`)
```python
fused_signal = state["fused_signal"]
final_score = fused_signal.final_score  # 0.539

# Determine action
def _determine_action(fused_score: float) -> str:
    if fused_score >= 0.4:    return "BUY"      # ✓ (0.539 >= 0.4)
    elif fused_score >= 0.1:  return "ACCUMULATE"
    elif fused_score >= -0.1: return "HOLD"
    elif fused_score >= -0.4: return "REDUCE"
    else:                     return "SELL"

action = _determine_action(0.539)  # → "BUY"

# Build final recommendation
recommendation = {
    "ticker": "AAPL",
    "action": "BUY",
    "confidence": 0.798,
    "fused_score": 0.539,
    "reasoning": "All agents in agreement",
    "signal_contributions": {
        "financial_analyst": 0.300,
        "market_intelligence": 0.100,
        "qualitative_signal": 0.069,
        "workforce_intelligence": 0.035,
        "context_engine": 0.035
    },
    "supporting_evidence": [
        {
            "source": "financial_data",
            "description": "Strong revenue growth of 15%",
            "confidence": 0.9
        },
        {
            "source": "financial_data",
            "description": "Healthy gross margin of 42%",
            "confidence": 0.85
        },
        {
            "source": "finnhub_ratings",
            "description": "32 analysts consensus BUY, target $195.50",
            "confidence": 0.78
        },
        {
            "source": "news_api",
            "description": "25 articles, avg sentiment 0.35, trending up",
            "confidence": 0.75
        },
        {
            "source": "employee_reviews",
            "description": "Employee rating 3.8/5.0, improving trend",
            "confidence": 0.70
        }
    ],
    "llm_summary": "Apple demonstrates strong financial health with 15% revenue growth..."
}

return {
    "recommendation": recommendation,
    "workflow_log": [{"node": "recommendation", "status": "completed"}]
}
```

---

#### **Step 4: Orchestrator returns to dashboard** (`agents/strategic_orchestrator.py`)
```python
final_state = await graph.ainvoke(...)

return AgentResponse(
    agent_id="strategic_orchestrator",
    status="success",
    data={
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "recommendation": final_state["recommendation"],
        "agent_outputs": [output.dict() for output in final_state["agent_outputs"]],
        "conflicts_detected": False,
        "workflow_summary": [...],
        "langgraph_workflow_log": final_state["workflow_log"]
    },
    metadata={
        "agent_role": "strategic_orchestrator",
        "workflow_type": "LangGraph",
        "total_steps": 7,
        "num_agents": 5,
        "errors": 0
    },
    timestamp="2025-01-20T10:35:00Z"
)
```

---

#### **Step 5: Dashboard displays results** (`apps/dashboard.py`)
```python
recommendation = response.data["recommendation"]

# Display action and confidence
st.metric(
    label="Recommendation",
    value=recommendation["action"],
    delta=f"{recommendation['confidence']:.1%} confidence"
)
# Shows: BUY with 79.8% confidence

# Display signal contributions (Plotly bar chart)
fig = go.Figure(data=[
    go.Bar(
        x=list(recommendation["signal_contributions"].keys()),
        y=list(recommendation["signal_contributions"].values()),
        text=[f"{v:.1%}" for v in recommendation["signal_contributions"].values()],
        textposition="auto"
    )
])
st.plotly_chart(fig)

# Display LLM summary
st.info(recommendation["llm_summary"])
# Shows: "Apple demonstrates strong financial health..."

# Display supporting evidence
for evidence in recommendation["supporting_evidence"]:
    with st.expander(f"📋 {evidence['source']}"):
        st.write(f"**Description:** {evidence['description']}")
        st.write(f"**Confidence:** {evidence['confidence']:.1%}")

# Display workflow timeline
for log_entry in response.data["langgraph_workflow_log"]:
    st.write(f"✓ {log_entry['node']}: {log_entry['status']}")
```

---

## **Final Output Visualization**

```
┌─────────────────────────────────────────────────────────────┐
│  InvestmentIQ MVAS Analysis - Apple Inc. (AAPL)            │
├─────────────────────────────────────────────────────────────┤
│  Action: BUY                          Confidence: 79.8%     │
│  Fused Score: 0.539                   Sector: Technology    │
├─────────────────────────────────────────────────────────────┤
│  📊 Signal Contributions                                    │
│  ████████████████████████████████ Financial (30.0%)        │
│  ███████████████████ Market Intelligence (10.0%)           │
│  ██████████████ Qualitative Sentiment (6.9%)               │
│  █████████ Workforce Intelligence (3.5%)                   │
│  █████████ Context Engine (3.5%)                           │
├─────────────────────────────────────────────────────────────┤
│  💡 AI Summary (BART-generated):                           │
│  Apple demonstrates strong financial health with 15%       │
│  revenue growth and healthy 42% gross margin. Analyst      │
│  consensus shows bullish sentiment with 32 analysts        │
│  rating it a buy, targeting $195.50. Recent news           │
│  coverage has been predominantly positive, highlighting    │
│  innovation and market leadership.                         │
├─────────────────────────────────────────────────────────────┤
│  📋 Top Evidence:                                          │
│  ✓ Strong revenue growth of 15.0% (confidence: 90%)       │
│  ✓ Healthy gross margin of 42.0% (confidence: 85%)        │
│  ✓ 32 analysts consensus BUY, target $195.50 (conf: 78%)  │
│  ✓ 25 articles avg sentiment 0.35, trending up (75%)      │
│  ✓ Employee rating 3.8/5.0, improving trend (70%)         │
├─────────────────────────────────────────────────────────────┤
│  🔄 Workflow Timeline:                                     │
│  ✓ financial_analysis: completed                          │
│  ✓ qualitative_analysis: completed                        │
│  ✓ context_analysis: completed                            │
│  ✓ workforce_analysis: completed                          │
│  ✓ market_intelligence: completed                         │
│  ✓ signal_fusion: completed                               │
│  ✓ recommendation: completed                              │
│  Total execution time: 8.3 seconds                        │
└─────────────────────────────────────────────────────────────┘
```

---

## **Key Takeaways**

### **Architecture Highlights**
1. **Modular Design:** Each agent is independent and reusable
2. **Type Safety:** Pydantic models prevent runtime errors
3. **Declarative Workflows:** LangGraph eliminates manual orchestration
4. **Explainable AI:** SHAP-like contributions + LLM summaries
5. **Zero-Cost AI:** Free HuggingFace models (FinBERT + BART)
6. **Pub/Sub Communication:** Agents communicate via message bus
7. **Automatic State Management:** LangGraph handles state merging
8. **Conflict Resolution:** Built-in debate mechanism
9. **Comprehensive Observability:** LangSmith tracing integration
10. **Production-Ready:** Error handling, logging, testing

### **Data Flow Summary**
```
User Input (ticker)
  → Orchestrator
    → LangGraph State Init
      → 5 Agents (parallel conceptually, sequential in execution)
        → Signal Fusion (weighted ensemble)
          → Conflict Detection
            → [Debate if needed]
              → Final Recommendation
                → Dashboard Display
```

### **File Count**
- **Total Python Files:** 30
- **Core Framework Files:** 4 (contracts, bus, graph, fusion)
- **Agent Files:** 7 (5 specialists + base + orchestrator)
- **Tool Files:** 5 (edgar, news, finnhub, rag, data_tools)
- **Utility Files:** 5 (hf_client, llm_factory, observability, logger, ui)
- **App Files:** 1 (dashboard)
- **Config Files:** 1 (settings)
- **Evaluation Files:** 2 (evaluators, dataset_creator)
- **Test Files:** 1 (test_agents)

### **Lines of Code**
- **Total:** ~4,500 lines (excluding tests and backups)
- **Largest Files:**
  - `market_intelligence.py` (444 lines)
  - `signal_fusion.py` (442 lines)
  - `investment_graph.py` (518 lines)

---

---

## **Agent Data Templates & Requirements**

### **Purpose**
This section provides complete data templates for each of the 5 specialist agents. Use these templates to understand what data elements are required for each agent to perform its analysis.

---

### **1. Financial Analyst Agent** 💰

**File Location:** `data/samples/financial/{ticker}_financial.json`

**Data Template:**

```json
{
  "ticker": "AAPL",                        // REQUIRED: Stock ticker symbol
  "company_name": "Apple Inc.",            // Optional: Full company name
  "sector": "Technology",                  // Optional: Industry sector

  // CORE FINANCIAL METRICS (All values as decimals, not percentages)
  "revenue_growth": 0.15,                  // REQUIRED: Revenue growth rate (0.15 = 15%)
  "gross_margin": 0.42,                    // REQUIRED: Gross profit margin (0.42 = 42%)
  "operating_margin": 0.25,                // REQUIRED: Operating margin (0.25 = 25%)
  "net_margin": 0.20,                      // REQUIRED: Net profit margin (0.20 = 20%)

  // FINANCIAL HEALTH METRICS
  "debt_to_equity": 0.45,                  // REQUIRED: Debt-to-equity ratio
  "current_ratio": 1.8,                    // Optional: Current ratio (liquidity)
  "roe": 0.22,                            // REQUIRED: Return on equity (0.22 = 22%)
  "cash_flow_positive": true,              // Optional: Boolean for cash flow status

  // OPTIONAL DETAILED DATA
  "revenue": 1000000000,                   // Optional: Actual revenue in dollars
  "gross_profit": 420000000,               // Optional: Gross profit in dollars
  "operating_income": 250000000,           // Optional: Operating income in dollars
  "net_income": 200000000,                 // Optional: Net income in dollars
  "total_debt": 450000000,                 // Optional: Total debt in dollars
  "total_equity": 1000000000,              // Optional: Total equity in dollars
  "total_assets": 6500000000,              // Optional: Total assets
  "cash_and_equivalents": 1800000000,      // Optional: Cash position
  "research_and_development": 580000000,   // Optional: R&D spending
  "earnings_per_share": 4.85,              // Optional: EPS
  "price_to_earnings": 22.5,               // Optional: P/E ratio
  "quick_ratio": 1.8,                      // Optional: Quick ratio
  "reporting_quality": "high",             // Optional: Reporting quality assessment
  "audit_opinion": "unqualified",          // Optional: Audit opinion
  "notes": "Strong financials..."          // Optional: Additional notes
}
```

**Sentiment Calculation Logic:**
- `revenue_growth > 0.15`: +0.3 (or +0.1 if > 0)
- `gross_margin > 0.4`: +0.2
- `operating_margin > 0.2`: +0.2
- `debt_to_equity < 0.5`: +0.2
- `roe > 0.15`: +0.1
- **Max sentiment:** 1.0 (capped)

**Alerts Triggered:**
- `debt_to_equity > 1.5`: High severity alert

**Implementation:** See `agents/financial_analyst.py:124-143` for data loading

---

### **2. Qualitative Signal Agent** 📰

**File Location:** `data/samples/qualitative/{ticker}_qualitative.json`

**Data Template:**

```json
{
  "ticker": "AAPL",                        // REQUIRED: Stock ticker symbol
  "company_name": "Apple Inc.",            // Optional: Full company name

  // SENTIMENT SCORES
  "overall_sentiment": "Positive",         // REQUIRED: Text label (Positive/Negative/Neutral)
  "sentiment_score": 0.3,                  // REQUIRED: Numeric sentiment [-1, 1]
  "reputation_score": 0.75,                // REQUIRED: Reputation score [0, 1]

  // NEWS METRICS
  "news_count": 45,                        // REQUIRED: Number of news articles analyzed
  "sentiment_breakdown": {                 // REQUIRED: Sentiment distribution
    "positive": 0.55,                      // % of positive articles (0.55 = 55%)
    "neutral": 0.25,                       // % of neutral articles
    "negative": 0.20                       // % of negative articles
  },

  // MARKET PERCEPTION
  "market_perception": "positive",         // REQUIRED: positive/negative/neutral

  // THEMATIC ANALYSIS
  "key_themes": [                          // REQUIRED: List of key themes (can be empty)
    "innovation",
    "growth",
    "ai",
    "leadership"
  ],

  // OPTIONAL DETAILED NEWS DATA
  "recent_news": [                         // Optional: Individual news items
    {
      "title": "Apple announces record iPhone sales",
      "date": "2025-01-20",
      "source": "Reuters",
      "sentiment": 0.8,
      "url": "https://..."
    }
  ],
  "social_sentiment": 0.4,                 // Optional: Social media sentiment
  "brand_strength": 0.85                   // Optional: Brand strength metric
}
```

**Sentiment Calculation Logic:**
```python
sentiment = (
    0.4 * sentiment_score +                          # Base sentiment
    0.3 * (positive_ratio - negative_ratio) +        # Breakdown
    0.2 * ((reputation_score - 0.5) * 2) +          # Reputation
    0.1 * perception_boost                           # Market perception (+0.2/-0.2/0)
)
```

**Confidence Calculation:**
- `news_count > 50`: 0.85
- `news_count > 20`: 0.75
- `news_count > 5`: 0.65
- `news_count <= 5`: 0.50

**Alerts Triggered:**
- `sentiment_score < -0.5`: High severity - Negative Market Sentiment
- `reputation_score < 0.3`: High severity - Reputation Risk
- Risk themes detected (scandal, crisis, lawsuit): Medium severity

**Implementation:** See `agents/qualitative_signal.py:126-149` for data loading

---

### **3. Context Engine Agent** 🔍

**File Location:** `data/samples/context/{ticker}_context.json`

**Data Template:**

```json
{
  "ticker": "AAPL",                        // REQUIRED: Stock ticker symbol

  // SCENARIO CLASSIFICATION
  "scenario_type": "strong_momentum",      // REQUIRED: Scenario type
                                          // Options: "strong_momentum", "contrarian_opportunity",
                                          //          "risk_warning", "normal", "uncertain"

  // PATTERN MATCHING
  "pattern_matches": 8,                    // REQUIRED: Number of similar historical patterns
  "historical_accuracy": 0.75,             // REQUIRED: Accuracy rate of patterns [0, 1]
  "pattern_confidence": 0.7,               // REQUIRED: Confidence in pattern match [0, 1]

  // MARKET CONTEXT
  "market_cycle": "expansion",             // REQUIRED: expansion/peak/contraction/trough
  "sector_trend": "bullish",               // REQUIRED: bullish/stable/bearish

  // CONTRARIAN SIGNALS
  "contrarian_signal": false,              // REQUIRED: Boolean for contrarian opportunity

  // HISTORICAL COMPARISONS
  "similar_situations": [                  // REQUIRED: List of similar past scenarios (can be empty)
    {
      "ticker": "MSFT",
      "year": 2018,
      "outcome": "positive",
      "similarity_score": 0.85
    },
    {
      "ticker": "GOOGL",
      "year": 2020,
      "outcome": "positive",
      "similarity_score": 0.78
    }
  ],

  // OPTIONAL ADDITIONAL CONTEXT
  "market_conditions": {                   // Optional: Broader market conditions
    "volatility": "low",
    "momentum": "bullish",
    "risk_appetite": "high"
  },
  "sector_rotation": "into",               // Optional: Sector rotation status
  "technical_indicators": {                // Optional: Technical signals
    "rsi": 62,
    "macd": "bullish"
  }
}
```

**Sentiment Calculation Logic:**
```python
# Scenario sentiment mapping
scenario_sentiment = {
    "contrarian_opportunity": 0.6,
    "strong_momentum": 0.7,
    "risk_warning": -0.6,
    "normal": 0.0,
    "uncertain": 0.0
}

# Market cycle boost
cycle_boost = {"expansion": 0.2, "peak": 0.0, "contraction": -0.2, "trough": -0.1}

# Sector trend boost
trend_boost = {"bullish": 0.15, "stable": 0.0, "bearish": -0.15}

# Contrarian adjustment
contrarian_adj = 0.3 if contrarian_signal else 0.0

sentiment = (
    0.4 * scenario_sentiment +
    0.2 * cycle_boost +
    0.2 * trend_boost +
    0.2 * contrarian_adj
)
```

**Confidence Calculation:**
- Based on `pattern_matches`:
  - `> 10`: 0.9
  - `> 5`: 0.8
  - `> 2`: 0.7
  - `<= 2`: 0.6
- Adjusted by `historical_accuracy`

**Alerts Triggered:**
- `contrarian_signal = true`: Medium severity - Contrarian Opportunity
- `scenario_type = "risk_warning"`: High severity - Historical Risk Pattern
- `pattern_confidence < 0.5`: Low severity - Low Pattern Confidence

**Implementation:** See `agents/context_engine.py:126-145` for data loading

---

### **4. Workforce Intelligence Agent** 👥

**File Location:** `data/samples/workforce/{ticker}_workforce.json`

**Data Template:**

```json
{
  "company": "AAPL",                       // REQUIRED: Ticker
  "company_name": "Apple Inc.",            // REQUIRED: Full company name

  // EMPLOYEE RATINGS
  "rating": 3.8,                          // REQUIRED: Employee rating (1-5 scale)
  "rating_trend": "increasing",            // REQUIRED: increasing/declining/stable
  "review_count": 1250,                    // REQUIRED: Number of employee reviews

  // RATING HISTORY
  "rating_history": [                      // REQUIRED: Historical ratings (can be 1 entry)
    {
      "date": "2025-01-01",
      "rating": 3.7
    },
    {
      "date": "2024-10-01",
      "rating": 3.6
    },
    {
      "date": "2024-07-01",
      "rating": 3.5
    }
  ],

  // TOPIC ANALYSIS
  "topics": {                              // REQUIRED: Topic sentiment distribution [0, 1]
    "work_life_balance": 0.35,             // Sentiment score for work-life balance
    "compensation": 0.40,                  // Sentiment for compensation
    "culture": 0.45,                       // Sentiment for company culture
    "management": 0.30,                    // Sentiment for management
    "career": 0.38                         // Sentiment for career growth
  },

  // HIRING SIGNALS
  "hiring_signal": "aggressive",           // REQUIRED: aggressive/moderate/slow/freeze/unknown
  "job_postings_count": 150,               // REQUIRED: Number of active job postings
  "job_postings_trend": "increasing",      // REQUIRED: increasing/decreasing/stable
  "job_postings_history": [                // Optional: Historical job posting data
    {
      "date": "2025-01-01",
      "count": 150
    },
    {
      "date": "2024-12-01",
      "count": 135
    }
  ],

  // RETENTION METRICS
  "tenure_metrics": {                      // REQUIRED: Employee retention data
    "avg_tenure_years": 4.2,               // Average tenure in years
    "churn_rate": 0.15,                    // Churn rate (0.15 = 15%)
    "trend": "stable"                      // increasing/stable/decreasing churn
  },

  // SENTIMENT BREAKDOWN
  "sentiment_breakdown": {                 // REQUIRED: Overall sentiment distribution
    "positive": 0.45,                      // % of positive reviews
    "neutral": 0.35,                       // % of neutral reviews
    "negative": 0.20                       // % of negative reviews
  },

  "timestamp": "2025-01-20T10:00:00Z"     // REQUIRED: Data timestamp (ISO 8601)
}
```

**Sentiment Calculation Logic:**
```python
# Normalize rating (1-5 scale) to sentiment
rating_sentiment = (rating - 3.0) / 2.0  # 5→1.0, 3→0, 1→-1.0

# Sentiment breakdown
breakdown_sentiment = positive_ratio - negative_ratio

# Trend boosts
trend_boost = 0.0
if rating_trend == "increasing": trend_boost += 0.1
elif rating_trend == "declining": trend_boost -= 0.1

if hiring_signal == "aggressive": trend_boost += 0.05
elif hiring_signal == "freeze": trend_boost -= 0.05

sentiment = (
    0.5 * rating_sentiment +
    0.3 * breakdown_sentiment +
    0.2 * trend_boost
)
```

**Confidence Calculation:**
- Based on `review_count`:
  - `> 10000`: 0.9
  - `> 5000`: 0.8
  - `> 1000`: 0.7
  - `> 100`: 0.6
  - `<= 100`: 0.4
- Reduced by 15% if rating volatility > 0.5

**Alerts Triggered:**
- `rating < 3.0`: High/Medium severity - Low Employee Satisfaction
- `hiring_signal = "freeze"`: Medium severity - Hiring Freeze Detected
- `churn_rate > 0.20`: High/Medium severity - Elevated Employee Churn
- `rating_trend = "declining"`: Medium severity - Declining Employee Sentiment

**Implementation:** See `agents/workforce_intelligence.py:128-188` for data loading

---

### **5. Market Intelligence Agent** 📊

**Note:** This agent uses MULTIPLE data sources - it doesn't have a single JSON file. Instead, it aggregates data from 3 tools.

#### **5a. SEC EDGAR Tool Data**

**Required by:** `tools/edgar_tool.py`

```json
{
  "ticker": "AAPL",
  "filings": [
    {
      "type": "8-K",                       // REQUIRED: Filing type
      "date": "2025-01-15",                // REQUIRED: Filing date
      "accession_number": "0001234567890", // Optional: SEC accession number
      "description": "Item 2.02 - Results of Operations and Financial Condition",
      "url": "https://www.sec.gov/...",
      "items": ["2.02"],                   // Optional: Specific 8-K items
      "event_type": "financial_results"    // Optional: Categorized event type
    }
  ]
}
```

**Event Categorization:**
- `financial_results`: Earnings, financial results
- `leadership_changes`: Executive changes, board changes
- `strategic_actions`: M&A, partnerships, major contracts

#### **5b. News API Tool Data**

**Required by:** `tools/news_api_tool.py`

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "articles": [
    {
      "title": "Apple announces record iPhone sales",  // REQUIRED
      "description": "Apple Inc. reported...",         // REQUIRED
      "source": {"name": "Reuters"},                   // REQUIRED
      "publishedAt": "2025-01-20T10:00:00Z",          // REQUIRED (ISO 8601)
      "url": "https://...",                            // REQUIRED
      "content": "Full article text...",               // Optional
      "sentiment": {                                   // Added by FinBERT analysis
        "label": "positive",                           // positive/negative/neutral
        "score": 0.89                                  // Confidence [0, 1]
      }
    }
  ],
  "totalResults": 25,                                  // REQUIRED: Total articles count
  "sentiment_metrics": {                               // Calculated by agent
    "avg_sentiment": 0.35,
    "sentiment_trend": "improving",
    "positive_ratio": 0.60,
    "negative_ratio": 0.15,
    "article_count": 25
  }
}
```

#### **5c. Finnhub Tool Data**

**Required by:** `tools/finnhub_tool.py`

**Analyst Ratings:**
```json
{
  "ticker": "AAPL",
  "consensus": {
    "avg_target": 195.50,                  // REQUIRED: Average price target
    "buy": 25,                             // REQUIRED: Number of buy ratings
    "hold": 5,                             // REQUIRED: Number of hold ratings
    "sell": 2,                             // REQUIRED: Number of sell ratings
    "strong_buy": 10,                      // Optional: Strong buy count
    "strong_sell": 0                       // Optional: Strong sell count
  },
  "recent_changes": [                      // Optional: Recent rating changes
    {
      "date": "2025-01-15",
      "analyst": "Goldman Sachs",
      "from_rating": "hold",
      "to_rating": "buy",
      "price_target": 200.00
    }
  ]
}
```

**Executive Changes:**
```json
{
  "ticker": "AAPL",
  "recent_changes": [                      // REQUIRED: List of changes (can be empty)
    {
      "date": "2025-01-10",
      "name": "John Doe",
      "position": "CFO",
      "change_type": "departure",          // departure/appointment
      "reason": "retirement"               // Optional
    }
  ],
  "stability_score": 0.85,                 // REQUIRED: Stability score [0, 1]
  "leadership_count": 12                   // Optional: Total leadership count
}
```

#### **Market Intelligence Aggregation**

The Market Intelligence Agent **combines all three sources** with the following weights:

```python
weights = {
    "analyst": 0.35,      # Analyst ratings (highest weight)
    "news": 0.30,         # News sentiment
    "filing": 0.25,       # SEC filings
    "exec": 0.10          # Executive stability
}
```

**Final Output Metrics:**
```json
{
  "sec_filings": {
    "total_filings": 3,
    "total_events": 5,
    "financial_events": 2,
    "leadership_events": 1,
    "strategic_events": 2
  },
  "news_sentiment": {
    "total_articles": 25,
    "avg_sentiment": 0.35,
    "sentiment_trend": "improving",
    "positive_ratio": 0.60,
    "negative_ratio": 0.15
  },
  "analyst_ratings": {
    "consensus_sentiment": 0.60,
    "avg_price_target": 195.50,
    "total_analysts": 32,
    "bullish_ratio": 0.78,
    "bearish_ratio": 0.06,
    "upgrades": 2,
    "downgrades": 0,
    "momentum": "positive"
  },
  "executive_stability": {
    "stability_score": 0.85,
    "recent_changes": 0,
    "departures": 0,
    "leadership_count": 12
  }
}
```

**Implementation:** See `agents/market_intelligence.py:47-145` for orchestration

---

### **Agent Data Requirements Summary**

| Agent | Required Fields | Data Format | Sentiment Range | Confidence Range |
|-------|----------------|-------------|-----------------|------------------|
| **Financial Analyst** | revenue_growth, gross_margin, operating_margin, net_margin, debt_to_equity, roe | JSON | [-1, 1] | [0.85] (fixed) |
| **Qualitative Signal** | sentiment_score, news_count, sentiment_breakdown, reputation_score, market_perception, key_themes | JSON | [-1, 1] | [0.5 - 0.85] |
| **Context Engine** | scenario_type, pattern_matches, historical_accuracy, market_cycle, sector_trend, contrarian_signal | JSON | [-1, 1] | [0.6 - 0.9] |
| **Workforce Intelligence** | rating, rating_trend, review_count, topics, hiring_signal, job_postings_count, tenure_metrics, sentiment_breakdown | JSON | [-1, 1] | [0.4 - 0.9] |
| **Market Intelligence** | Multiple sources (Edgar, NewsAPI, Finnhub) | Multiple JSONs | [-1, 1] | [0.2 - 0.9] |

---

### **Quick Reference: Creating Sample Data**

To add sample data for a new ticker (e.g., "TSLA"):

1. **Financial Data:** Create `data/samples/financial/tsla_financial.json`
2. **Qualitative Data:** Create `data/samples/qualitative/tsla_qualitative.json`
3. **Context Data:** Create `data/samples/context/tsla_context.json`
4. **Workforce Data:** Create `data/samples/workforce/tsla_workforce.json`
5. **Market Intelligence:** Uses live tools (edgar_tool, news_api_tool, finnhub_tool)

The system will automatically load and use these files when analyzing the ticker in **sample mode**.

---

## **REQUIRED Quantitative Financial Data Elements for Stock Analysis**

### **Purpose**
This section identifies all REQUIRED quantitative data elements needed for comprehensive stock analysis across all 5 agents. Use this as a reference when integrating with external data sources.

---

### **Complete Data Requirements Table**

| # | Data Element | Example Value | Data Type | Agent Name | Purpose / Use Case | Calculation Impact |
|---|--------------|---------------|-----------|------------|-------------------|-------------------|
| **CORE FINANCIAL METRICS** |
| 1 | `revenue_growth` | 0.15 (15%) | Decimal (ratio) | Financial Analyst | Measures year-over-year revenue growth rate | +0.3 sentiment if >15%, +0.1 if >0% |
| 2 | `gross_margin` | 0.42 (42%) | Decimal (ratio) | Financial Analyst | Profitability after COGS (Cost of Goods Sold) | +0.2 sentiment if >40% |
| 3 | `operating_margin` | 0.25 (25%) | Decimal (ratio) | Financial Analyst | Profitability after operating expenses | +0.2 sentiment if >20% |
| 4 | `net_margin` | 0.20 (20%) | Decimal (ratio) | Financial Analyst | Bottom-line profitability after all expenses | Used in metrics extraction |
| 5 | `debt_to_equity` | 0.45 | Decimal (ratio) | Financial Analyst | Financial leverage and solvency risk | +0.2 sentiment if <0.5, ALERT if >1.5 |
| 6 | `roe` | 0.22 (22%) | Decimal (ratio) | Financial Analyst | Return on Equity - profitability efficiency | +0.1 sentiment if >15% |
| **MARKET INTELLIGENCE METRICS** |
| 7 | `avg_price_target` | 195.50 | Dollar amount | Market Intelligence | Analyst consensus price target | Used for analyst sentiment calculation |
| 8 | `buy_ratings_count` | 25 | Integer (count) | Market Intelligence | Number of analyst "buy" ratings | Used to calculate bullish_ratio |
| 9 | `hold_ratings_count` | 5 | Integer (count) | Market Intelligence | Number of analyst "hold" ratings | Used for consensus calculation |
| 10 | `sell_ratings_count` | 2 | Integer (count) | Market Intelligence | Number of analyst "sell" ratings | Used to calculate bearish_ratio |
| **QUALITATIVE METRICS** |
| 11 | `sentiment_score` | 0.3 | Decimal [-1, 1] | Qualitative Signal | Overall news/social sentiment score | 40% weight in sentiment calculation |
| 12 | `reputation_score` | 0.75 | Decimal [0, 1] | Qualitative Signal | Brand/company reputation metric | 20% weight in sentiment calculation |
| 13 | `news_count` | 45 | Integer (count) | Qualitative Signal | Number of news articles analyzed | Determines confidence level (>50=0.85, >20=0.75) |
| 14 | `positive_ratio` | 0.55 (55%) | Decimal (ratio) | Qualitative Signal | Percentage of positive news articles | Used in sentiment breakdown calculation |
| 15 | `negative_ratio` | 0.20 (20%) | Decimal (ratio) | Qualitative Signal | Percentage of negative news articles | Used in sentiment breakdown calculation |
| **WORKFORCE METRICS** |
| 16 | `employee_rating` | 3.8 | Decimal [1-5] | Workforce Intelligence | Employee satisfaction rating (Glassdoor-style) | Normalized to [-1,1]: (rating-3)/2 |
| 17 | `review_count` | 1250 | Integer (count) | Workforce Intelligence | Number of employee reviews | Confidence: >10K=0.9, >5K=0.8, >1K=0.7 |
| 18 | `job_postings_count` | 150 | Integer (count) | Workforce Intelligence | Number of active job postings | Used for hiring signal assessment |
| 19 | `churn_rate` | 0.15 (15%) | Decimal (ratio) | Workforce Intelligence | Employee turnover rate | ALERT if >20%, +sentiment if aggressive hiring |
| 20 | `avg_tenure_years` | 4.2 | Decimal (years) | Workforce Intelligence | Average employee tenure | Indicator of stability |
| **CONTEXT METRICS** |
| 21 | `pattern_matches` | 8 | Integer (count) | Context Engine | Number of similar historical patterns found | Confidence: >10=0.9, >5=0.8, >2=0.7 |
| 22 | `historical_accuracy` | 0.75 (75%) | Decimal [0, 1] | Context Engine | Accuracy rate of historical pattern predictions | Adjusts confidence score |
| 23 | `pattern_confidence` | 0.7 (70%) | Decimal [0, 1] | Context Engine | Confidence in pattern match quality | ALERT if <0.5 (low confidence) |
| **SEC FILINGS METRICS** |
| 24 | `total_filings` | 3 | Integer (count) | Market Intelligence | Number of recent SEC filings (8-K) | Quality metric for filing analysis |
| 25 | `financial_events_count` | 2 | Integer (count) | Market Intelligence | Material financial events from 8-K | Categorized event sentiment |
| 26 | `leadership_events_count` | 1 | Integer (count) | Market Intelligence | Leadership changes from 8-K | Executive stability assessment |
| 27 | `strategic_events_count` | 2 | Integer (count) | Market Intelligence | Strategic actions (M&A, partnerships) | ALERT for material strategic events |
| **EXECUTIVE METRICS** |
| 28 | `stability_score` | 0.85 | Decimal [0, 1] | Market Intelligence | Leadership team stability score | 10% weight in market intelligence |
| 29 | `recent_departures` | 0 | Integer (count) | Market Intelligence | Number of recent executive departures | ALERT if >0, sentiment impact |

---

### **Data Element Categories Summary**

| Category | Count | Primary Agent | Data Format | Impact Level |
|----------|-------|---------------|-------------|--------------|
| **Financial Health** | 6 | Financial Analyst | Ratios (decimals) | **CRITICAL** - 30% final weight |
| **Market Intelligence** | 9 | Market Intelligence | Mixed (counts, dollars, ratios) | **HIGH** - 25% final weight |
| **Sentiment/Reputation** | 5 | Qualitative Signal | Scores & counts | **HIGH** - 20% final weight |
| **Workforce Signals** | 5 | Workforce Intelligence | Ratings & counts | **MEDIUM** - 15% final weight |
| **Historical Context** | 3 | Context Engine | Counts & probabilities | **MEDIUM** - 10% final weight |
| **SEC/Leadership** | 5 | Market Intelligence | Counts & scores | **SUPPORTING** - within Market Intelligence |

---

### **Critical Thresholds by Data Element**

| Data Element | Positive Threshold | Negative Threshold | Alert Threshold |
|--------------|-------------------|-------------------|-----------------|
| `revenue_growth` | >15% (+0.3) | <0% (0.0) | N/A |
| `gross_margin` | >40% (+0.2) | <35% (lower evidence) | N/A |
| `operating_margin` | >20% (+0.2) | <10% (concern) | N/A |
| `debt_to_equity` | <0.5 (+0.2) | >1.0 (concern) | **>1.5 (HIGH)** |
| `roe` | >15% (+0.1) | <10% (weak) | N/A |
| `sentiment_score` | >0.3 (positive) | <-0.3 (negative) | **<-0.5 (HIGH)** |
| `reputation_score` | >0.7 (strong) | <0.4 (weak) | **<0.3 (HIGH)** |
| `employee_rating` | >4.0 (excellent) | <3.5 (below avg) | **<3.0 (MEDIUM/HIGH)** |
| `churn_rate` | <10% (stable) | >15% (concerning) | **>20% (MEDIUM/HIGH)** |
| `pattern_confidence` | >0.7 (reliable) | <0.6 (uncertain) | **<0.5 (LOW)** |

---

### **Minimum Required Data for System to Function**

#### **Absolute Minimum (Financial Analyst Only)**
```json
{
  "revenue_growth": 0.15,
  "gross_margin": 0.42,
  "operating_margin": 0.25,
  "net_margin": 0.20,
  "debt_to_equity": 0.45,
  "roe": 0.22
}
```
**Result:** System can run with only Financial Analyst (other agents use defaults)

#### **Recommended Minimum (3 Core Agents)**
Add to above:
```json
{
  "sentiment_score": 0.3,
  "reputation_score": 0.75,
  "news_count": 45,
  "positive_ratio": 0.55,
  "negative_ratio": 0.20,
  "pattern_matches": 8,
  "historical_accuracy": 0.75
}
```
**Result:** Financial + Qualitative + Context agents active

#### **Full System (All 5 Agents)**
Add to above:
```json
{
  "employee_rating": 3.8,
  "review_count": 1250,
  "churn_rate": 0.15,
  "avg_price_target": 195.50,
  "buy_ratings_count": 25,
  "hold_ratings_count": 5,
  "sell_ratings_count": 2
}
```
**Result:** All 5 agents active with full analysis

---

### **Data Quality Notes**
1. **All decimal values** should be expressed as ratios (0.15 = 15%, not 15)
2. **Default values** are used when data is missing (see agent implementation for defaults)
3. **Confidence scores** are automatically calculated based on data availability
4. **Market Intelligence** is unique - it aggregates from 3 separate data sources (Edgar, NewsAPI, Finnhub)
5. **Sentiment ranges** are normalized to [-1, 1] for all agents for consistency

---

## **Recommended Data Sources: Option 1 - FMP + Finnhub**

### **Purpose**
This section documents the recommended production data sources for replacing sample/mock data with real financial data APIs.

---

### **🏆 Recommended Stack**

```
┌─────────────────────────────────────────────────────┐
│ Financial Modeling Prep (FMP) - Primary Source     │
│ ✅ All 6 core financial metrics                    │
│ ✅ Analyst ratings & price targets                  │
│ ✅ Company news & sentiment                         │
│ ✅ SEC filings (8-K)                                │
│ ✅ Earnings data                                    │
│                                                     │
│ 💰 FREE: 250 calls/day                             │
│ 💰 PAID: $14-29/month                              │
│ 🌐 https://financialmodelingprep.com               │
└─────────────────────────────────────────────────────┘
              +
┌─────────────────────────────────────────────────────┐
│ Finnhub - Secondary Source                          │
│ ✅ Enhanced news sentiment                          │
│ ✅ Executive changes                                │
│ ✅ Social sentiment signals                         │
│ ✅ Alternative data                                 │
│                                                     │
│ 💰 FREE: 60 calls/minute                           │
│ 💰 PAID: $0-399/month                              │
│ 🌐 https://finnhub.io                              │
└─────────────────────────────────────────────────────┘
              +
┌─────────────────────────────────────────────────────┐
│ Workforce Data - Manual/Alternative Approach        │
│ ⚠️ Employee ratings (Glassdoor API restricted)     │
│ ⚠️ Company reviews                                  │
│ ⚠️ Job postings                                     │
│                                                     │
│ 💡 ALTERNATIVE: Manual curated dataset or          │
│    public data aggregation                          │
└─────────────────────────────────────────────────────┘
```

---

### **Coverage Analysis**

| Agent | Data Source | Coverage | Notes |
|-------|------------|----------|-------|
| **Financial Analyst** | FMP | ✅ 100% | All 6 metrics available via `/ratios/{ticker}` |
| **Market Intelligence** | FMP + Finnhub | ✅ 95% | Analyst ratings, news, SEC filings, executive changes |
| **Qualitative Signal** | FMP + Finnhub | ✅ 90% | News sentiment, social signals, reputation proxies |
| **Context Engine** | FMP | ✅ 80% | Historical data for pattern matching via `/historical-price-full/` |
| **Workforce Intelligence** | Manual/Alternative | ⚠️ 40% | Limited API access; requires alternative solution |

**Overall System Coverage:** **85% with real-time data**

---

### **Cost Analysis**

#### **Free Tier (Testing & Development)**
| Service | Free Tier | Limits | Cost |
|---------|-----------|--------|------|
| FMP | ✅ Available | 250 calls/day | **$0** |
| Finnhub | ✅ Available | 60 calls/minute | **$0** |
| **TOTAL** | | **~750 calls/day** | **$0** |

**Sufficient for:** Development, testing, small portfolios (<50 tickers)

#### **Production Tier (Recommended)**
| Service | Plan | Limits | Cost/Month |
|---------|------|--------|------------|
| FMP | Starter | 1,000 calls/day | **$14** |
| FMP | Professional | 10,000 calls/day | **$29** |
| Finnhub | Free | 60 calls/minute | **$0** |
| **TOTAL (Recommended)** | | | **$14-29** |

**Sufficient for:** Production systems, portfolios up to 500 tickers

#### **Enterprise Tier (High Volume)**
| Service | Plan | Limits | Cost/Month |
|---------|------|--------|------------|
| FMP | Professional | 10,000 calls/day | $29 |
| Finnhub | Starter | 300 calls/minute | $59 |
| **TOTAL** | | | **$88** |

---

### **API Endpoint Mapping**

#### **Financial Analyst Agent → FMP Endpoints**

```python
# Single endpoint provides all 6 required metrics
GET https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={key}

Response includes:
{
  "revenueGrowth": 0.15,              # → revenue_growth
  "grossProfitMargin": 0.42,          # → gross_margin
  "operatingProfitMargin": 0.25,      # → operating_margin
  "netProfitMargin": 0.20,            # → net_margin
  "debtEquityRatio": 0.45,            # → debt_to_equity
  "returnOnEquity": 0.22              # → roe
}
```

**Agent Coverage:** ✅ 100% (all 6 metrics)
**API Calls Required:** 1 per ticker

---

#### **Market Intelligence Agent → FMP + Finnhub Endpoints**

```python
# FMP: Analyst Ratings
GET https://financialmodelingprep.com/api/v3/grade/{ticker}?apikey={key}

Response includes:
{
  "symbol": "AAPL",
  "gradingCompany": "Goldman Sachs",
  "newGrade": "Buy",
  "previousGrade": "Hold",
  "date": "2025-01-15"
}

# FMP: Price Target Consensus
GET https://financialmodelingprep.com/api/v3/price-target-consensus/{ticker}?apikey={key}

Response includes:
{
  "targetHigh": 210.00,
  "targetLow": 180.00,
  "targetConsensus": 195.50,          # → avg_price_target
  "targetMedian": 195.00
}

# FMP: News with Sentiment
GET https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=50&apikey={key}

# FMP: SEC Filings
GET https://financialmodelingprep.com/api/v3/sec_filings/{ticker}?type=8-K&apikey={key}

# Finnhub: Executive Changes
GET https://finnhub.io/api/v1/executive-changes?symbol={ticker}&token={key}
```

**Agent Coverage:** ✅ 95% (all except some workforce-related metrics)
**API Calls Required:** 5 per ticker (can be optimized with caching)

---

#### **Qualitative Signal Agent → FMP + Finnhub Endpoints**

```python
# FMP: Company News
GET https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit=50&apikey={key}

# Process with existing FinBERT for sentiment analysis
# Calculate: sentiment_score, positive_ratio, negative_ratio, news_count

# Finnhub: Social Sentiment (optional enhancement)
GET https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={key}

Response includes:
{
  "buzz": {
    "articlesInLastWeek": 45,         # → news_count
    "weeklyAverage": 40
  },
  "sentiment": {
    "bullishPercent": 0.55,           # → positive_ratio
    "bearishPercent": 0.20            # → negative_ratio
  }
}
```

**Agent Coverage:** ✅ 90% (sentiment, news count, reputation proxies)
**API Calls Required:** 2 per ticker

---

#### **Context Engine Agent → FMP Endpoints**

```python
# FMP: Historical Price Data
GET https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={date}&to={date}&apikey={key}

# FMP: Historical Ratios (for pattern matching)
GET https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=annual&limit=10&apikey={key}

# Use historical data to identify patterns
# Calculate: pattern_matches, historical_accuracy, pattern_confidence
```

**Agent Coverage:** ✅ 80% (can build pattern matching on historical data)
**API Calls Required:** 2 per ticker (1 for prices, 1 for ratios)

---

#### **Workforce Intelligence Agent → Alternative Approach**

```python
# No direct API available due to Glassdoor API restrictions

# Option A: Manual Curated Dataset
# Create quarterly-updated dataset for top 500 companies
# Source: Public Glassdoor ratings, LinkedIn job postings

# Option B: Proxy Metrics (from available APIs)
# - Job postings count → FMP or Indeed API
# - LinkedIn follower growth → LinkedIn API (if available)
# - News mentions of "hiring" → FMP news API
# - Employee satisfaction → manual research

# Option C: Third-party Aggregators
# - Built In API (limited)
# - Comparably API (limited)
```

**Agent Coverage:** ⚠️ 40% (requires manual or alternative data)
**Recommendation:** Start with manual curated data for top companies, expand later

---

### **Implementation Priority**

#### **Phase 1: Core Financial Data (Week 1)**
**Priority: CRITICAL**

- Implement FMP integration for Financial Analyst Agent
- Replace sample data with real API calls
- Test with 10-20 tickers

**Files to modify:**
- `tools/fmp_tool.py` (new)
- `agents/financial_analyst.py`

**Estimated effort:** 2-4 hours

---

#### **Phase 2: Market Intelligence (Week 1-2)**
**Priority: HIGH**

- Implement FMP + Finnhub for Market Intelligence Agent
- Add analyst ratings, price targets, SEC filings
- Enhance news sentiment analysis

**Files to modify:**
- `tools/fmp_tool.py` (enhance)
- `tools/finnhub_enhanced_tool.py` (new)
- `agents/market_intelligence.py`

**Estimated effort:** 4-6 hours

---

#### **Phase 3: Qualitative Enhancement (Week 2)**
**Priority: HIGH**

- Integrate FMP news API with existing FinBERT
- Add Finnhub social sentiment
- Calculate sentiment metrics

**Files to modify:**
- `tools/fmp_tool.py` (enhance)
- `agents/qualitative_signal.py`

**Estimated effort:** 3-4 hours

---

#### **Phase 4: Historical Context (Week 2-3)**
**Priority: MEDIUM**

- Use FMP historical data for pattern matching
- Build pattern recognition logic
- Calculate accuracy metrics

**Files to modify:**
- `tools/fmp_tool.py` (enhance)
- `agents/context_engine.py`

**Estimated effort:** 6-8 hours

---

#### **Phase 5: Workforce Data (Week 3-4)**
**Priority: LOW**

- Create manual curated dataset for top 500 companies
- Build update mechanism
- Add proxy metrics where possible

**Files to create:**
- `data/workforce/curated_workforce_data.json`
- `tools/workforce_data_updater.py`

**Estimated effort:** 8-12 hours

---

### **Environment Setup**

#### **Required Environment Variables**

```bash
# .env file
# Financial Modeling Prep
FMP_API_KEY=your_fmp_api_key_here

# Finnhub
FINNHUB_API_KEY=your_finnhub_api_key_here

# Data mode (sample vs live)
LIVE_CONNECTORS=true

# Optional: Rate limiting
API_RATE_LIMIT_PER_MINUTE=60
```

#### **Getting API Keys**

**Financial Modeling Prep:**
1. Visit https://financialmodelingprep.com/developer/docs/pricing
2. Sign up for free tier (250 calls/day) or paid plan
3. Copy API key from dashboard
4. Add to `.env` file

**Finnhub:**
1. Visit https://finnhub.io/register
2. Sign up for free account
3. Copy API key from dashboard
4. Add to `.env` file

---

### **Sample Integration Code**

```python
# tools/fmp_tool.py
import os
import requests
from typing import Dict, Any, Optional

class FMPTool:
    """Financial Modeling Prep API integration."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")

    def get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """
        Get all 6 required financial metrics in ONE API call.

        Maps FMP response to InvestmentIQ data format.
        """
        url = f"{self.BASE_URL}/ratios/{ticker}"
        params = {"apikey": self.api_key}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if not data:
            return self._get_defaults()

        latest = data[0]  # Most recent period

        return {
            "revenue_growth": latest.get("revenueGrowth", 0),
            "gross_margin": latest.get("grossProfitMargin", 0),
            "operating_margin": latest.get("operatingProfitMargin", 0),
            "net_margin": latest.get("netProfitMargin", 0),
            "debt_to_equity": latest.get("debtEquityRatio", 0),
            "roe": latest.get("returnOnEquity", 0)
        }

    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings and price targets."""
        # Implementation details...
        pass

    def _get_defaults(self) -> Dict[str, float]:
        """Return default values if API fails."""
        return {
            "revenue_growth": 0,
            "gross_margin": 0,
            "operating_margin": 0,
            "net_margin": 0,
            "debt_to_equity": 0,
            "roe": 0
        }
```

---

### **Migration Checklist**

- [ ] Sign up for FMP account (free or paid)
- [ ] Sign up for Finnhub account (free)
- [ ] Add API keys to `.env` file
- [ ] Create `tools/fmp_tool.py`
- [ ] Create `tools/finnhub_enhanced_tool.py`
- [ ] Update `agents/financial_analyst.py` to use FMP
- [ ] Update `agents/market_intelligence.py` to use FMP + Finnhub
- [ ] Update `agents/qualitative_signal.py` to use FMP + Finnhub
- [ ] Update `agents/context_engine.py` to use FMP historical data
- [ ] Test with 10-20 sample tickers
- [ ] Verify all metrics are correctly mapped
- [ ] Set up rate limiting and error handling
- [ ] Keep sample data as fallback for demos
- [ ] Update documentation with new data sources
- [ ] Deploy to production

---

### **Benefits of This Approach**

✅ **Cost-Effective:** $14-29/month vs $200+ for alternatives
✅ **Comprehensive:** 85% coverage of required metrics
✅ **Reliable:** Industry-standard APIs with 99.9% uptime
✅ **Easy Integration:** RESTful APIs with Python support
✅ **Scalable:** Can upgrade to higher tiers as needed
✅ **Maintainable:** Well-documented APIs with active support
✅ **Real-Time:** Access to live market data
✅ **Historical:** 30+ years of historical data for pattern matching

---

**End of Complete System Guide**

**Last Updated:** 2025-01-06T14:30:00Z
