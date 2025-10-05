"""
Agent Communication Contracts

Pydantic models for A2A messages and standardized agent outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """A2A message types for agent communication."""
    OBSERVATION = "observation"
    ALERT = "alert"
    QUERY = "query"
    HYPOTHESIS = "hypothesis"
    COUNTERPOINT = "counterpoint"
    CONSENSUS = "consensus"


class SignalType(str, Enum):
    """Types of signals agents can produce."""
    FINANCIAL = "financial"
    SENTIMENT = "sentiment"
    WORKFORCE = "workforce"
    MARKET_INTELLIGENCE = "market_intelligence"
    CONTEXT = "context"


class AgentMessage(BaseModel):
    """Standardized A2A message format."""
    message_type: MessageType
    sender: str
    receiver: Optional[str] = None  # None = broadcast
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)  # 1=lowest, 5=highest


class Evidence(BaseModel):
    """Evidence supporting a signal."""
    source: str
    value: Any
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    url: Optional[str] = None
    description: Optional[str] = None


class AgentOutput(BaseModel):
    """Unified output schema for all agents."""
    signal: SignalType
    agent_id: str
    ticker: str

    # Core metrics
    metrics: Dict[str, Any]
    sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Supporting data
    evidence: List[Evidence]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional context
    alerts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Agent observation message."""
    agent_id: str
    ticker: str
    observation: str
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    """Agent alert message."""
    agent_id: str
    ticker: str
    severity: Literal["low", "medium", "high", "critical"]
    title: str
    description: str
    recommended_action: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Hypothesis(BaseModel):
    """Agent hypothesis for debate."""
    agent_id: str
    ticker: str
    hypothesis: str
    supporting_evidence: List[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Counterpoint(BaseModel):
    """Counterargument to a hypothesis."""
    agent_id: str
    original_hypothesis_id: str
    counterpoint: str
    supporting_evidence: List[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Consensus(BaseModel):
    """Final consensus after debate."""
    participating_agents: List[str]
    ticker: str
    final_recommendation: str
    fused_score: float = Field(ge=-1.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)

    # Signal contributions
    signal_contributions: Dict[str, float]  # agent_id -> weight

    # Evidence
    supporting_evidence: List[Evidence]
    conflicting_points: List[str] = Field(default_factory=list)

    # Metadata
    debate_rounds: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FusedSignal(BaseModel):
    """Result of signal fusion."""
    ticker: str
    final_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Breakdown by agent
    agent_signals: Dict[str, AgentOutput]
    signal_weights: Dict[str, float]

    # Explainability
    explanations: List[str]
    top_evidence: List[Evidence]

    # Metadata
    fusion_method: str = "weighted_average"
    sector: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
