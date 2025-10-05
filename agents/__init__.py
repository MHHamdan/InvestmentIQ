"""
Agents module for InvestmentIQ MVAS

This package contains all agent implementations for the multi-agent system.
"""

from .base_agent import (
    BaseAgent,
    AgentRole,
    AgentMessage,
    AgentResponse,
    MessageType
)
from .strategic_orchestrator import StrategicOrchestratorAgent
from .workforce_intelligence import WorkforceIntelligenceAgent
from .market_intelligence import MarketIntelligenceAgent

__all__ = [
    "BaseAgent",
    "AgentRole",
    "AgentMessage",
    "AgentResponse",
    "MessageType",
    "StrategicOrchestratorAgent",
    "WorkforceIntelligenceAgent",
    "MarketIntelligenceAgent"
]
