"""
Base Agent Abstract Class for InvestmentIQ MVAS

This module defines the core agent interface that all specialized agents must implement.
It establishes the contract for agent communication and tool usage.

Enhanced with LangSmith observability for tracing and monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from langsmith import traceable
import os


class AgentRole(Enum):
    FINANCIAL_ANALYST = "financial_analyst"
    QUALITATIVE_SIGNAL = "qualitative_signal"
    CONTEXT_ENGINE = "context_engine"
    STRATEGIC_ORCHESTRATOR = "strategic_orchestrator"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    INFO = "info"


@dataclass
class AgentMessage:
    """Standardized message format for A2A communication"""
    sender: str
    receiver: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


@dataclass
class AgentResponse:
    """Standardized response format for agent outputs"""
    agent_id: str
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the InvestmentIQ MVAS.

    Each agent must implement the process method and can optionally
    use tools for data access and processing.
    """

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.message_history: List[AgentMessage] = []

        # LangSmith tracing enabled
        self.tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    @abstractmethod
    @traceable(
        name="agent_process",
        metadata={"component": "base_agent"}
    )
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process an incoming request and return a structured response.

        Enhanced with LangSmith tracing for observability.

        Args:
            request: Dictionary containing the request parameters

        Returns:
            AgentResponse object with structured data
        """
        pass

    def log_message(self, message: AgentMessage) -> None:
        """Log a message for audit trail"""
        self.message_history.append(message)

    def create_message(
        self,
        receiver: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Create a standardized agent message"""
        return AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id
        )

    def create_response(
        self,
        status: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Create a standardized agent response"""
        return AgentResponse(
            agent_id=self.agent_id,
            status=status,
            data=data,
            metadata=metadata or {},
            timestamp=datetime.utcnow().isoformat()
        )
