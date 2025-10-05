"""
Agent Communication Bus

In-process pub/sub for A2A communication with debate and consensus mechanisms.
"""

import asyncio
from typing import Dict, List, Callable, Optional, Set
from collections import defaultdict
from datetime import datetime
import logging

from core.agent_contracts import (
    AgentMessage,
    MessageType,
    Observation,
    Alert,
    Hypothesis,
    Counterpoint,
    Consensus
)

logger = logging.getLogger(__name__)


class AgentBus:
    """
    In-process pub/sub message bus for agent communication.

    Supports:
    - Topic-based subscriptions
    - Message broadcasting
    - Debate orchestration
    - Message history for audit
    """

    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_history: List[AgentMessage] = []
        self.active_debates: Dict[str, List[Hypothesis]] = {}  # ticker -> hypotheses
        self.active_counterpoints: Dict[str, List[Counterpoint]] = {}  # hypothesis_id -> counterpoints

    def subscribe(self, message_type: MessageType, handler: Callable):
        """Subscribe a handler to a message type."""
        self.subscribers[message_type].append(handler)
        logger.debug(f"Subscribed handler to {message_type}")

    def unsubscribe(self, message_type: MessageType, handler: Callable):
        """Unsubscribe a handler from a message type."""
        if handler in self.subscribers[message_type]:
            self.subscribers[message_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {message_type}")

    async def publish(self, message: AgentMessage):
        """
        Publish a message to all subscribers.

        Args:
            message: AgentMessage to publish
        """
        # Store in history
        self.message_history.append(message)

        # Route to subscribers
        handlers = self.subscribers.get(message.message_type, [])

        if not handlers:
            logger.debug(f"No subscribers for {message.message_type}")
            return

        # Execute handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Handler error for {message.message_type}: {e}")

    def broadcast_observation(self, observation: Observation) -> None:
        """Broadcast an observation to all agents."""
        message = AgentMessage(
            message_type=MessageType.OBSERVATION,
            sender=observation.agent_id,
            content=observation.dict()
        )
        asyncio.create_task(self.publish(message))

    def broadcast_alert(self, alert: Alert) -> None:
        """Broadcast an alert."""
        message = AgentMessage(
            message_type=MessageType.ALERT,
            sender=alert.agent_id,
            content=alert.dict(),
            priority=self._alert_to_priority(alert.severity)
        )
        asyncio.create_task(self.publish(message))

    def start_debate(self, ticker: str, hypothesis: Hypothesis) -> str:
        """
        Start a debate on a hypothesis.

        Args:
            ticker: Stock ticker
            hypothesis: Initial hypothesis

        Returns:
            Debate ID
        """
        if ticker not in self.active_debates:
            self.active_debates[ticker] = []

        self.active_debates[ticker].append(hypothesis)

        # Broadcast hypothesis
        message = AgentMessage(
            message_type=MessageType.HYPOTHESIS,
            sender=hypothesis.agent_id,
            content=hypothesis.dict(),
            priority=3
        )
        asyncio.create_task(self.publish(message))

        debate_id = f"{ticker}_{hypothesis.timestamp.isoformat()}"
        logger.info(f"Started debate {debate_id}: {hypothesis.hypothesis}")

        return debate_id

    def add_counterpoint(self, hypothesis_id: str, counterpoint: Counterpoint) -> None:
        """
        Add a counterpoint to an existing hypothesis.

        Args:
            hypothesis_id: ID of hypothesis being countered
            counterpoint: Counterpoint argument
        """
        if hypothesis_id not in self.active_counterpoints:
            self.active_counterpoints[hypothesis_id] = []

        self.active_counterpoints[hypothesis_id].append(counterpoint)

        # Broadcast counterpoint
        message = AgentMessage(
            message_type=MessageType.COUNTERPOINT,
            sender=counterpoint.agent_id,
            content=counterpoint.dict(),
            priority=3
        )
        asyncio.create_task(self.publish(message))

        logger.info(f"Added counterpoint to {hypothesis_id} from {counterpoint.agent_id}")

    def reach_consensus(self, consensus: Consensus) -> None:
        """
        Broadcast final consensus after debate.

        Args:
            consensus: Final consensus decision
        """
        message = AgentMessage(
            message_type=MessageType.CONSENSUS,
            sender="orchestrator",
            content=consensus.dict(),
            priority=5
        )
        asyncio.create_task(self.publish(message))

        # Clear debates for this ticker
        if consensus.ticker in self.active_debates:
            del self.active_debates[consensus.ticker]

        logger.info(f"Consensus reached for {consensus.ticker}: {consensus.final_recommendation}")

    def get_debate_summary(self, ticker: str) -> Dict:
        """
        Get summary of active debate for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Debate summary with hypotheses and counterpoints
        """
        hypotheses = self.active_debates.get(ticker, [])

        summary = {
            "ticker": ticker,
            "active": len(hypotheses) > 0,
            "hypotheses": [],
            "total_counterpoints": 0
        }

        for hyp in hypotheses:
            hyp_id = f"{ticker}_{hyp.timestamp.isoformat()}"
            counterpoints = self.active_counterpoints.get(hyp_id, [])

            summary["hypotheses"].append({
                "id": hyp_id,
                "agent": hyp.agent_id,
                "hypothesis": hyp.hypothesis,
                "confidence": hyp.confidence,
                "counterpoints": len(counterpoints),
                "counterpoint_details": [
                    {
                        "agent": cp.agent_id,
                        "counterpoint": cp.counterpoint,
                        "confidence": cp.confidence
                    }
                    for cp in counterpoints
                ]
            })

            summary["total_counterpoints"] += len(counterpoints)

        return summary

    def get_message_history(
        self,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Retrieve message history.

        Args:
            message_type: Filter by message type (optional)
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        messages = self.message_history

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages[-limit:]

    def clear_history(self):
        """Clear message history (for testing)."""
        self.message_history.clear()
        self.active_debates.clear()
        self.active_counterpoints.clear()

    @staticmethod
    def _alert_to_priority(severity: str) -> int:
        """Convert alert severity to message priority."""
        mapping = {
            "low": 2,
            "medium": 3,
            "high": 4,
            "critical": 5
        }
        return mapping.get(severity, 3)


# Singleton instance
_agent_bus = None


def get_agent_bus() -> AgentBus:
    """Get singleton agent bus instance."""
    global _agent_bus
    if _agent_bus is None:
        _agent_bus = AgentBus()
    return _agent_bus
