"""
Observability Integration for InvestmentIQ MVAS

Advanced Feature: LangSmith observability with comprehensive tracing,
monitoring, and feedback collection for multi-agent systems.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client, traceable

load_dotenv()


class ObservabilityManager:
    """
    Manages observability for InvestmentIQ MVAS.

    Features:
    - Distributed tracing with LangSmith
    - A2A communication tracking
    - Performance monitoring
    - Feedback collection
    """

    def __init__(self):
        self.enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

        if self.enabled:
            self.client = Client(
                api_key=os.getenv("LANGSMITH_API_KEY"),
                api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
            )
            self.project = os.getenv("LANGSMITH_PROJECT", "investment-iq-production")
        else:
            self.client = None
            self.project = None

    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self.enabled

    def create_feedback(
        self,
        run_id: str,
        score: float,
        key: str = "user-satisfaction",
        comment: Optional[str] = None
    ) -> None:
        """
        Collect user feedback for a specific run.

        Args:
            run_id: LangSmith run identifier
            score: Feedback score (0.0 to 1.0)
            key: Feedback key/metric name
            comment: Optional feedback comment
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
        except Exception as e:
            print(f"Warning: Failed to create feedback: {e}")

    def get_project_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for the current project.

        Returns:
            Dictionary with project performance metrics
        """
        if not self.enabled or not self.client:
            return {"error": "Observability not enabled"}

        try:
            # Placeholder for metrics aggregation
            # In production, you'd query LangSmith API for metrics
            return {
                "project": self.project,
                "status": "active",
                "tracing_enabled": True
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_observability_manager = None


def get_observability_manager() -> ObservabilityManager:
    """Get singleton observability manager instance."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


# Decorator for agent tracing with custom metadata
def trace_agent(
    agent_name: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator for tracing agent operations with LangSmith.

    Args:
        agent_name: Name of the agent being traced
        metadata: Additional metadata to include in trace

    Example:
        @trace_agent("financial_analyst", {"version": "1.0"})
        async def analyze_financials(data):
            ...
    """
    def decorator(func):
        base_metadata = {
            "agent": agent_name,
            "component": "investment_iq_mvas",
            **(metadata or {})
        }

        return traceable(
            name=f"{agent_name}_{func.__name__}",
            metadata=base_metadata
        )(func)

    return decorator


# Decorator for A2A message tracing
def trace_a2a_message(message_type: str):
    """
    Decorator for tracing A2A communication.

    Args:
        message_type: Type of A2A message (request, response, etc.)

    Example:
        @trace_a2a_message("request")
        def send_request(message):
            ...
    """
    def decorator(func):
        return traceable(
            name=f"a2a_{message_type}",
            metadata={"message_type": message_type, "component": "a2a_protocol"}
        )(func)

    return decorator


# Decorator for tool usage tracing
def trace_tool(tool_name: str):
    """
    Decorator for tracing tool usage (MCP pattern).

    Args:
        tool_name: Name of the tool being traced

    Example:
        @trace_tool("financial_data_tool")
        def get_financial_data(company_id):
            ...
    """
    def decorator(func):
        return traceable(
            name=f"tool_{tool_name}",
            metadata={"tool": tool_name, "pattern": "mcp"}
        )(func)

    return decorator
