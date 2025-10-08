"""
Strategic Orchestrator Agent - LangGraph Edition

Orchestrates multi-agent investment analysis using LangGraph workflows.
Coordinates 5 specialist agents with debate/consensus mechanisms.
"""

from typing import Dict, Any, List, Optional
import logging
import time
from agents.base_agent import BaseAgent, AgentRole, AgentResponse
from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent
from agents.context_engine import ContextEngineAgent
from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent
from core.investment_graph import create_investment_graph
from utils.observability import trace_agent

logger = logging.getLogger(__name__)


class StrategicOrchestratorAgent(BaseAgent):
    """
    Strategic orchestrator using LangGraph for workflow management.

    Coordinates 5 specialist agents through a declarative graph workflow:
    1. Financial Analyst
    2. Qualitative Signal
    3. Context Engine
    4. Workforce Intelligence
    5. Market Intelligence

    Features:
    - Declarative LangGraph workflow
    - Automatic state management
    - Conditional routing based on conflicts
    - Built-in checkpointing
    - Workflow logging
    """

    def __init__(
        self,
        agent_id: str,
        financial_agent: FinancialAnalystAgent,
        qualitative_agent: QualitativeSignalAgent,
        context_agent: ContextEngineAgent,
        workforce_agent: Optional[WorkforceIntelligenceAgent] = None,
        market_agent: Optional[MarketIntelligenceAgent] = None
    ):
        """
        Initialize Strategic Orchestrator with LangGraph workflow.

        Args:
            agent_id: Unique identifier for this agent
            financial_agent: Financial analysis agent
            qualitative_agent: Qualitative signal agent
            context_agent: Context engine agent
            workforce_agent: Workforce intelligence agent (optional)
            market_agent: Market intelligence agent (optional)
        """
        super().__init__(agent_id, AgentRole.STRATEGIC_ORCHESTRATOR)

        self.financial_agent = financial_agent
        self.qualitative_agent = qualitative_agent
        self.context_agent = context_agent
        self.workforce_agent = workforce_agent
        self.market_agent = market_agent
        self.workflow_log: List[Dict[str, Any]] = []

        # Create LangGraph workflow
        self.graph = create_investment_graph(
            financial_agent=financial_agent,
            qualitative_agent=qualitative_agent,
            context_agent=context_agent,
            workforce_agent=workforce_agent,
            market_agent=market_agent
        )

        num_agents = 3 + (1 if workforce_agent else 0) + (1 if market_agent else 0)
        logger.info(
            f"StrategicOrchestrator initialized with LangGraph workflow "
            f"({num_agents} agents)"
        )

    @trace_agent("strategic_orchestrator", {"version": "3.0-langgraph"})
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process investment analysis request through LangGraph workflow.

        The workflow executes these steps automatically:
        1. Run all 5 agents (financial, qualitative, context, workforce, market)
        2. Fuse signals and detect conflicts
        3. Route to debate if conflicts exist, otherwise to recommendation
        4. Generate final recommendation with LLM reasoning

        Args:
            request: Analysis request containing:
                - ticker: Stock ticker (required)
                - company_name: Company name (optional, defaults to ticker)
                - sector: Industry sector (optional)
                - analysis_depth: Analysis depth (optional)

        Returns:
            AgentResponse with:
                - status: "success" or "error"
                - data: Contains recommendation, agent_outputs, workflow_log
                - metadata: Contains workflow_type, total_steps, num_agents

        Example:
            request = {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "sector": "Technology"
            }
            response = await orchestrator.process(request)
            recommendation = response.data["recommendation"]
        """
        ticker = request.get("ticker") or request.get("company_id")
        company_name = request.get("company_name", ticker)
        sector = request.get("sector")

        if not ticker:
            return self.create_response(
                status="error",
                data={"error": "ticker is required"},
                metadata={"request": request}
            )

        self._log_workflow_step("workflow_initiated", {
            "ticker": ticker,
            "company_name": company_name,
            "workflow_type": "LangGraph"
        })

        logger.info(f"Executing LangGraph workflow for {ticker}")

        try:
            # Initialize LangGraph state
            initial_state = {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "agent_outputs": [],
                "financial_analysis": None,
                "qualitative_analysis": None,
                "context_analysis": None,
                "workforce_analysis": None,
                "market_intelligence": None,
                "fused_signal": None,
                "conflicts": [],
                "consensus": None,
                "requires_debate": False,
                "recommendation": None,
                "workflow_log": [],
                "errors": []
            }

            # Execute LangGraph workflow with unique thread_id to prevent state accumulation
            thread_id = f"{ticker}_{int(time.time() * 1000)}"
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.graph.ainvoke(initial_state, config)

            # Extract results
            final_recommendation = final_state.get("recommendation")
            workflow_log = final_state.get("workflow_log", [])
            errors = final_state.get("errors", [])
            agent_outputs = final_state.get("agent_outputs", [])
            conflicts = final_state.get("conflicts", [])

            # Update internal workflow log
            self.workflow_log.extend(workflow_log)

            if errors:
                logger.warning(
                    f"Workflow completed with {len(errors)} errors: {errors}"
                )

            if not final_recommendation:
                return self.create_response(
                    status="error",
                    data={
                        "error": "Workflow did not produce recommendation",
                        "errors": errors,
                        "workflow_log": workflow_log
                    },
                    metadata={"request": request}
                )

            self._log_workflow_step("workflow_completed", final_recommendation)

            return self.create_response(
                status="success",
                data={
                    "ticker": ticker,
                    "company_name": company_name,
                    "recommendation": final_recommendation,
                    "agent_outputs": [output.dict() for output in agent_outputs],
                    "conflicts_detected": len(conflicts) > 0,
                    "workflow_summary": self._create_workflow_summary(),
                    "langgraph_workflow_log": workflow_log
                },
                metadata={
                    "agent_role": self.role.value,
                    "workflow_type": "LangGraph",
                    "total_steps": len(workflow_log),
                    "num_agents": len(agent_outputs),
                    "errors": len(errors)
                }
            )

        except Exception as e:
            logger.error(f"LangGraph workflow failed: {e}")
            import traceback
            traceback.print_exc()

            return self.create_response(
                status="error",
                data={
                    "error": f"Workflow execution failed: {str(e)}",
                    "ticker": ticker
                },
                metadata={"request": request}
            )

    def get_graph(self):
        """
        Get the compiled LangGraph workflow.

        Returns:
            Compiled StateGraph instance

        Example:
            graph = orchestrator.get_graph()
            nodes = list(graph.get_graph().nodes())
        """
        return self.graph

    def get_workflow_log(self) -> List[Dict[str, Any]]:
        """
        Get detailed workflow execution log.

        Returns:
            List of workflow step dictionaries

        Example:
            log = orchestrator.get_workflow_log()
            for step in log:
                print(f"{step['node']}: {step['status']}")
        """
        return self.workflow_log

    def _log_workflow_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """Log a workflow step for debugging and tracking."""
        self.workflow_log.append({
            "step": step_name,
            "data": data,
            "timestamp": self._get_timestamp()
        })
        logger.debug(f"Workflow step: {step_name}")

    def _create_workflow_summary(self) -> List[Dict[str, Any]]:
        """Create a summary of the workflow execution."""
        return [
            {
                "step": log.get("step"),
                "timestamp": log.get("timestamp")
            }
            for log in self.workflow_log
        ]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
