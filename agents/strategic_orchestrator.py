"""
Strategic Orchestrator Agent

Responsible for managing the entire workflow, coordinating A2A communication,
orchestrating debate/consensus, and making final investment recommendations.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from agents.base_agent import BaseAgent, AgentRole, AgentResponse, MessageType
from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent
from agents.context_engine import ContextEngineAgent
from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent
from core.agent_bus import get_agent_bus
from core.agent_contracts import (
    AgentOutput,
    Hypothesis,
    Counterpoint,
    Consensus,
    MessageType as ContractMessageType
)
from core.signal_fusion import SignalFusion
from core.confidence import EnsembleConfidence
from utils.observability import trace_agent

logger = logging.getLogger(__name__)


class StrategicOrchestratorAgent(BaseAgent):
    """
    Agent responsible for orchestrating the entire analysis workflow.

    Coordinates 5 specialist agents:
    1. Financial Analyst
    2. Qualitative Signal
    3. Context Engine
    4. Workforce Intelligence
    5. Market Intelligence

    Implements debate/consensus mechanism for conflicting signals.
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
        super().__init__(agent_id, AgentRole.STRATEGIC_ORCHESTRATOR)
        self.financial_agent = financial_agent
        self.qualitative_agent = qualitative_agent
        self.context_agent = context_agent
        self.workforce_agent = workforce_agent
        self.market_agent = market_agent
        self.workflow_log: List[Dict[str, Any]] = []

        # New components
        self.agent_bus = get_agent_bus()
        self.signal_fusion = SignalFusion(method="weighted_average")

        logger.info(
            f"StrategicOrchestrator initialized with "
            f"{3 + (1 if workforce_agent else 0) + (1 if market_agent else 0)} agents"
        )

    @trace_agent("strategic_orchestrator", {"version": "2.0"})
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process investment analysis request through 5-agent workflow.

        Expected request format:
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "analysis_depth": "comprehensive"
        }

        Returns:
            AgentResponse with fused signal and final recommendation
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
            "company_name": company_name
        })

        # Step 1: Gather all agent outputs in parallel
        self._log_workflow_step("step_1_parallel_analysis", {
            "action": "Calling all specialist agents in parallel"
        })

        agent_outputs = await self._gather_all_agent_outputs(
            ticker, company_name, sector
        )

        # Step 2: Detect conflicts using signal fusion
        conflicts = self.signal_fusion.detect_conflicts(
            {output.agent_id: output for output in agent_outputs},
            threshold=1.0
        )

        if conflicts:
            self._log_workflow_step("step_2_conflicts_detected", {
                "conflict_count": len(conflicts),
                "conflicts": conflicts
            })

            # Step 3: Orchestrate debate
            consensus = await self._orchestrate_debate(
                ticker, agent_outputs, conflicts
            )

            self._log_workflow_step("step_3_consensus_reached", {
                "final_recommendation": consensus.final_recommendation,
                "fused_score": consensus.fused_score
            })

            # Step 4: Generate final recommendation from consensus
            final_recommendation = self._generate_consensus_recommendation(
                ticker, consensus
            )
        else:
            self._log_workflow_step("step_2_no_major_conflicts", {
                "message": "Agents are in general agreement"
            })

            # Fuse signals without debate
            fused_signal = self.signal_fusion.fuse(ticker, agent_outputs)

            final_recommendation = self._generate_fused_recommendation(
                ticker, fused_signal
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
                "workflow_summary": self._create_workflow_summary()
            },
            metadata={
                "agent_role": self.role.value,
                "total_steps": len(self.workflow_log),
                "num_agents": len(agent_outputs)
            }
        )

    async def _gather_all_agent_outputs(
        self,
        ticker: str,
        company_name: str,
        sector: Optional[str]
    ) -> List[AgentOutput]:
        """Gather outputs from all available agents in parallel."""
        tasks = []

        # Always run core 3 agents (backward compatible)
        # These return AgentResponse, need to convert to AgentOutput
        # For now, run new agents that return AgentOutput

        outputs = []

        # Run new agents if available
        if self.workforce_agent:
            outputs.append(
                await self.workforce_agent.analyze(
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector
                )
            )

        if self.market_agent:
            outputs.append(
                await self.market_agent.analyze(
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector
                )
            )

        logger.info(f"Gathered {len(outputs)} agent outputs for {ticker}")

        return outputs

    async def _orchestrate_debate(
        self,
        ticker: str,
        agent_outputs: List[AgentOutput],
        conflicts: List[Dict[str, Any]]
    ) -> Consensus:
        """
        Orchestrate debate between conflicting agents.

        Returns:
            Consensus after debate
        """
        logger.info(f"Orchestrating debate for {ticker} with {len(conflicts)} conflicts")

        # Start debate with first major conflict
        if conflicts:
            conflict = conflicts[0]
            agent1_id = conflict["agent_1"]
            agent2_id = conflict["agent_2"]

            # Find agent outputs
            agent1_output = next(
                (o for o in agent_outputs if o.agent_id == agent1_id),
                None
            )
            agent2_output = next(
                (o for o in agent_outputs if o.agent_id == agent2_id),
                None
            )

            if agent1_output and agent2_output:
                # Create hypothesis from agent1
                hypothesis = Hypothesis(
                    agent_id=agent1_id,
                    ticker=ticker,
                    hypothesis=f"Sentiment is {agent1_output.sentiment:+.2f}",
                    supporting_evidence=agent1_output.evidence,
                    confidence=agent1_output.confidence
                )

                # Start debate
                debate_id = self.agent_bus.start_debate(ticker, hypothesis)

                # Add counterpoint from agent2
                counterpoint = Counterpoint(
                    agent_id=agent2_id,
                    original_hypothesis_id=debate_id,
                    counterpoint=f"Sentiment is {agent2_output.sentiment:+.2f}",
                    supporting_evidence=agent2_output.evidence,
                    confidence=agent2_output.confidence
                )

                self.agent_bus.add_counterpoint(debate_id, counterpoint)

        # Fuse signals to reach consensus
        agent_signals = {output.agent_id: output for output in agent_outputs}
        fused_signal = self.signal_fusion.fuse(ticker, agent_outputs)

        # Create consensus
        consensus = Consensus(
            participating_agents=[output.agent_id for output in agent_outputs],
            ticker=ticker,
            final_recommendation=self._determine_action(fused_signal.final_score),
            fused_score=fused_signal.final_score,
            calibrated_confidence=fused_signal.confidence,
            signal_contributions=fused_signal.signal_weights,
            supporting_evidence=fused_signal.top_evidence,
            conflicting_points=[c["description"] for c in conflicts],
            debate_rounds=1
        )

        # Broadcast consensus
        self.agent_bus.reach_consensus(consensus)

        return consensus

    def _determine_action(self, fused_score: float) -> str:
        """Determine investment action from fused score."""
        if fused_score >= 0.4:
            return "BUY"
        elif fused_score >= 0.1:
            return "ACCUMULATE"
        elif fused_score >= -0.1:
            return "HOLD"
        elif fused_score >= -0.4:
            return "REDUCE"
        else:
            return "SELL"

    def _generate_consensus_recommendation(
        self,
        ticker: str,
        consensus: Consensus
    ) -> Dict[str, Any]:
        """Generate final recommendation from consensus."""
        return {
            "ticker": ticker,
            "action": consensus.final_recommendation,
            "confidence": consensus.calibrated_confidence,
            "fused_score": consensus.fused_score,
            "reasoning": f"Consensus from {len(consensus.participating_agents)} agents",
            "signal_contributions": consensus.signal_contributions,
            "supporting_evidence": [
                {
                    "source": e.source,
                    "description": e.description,
                    "confidence": e.confidence
                }
                for e in consensus.supporting_evidence[:5]
            ],
            "conflicting_points": consensus.conflicting_points,
            "debate_rounds": consensus.debate_rounds
        }

    def _generate_fused_recommendation(
        self,
        ticker: str,
        fused_signal
    ) -> Dict[str, Any]:
        """Generate recommendation from fused signal without debate."""
        action = self._determine_action(fused_signal.final_score)

        return {
            "ticker": ticker,
            "action": action,
            "confidence": fused_signal.confidence,
            "fused_score": fused_signal.final_score,
            "reasoning": "All agents in agreement",
            "signal_contributions": fused_signal.signal_weights,
            "explanations": fused_signal.explanations,
            "supporting_evidence": [
                {
                    "source": e.source,
                    "description": e.description,
                    "confidence": e.confidence
                }
                for e in fused_signal.top_evidence[:5]
            ]
        }

    async def _gather_specialist_inputs(
        self,
        company_id: str
    ) -> tuple:
        """Gather inputs from specialist agents in parallel"""
        # Execute both agent calls concurrently
        financial_task = self.financial_agent.process({
            "company_id": company_id,
            "analysis_type": "comprehensive"
        })

        qualitative_task = self.qualitative_agent.process({
            "company_id": company_id,
            "focus_areas": ["sentiment", "workforce", "market_position"]
        })

        # Wait for both to complete
        financial_response, qualitative_response = await asyncio.gather(
            financial_task,
            qualitative_task
        )

        # Log A2A communication
        self._log_a2a_message(
            "strategic_orchestrator",
            "financial_analyst",
            MessageType.REQUEST,
            {"company_id": company_id}
        )

        self._log_a2a_message(
            "financial_analyst",
            "strategic_orchestrator",
            MessageType.RESPONSE,
            financial_response.data
        )

        self._log_a2a_message(
            "strategic_orchestrator",
            "qualitative_signal",
            MessageType.REQUEST,
            {"company_id": company_id}
        )

        self._log_a2a_message(
            "qualitative_signal",
            "strategic_orchestrator",
            MessageType.RESPONSE,
            qualitative_response.data
        )

        return financial_response, qualitative_response

    def _detect_conflict(
        self,
        financial_response: AgentResponse,
        qualitative_response: AgentResponse
    ) -> Dict[str, Any]:
        """Detect conflicts between financial and qualitative signals"""
        financial_health = financial_response.data.get("financial_health", "").lower()
        sentiment = qualitative_response.data.get("overall_sentiment", "").lower()

        # Define conflict conditions
        has_conflict = False
        conflict_type = None
        severity = "low"

        # Contrarian scenario: Strong financials + Very negative sentiment
        if financial_health == "strong" and "negative" in sentiment:
            has_conflict = True
            conflict_type = "contrarian_opportunity"
            severity = "high" if "very" in sentiment else "medium"

        # Deteriorating fundamentals + Positive sentiment
        elif financial_health == "weak" and "positive" in sentiment:
            has_conflict = True
            conflict_type = "overvalued_risk"
            severity = "high"

        return {
            "has_conflict": has_conflict,
            "conflict_type": conflict_type,
            "severity": severity,
            "financial_signal": financial_health,
            "qualitative_signal": sentiment,
            "description": self._get_conflict_description(
                conflict_type,
                financial_health,
                sentiment
            )
        }

    def _get_conflict_description(
        self,
        conflict_type: Optional[str],
        financial_signal: str,
        qualitative_signal: str
    ) -> str:
        """Generate human-readable conflict description"""
        if not conflict_type:
            return "No significant conflict detected"

        if conflict_type == "contrarian_opportunity":
            return (
                f"Contrarian scenario detected: Strong financial fundamentals "
                f"({financial_signal}) contradicted by {qualitative_signal} sentiment"
            )
        elif conflict_type == "overvalued_risk":
            return (
                f"Overvaluation risk: Weak fundamentals ({financial_signal}) "
                f"masked by {qualitative_signal} sentiment"
            )

        return "Unclassified conflict"

    async def _resolve_conflict(
        self,
        financial_response: AgentResponse,
        qualitative_response: AgentResponse,
        conflict: Dict[str, Any]
    ) -> AgentResponse:
        """Resolve conflict using Context Engine"""
        # Prepare context for the Context Engine
        context = {
            "financial_health": financial_response.data.get("financial_health"),
            "sentiment": qualitative_response.data.get("overall_sentiment"),
            "sector": "technology",  # In production, this would come from company data
            "gross_margin": financial_response.data.get("key_metrics", {}).get("gross_margin", 0)
        }

        # Call Context Engine
        resolution = await self.context_agent.process({
            "scenario_type": conflict["conflict_type"],
            "context": context
        })

        # Log A2A communication
        self._log_a2a_message(
            "strategic_orchestrator",
            "context_engine",
            MessageType.REQUEST,
            {"scenario_type": conflict["conflict_type"], "context": context}
        )

        self._log_a2a_message(
            "context_engine",
            "strategic_orchestrator",
            MessageType.RESPONSE,
            resolution.data
        )

        return resolution

    def _generate_recommendation(
        self,
        financial_response: AgentResponse,
        qualitative_response: AgentResponse,
        resolution: AgentResponse,
        conflict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final investment recommendation based on conflict resolution"""
        recommendation_text = resolution.data.get("recommendation", "")
        confidence = resolution.data.get("confidence", 0.5)
        rationale = resolution.data.get("rationale", "")

        # Determine action
        if conflict["conflict_type"] == "contrarian_opportunity" and confidence >= 0.7:
            action = "ACCUMULATE"
            position_size = "moderate"
            reasoning = (
                "Contrarian opportunity identified. Market appears to have overreacted "
                "to temporary sentiment issues while core fundamentals remain strong."
            )
        elif confidence >= 0.6:
            action = "HOLD"
            position_size = "existing"
            reasoning = "Mixed signals suggest maintaining current position with close monitoring."
        else:
            action = "REDUCE"
            position_size = "minimal"
            reasoning = "Uncertainty too high - recommend reducing exposure."

        return {
            "action": action,
            "position_size": position_size,
            "confidence": confidence,
            "reasoning": reasoning,
            "context_rule_applied": resolution.data.get("rule_applied", {}),
            "supporting_factors": {
                "financial_health": financial_response.data.get("financial_health"),
                "sentiment": qualitative_response.data.get("overall_sentiment"),
                "risk_level": qualitative_response.data.get("risk_assessment", {}).get("risk_level")
            },
            "detailed_rationale": rationale
        }

    def _generate_aligned_recommendation(
        self,
        financial_response: AgentResponse,
        qualitative_response: AgentResponse
    ) -> Dict[str, Any]:
        """Generate recommendation when signals are aligned"""
        financial_health = financial_response.data.get("financial_health", "").lower()
        sentiment = qualitative_response.data.get("overall_sentiment", "").lower()

        if financial_health == "strong" and "positive" in sentiment:
            action = "BUY"
            confidence = 0.85
            reasoning = "Both fundamental and sentiment analysis are positive"
        elif financial_health == "weak" and "negative" in sentiment:
            action = "SELL"
            confidence = 0.80
            reasoning = "Both fundamental and sentiment analysis are negative"
        else:
            action = "HOLD"
            confidence = 0.65
            reasoning = "Moderate signals suggest maintaining current position"

        return {
            "action": action,
            "position_size": "standard",
            "confidence": confidence,
            "reasoning": reasoning,
            "supporting_factors": {
                "financial_health": financial_health,
                "sentiment": sentiment
            }
        }

    def _log_workflow_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """Log a workflow step for audit trail"""
        self.workflow_log.append({
            "step": step_name,
            "data": data,
            "timestamp": self._get_timestamp()
        })

    def _log_a2a_message(
        self,
        sender: str,
        receiver: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> None:
        """Log agent-to-agent communication"""
        message = self.create_message(receiver, message_type, payload)
        self.log_message(message)

    def _create_workflow_summary(self) -> List[Dict[str, Any]]:
        """Create a summary of the workflow execution"""
        return [
            {
                "step": log["step"],
                "timestamp": log["timestamp"]
            }
            for log in self.workflow_log
        ]

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def get_detailed_workflow_log(self) -> List[Dict[str, Any]]:
        """Return the complete workflow log for analysis"""
        return self.workflow_log

    def get_a2a_message_history(self) -> List[Dict[str, Any]]:
        """Return A2A message history"""
        return [msg.to_dict() for msg in self.message_history]
