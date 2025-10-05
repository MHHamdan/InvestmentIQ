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
from core.investment_graph import create_investment_graph
from utils.observability import trace_agent
from utils.hf_client import get_hf_client

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
        self.signal_fusion = SignalFusion(method="weighted_average", use_llm_summary=True)

        # Initialize HF client for enhanced reasoning
        try:
            self.hf_client = get_hf_client()
            self.use_llm_reasoning = True
            logger.info("LLM reasoning enabled for strategic orchestrator")
        except Exception as e:
            logger.warning(f"Could not initialize HF client for reasoning: {e}")
            self.use_llm_reasoning = False

        # Create LangGraph workflow
        self.graph = create_investment_graph(
            financial_agent=financial_agent,
            qualitative_agent=qualitative_agent,
            context_agent=context_agent,
            workforce_agent=workforce_agent,
            market_agent=market_agent
        )

        logger.info(
            f"StrategicOrchestrator initialized with LangGraph workflow and "
            f"{3 + (1 if workforce_agent else 0) + (1 if market_agent else 0)} agents"
        )

    @trace_agent("strategic_orchestrator", {"version": "3.0-langgraph"})
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process investment analysis request through LangGraph workflow.

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
            "company_name": company_name,
            "workflow_type": "LangGraph"
        })

        # Execute LangGraph workflow
        logger.info(f"Executing LangGraph workflow for {ticker}")

        try:
            # Initialize state
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

            # Run the graph
            config = {"configurable": {"thread_id": ticker}}
            final_state = await self.graph.ainvoke(initial_state, config)

            # Extract results
            final_recommendation = final_state.get("recommendation")
            workflow_log = final_state.get("workflow_log", [])
            errors = final_state.get("errors", [])

            # Update internal workflow log
            self.workflow_log.extend(workflow_log)

            if errors:
                logger.warning(f"Workflow completed with {len(errors)} errors: {errors}")

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

            # Extract agent outputs from final state
            agent_outputs = final_state.get("agent_outputs", [])
            conflicts = final_state.get("conflicts", [])

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

        # Generate enhanced reasoning with LLM if available
        reasoning = self._generate_llm_reasoning(
            ticker,
            action,
            fused_signal
        ) if self.use_llm_reasoning else "All agents in agreement"

        return {
            "ticker": ticker,
            "action": action,
            "confidence": fused_signal.confidence,
            "fused_score": fused_signal.final_score,
            "reasoning": reasoning,
            "llm_summary": fused_signal.llm_summary,
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

    def _generate_llm_reasoning(
        self,
        ticker: str,
        action: str,
        fused_signal
    ) -> str:
        """
        Generate enhanced investment reasoning using LLM.

        Uses Hugging Face LLM to create professional, coherent reasoning
        that explains the recommendation based on multiple agent signals.

        Args:
            ticker: Stock ticker
            action: Recommended action (BUY/SELL/HOLD)
            fused_signal: Fused signal with all agent data

        Returns:
            Enhanced reasoning text
        """
        try:
            # Prepare agent signal summary
            signal_summary = []
            for agent_id, weight in sorted(
                fused_signal.signal_weights.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if agent_id in fused_signal.agent_signals:
                    output = fused_signal.agent_signals[agent_id]
                    sentiment_desc = "bullish" if output.sentiment > 0.2 else "bearish" if output.sentiment < -0.2 else "neutral"
                    signal_summary.append(
                        f"{agent_id}: {sentiment_desc} ({output.sentiment:+.2f}, confidence: {output.confidence:.2f})"
                    )

            # Prepare top evidence
            evidence_text = "\n".join([
                f"- {e.description}"
                for e in fused_signal.top_evidence[:3]
            ])

            # Create prompt
            prompt = f"""Generate a professional investment reasoning statement for {action} recommendation on {ticker}.

Fused Score: {fused_signal.final_score:.2f}
Confidence: {fused_signal.confidence:.2f}

Agent Signals:
{chr(10).join(signal_summary)}

Key Evidence:
{evidence_text}

Provide 2-3 sentences explaining why this {action} recommendation makes sense based on the signals and evidence. Write in a professional, analytical tone suitable for an investment report."""

            # Generate reasoning using HF client
            reasoning = self.hf_client.generate_text(
                prompt=prompt,
                max_tokens=250,
                temperature=0.3
            )

            if reasoning:
                logger.info(f"Generated LLM reasoning for {ticker} {action} recommendation")
                return reasoning
            else:
                logger.warning(f"Empty response from HF API for {ticker} reasoning")
                return f"{action} recommendation based on consensus from {len(fused_signal.agent_signals)} agents with {fused_signal.confidence:.1%} confidence."

        except Exception as e:
            logger.warning(f"Failed to generate LLM reasoning: {e}")
            # Fallback to simple reasoning
            return f"{action} recommendation based on consensus from {len(fused_signal.agent_signals)} agents with {fused_signal.confidence:.1%} confidence."
