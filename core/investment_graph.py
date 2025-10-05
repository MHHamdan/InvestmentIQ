"""
LangGraph Investment Analysis Workflow

Defines the state graph for multi-agent investment analysis using LangGraph.
Coordinates 5 specialist agents with debate/consensus mechanisms.
"""

import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.agent_contracts import AgentOutput, FusedSignal, Consensus
from core.signal_fusion import SignalFusion

logger = logging.getLogger(__name__)


class InvestmentAnalysisState(TypedDict):
    """
    State schema for investment analysis workflow.

    This state is passed through all nodes in the graph.
    """
    # Input
    ticker: str
    company_name: str
    sector: Optional[str]

    # Agent outputs (accumulated)
    agent_outputs: Annotated[List[AgentOutput], operator.add]

    # Analysis results
    financial_analysis: Optional[Dict[str, Any]]
    qualitative_analysis: Optional[Dict[str, Any]]
    context_analysis: Optional[Dict[str, Any]]
    workforce_analysis: Optional[Dict[str, Any]]
    market_intelligence: Optional[Dict[str, Any]]

    # Signal fusion
    fused_signal: Optional[FusedSignal]
    conflicts: List[Dict[str, Any]]

    # Consensus
    consensus: Optional[Consensus]
    requires_debate: bool

    # Final output
    recommendation: Optional[Dict[str, Any]]

    # Metadata
    workflow_log: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]


def create_investment_graph(
    financial_agent,
    qualitative_agent,
    context_agent,
    workforce_agent=None,
    market_agent=None
):
    """
    Create LangGraph workflow for investment analysis.

    Args:
        financial_agent: FinancialAnalystAgent instance
        qualitative_agent: QualitativeSignalAgent instance
        context_agent: ContextEngineAgent instance
        workforce_agent: WorkforceIntelligenceAgent instance (optional)
        market_agent: MarketIntelligenceAgent instance (optional)

    Returns:
        Compiled StateGraph
    """

    # Initialize signal fusion
    signal_fusion = SignalFusion(method="weighted_average", use_llm_summary=True)

    # Define nodes

    async def financial_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Run financial analysis agent."""
        logger.info(f"Financial analysis node for {state['ticker']}")

        try:
            output = await financial_agent.analyze(
                ticker=state["ticker"],
                company_name=state["company_name"],
                sector=state.get("sector")
            )

            return {
                "agent_outputs": [output],
                "financial_analysis": {
                    "sentiment": output.sentiment,
                    "confidence": output.confidence,
                    "metrics": output.metrics
                },
                "workflow_log": [{
                    "node": "financial_analysis",
                    "status": "completed",
                    "ticker": state["ticker"]
                }]
            }

        except Exception as e:
            logger.error(f"Financial analysis failed: {e}")
            return {
                "errors": [f"Financial analysis error: {str(e)}"],
                "workflow_log": [{
                    "node": "financial_analysis",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    async def qualitative_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Run qualitative signal analysis agent."""
        logger.info(f"Qualitative analysis node for {state['ticker']}")

        try:
            output = await qualitative_agent.analyze(
                ticker=state["ticker"],
                company_name=state["company_name"],
                sector=state.get("sector")
            )

            return {
                "agent_outputs": [output],
                "qualitative_analysis": {
                    "sentiment": output.sentiment,
                    "confidence": output.confidence,
                    "metrics": output.metrics
                },
                "workflow_log": [{
                    "node": "qualitative_analysis",
                    "status": "completed",
                    "ticker": state["ticker"]
                }]
            }

        except Exception as e:
            logger.error(f"Qualitative analysis failed: {e}")
            return {
                "errors": [f"Qualitative analysis error: {str(e)}"],
                "workflow_log": [{
                    "node": "qualitative_analysis",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    async def context_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Run context engine agent."""
        logger.info(f"Context analysis node for {state['ticker']}")

        try:
            output = await context_agent.analyze(
                ticker=state["ticker"],
                company_name=state["company_name"],
                sector=state.get("sector")
            )

            return {
                "agent_outputs": [output],
                "context_analysis": {
                    "sentiment": output.sentiment,
                    "confidence": output.confidence,
                    "metrics": output.metrics
                },
                "workflow_log": [{
                    "node": "context_analysis",
                    "status": "completed",
                    "ticker": state["ticker"]
                }]
            }

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {
                "errors": [f"Context analysis error: {str(e)}"],
                "workflow_log": [{
                    "node": "context_analysis",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    async def workforce_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Run workforce intelligence agent."""
        if not workforce_agent:
            return {
                "workflow_log": [{
                    "node": "workforce_analysis",
                    "status": "skipped",
                    "reason": "agent_not_configured"
                }]
            }

        logger.info(f"Workforce analysis node for {state['ticker']}")

        try:
            output = await workforce_agent.analyze(
                ticker=state["ticker"],
                company_name=state["company_name"],
                sector=state.get("sector")
            )

            return {
                "agent_outputs": [output],
                "workforce_analysis": {
                    "sentiment": output.sentiment,
                    "confidence": output.confidence,
                    "metrics": output.metrics
                },
                "workflow_log": [{
                    "node": "workforce_analysis",
                    "status": "completed",
                    "ticker": state["ticker"]
                }]
            }

        except Exception as e:
            logger.error(f"Workforce analysis failed: {e}")
            return {
                "errors": [f"Workforce analysis error: {str(e)}"],
                "workflow_log": [{
                    "node": "workforce_analysis",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    async def market_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Run market intelligence agent."""
        if not market_agent:
            return {
                "workflow_log": [{
                    "node": "market_intelligence",
                    "status": "skipped",
                    "reason": "agent_not_configured"
                }]
            }

        logger.info(f"Market intelligence node for {state['ticker']}")

        try:
            output = await market_agent.analyze(
                ticker=state["ticker"],
                company_name=state["company_name"],
                sector=state.get("sector")
            )

            return {
                "agent_outputs": [output],
                "market_intelligence": {
                    "sentiment": output.sentiment,
                    "confidence": output.confidence,
                    "metrics": output.metrics
                },
                "workflow_log": [{
                    "node": "market_intelligence",
                    "status": "completed",
                    "ticker": state["ticker"]
                }]
            }

        except Exception as e:
            logger.error(f"Market intelligence failed: {e}")
            return {
                "errors": [f"Market intelligence error: {str(e)}"],
                "workflow_log": [{
                    "node": "market_intelligence",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    def fusion_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Fuse agent signals and detect conflicts."""
        logger.info(f"Signal fusion node for {state['ticker']}")

        agent_outputs = state.get("agent_outputs", [])

        if not agent_outputs:
            return {
                "errors": ["No agent outputs to fuse"],
                "workflow_log": [{
                    "node": "signal_fusion",
                    "status": "failed",
                    "reason": "no_outputs"
                }]
            }

        try:
            # Fuse signals
            fused = signal_fusion.fuse(
                ticker=state["ticker"],
                agent_outputs=agent_outputs
            )

            # Detect conflicts
            conflicts = signal_fusion.detect_conflicts(
                agent_signals={o.agent_id: o for o in agent_outputs},
                threshold=1.0
            )

            requires_debate = len(conflicts) > 0

            return {
                "fused_signal": fused,
                "conflicts": conflicts,
                "requires_debate": requires_debate,
                "workflow_log": [{
                    "node": "signal_fusion",
                    "status": "completed",
                    "conflicts_detected": len(conflicts),
                    "fused_score": fused.final_score
                }]
            }

        except Exception as e:
            logger.error(f"Signal fusion failed: {e}")
            return {
                "errors": [f"Signal fusion error: {str(e)}"],
                "workflow_log": [{
                    "node": "signal_fusion",
                    "status": "failed",
                    "error": str(e)
                }]
            }

    def debate_router(state: InvestmentAnalysisState) -> str:
        """Route to debate if conflicts detected."""
        if state.get("requires_debate", False):
            logger.info(f"Routing to debate for {state['ticker']}")
            return "debate"
        else:
            logger.info(f"Routing to recommendation for {state['ticker']}")
            return "recommendation"

    def debate_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Orchestrate debate between conflicting agents."""
        logger.info(f"Debate node for {state['ticker']}")

        fused_signal = state.get("fused_signal")
        conflicts = state.get("conflicts", [])
        agent_outputs = state.get("agent_outputs", [])

        if not fused_signal:
            return {
                "errors": ["No fused signal for debate"],
                "workflow_log": [{
                    "node": "debate",
                    "status": "failed",
                    "reason": "no_fused_signal"
                }]
            }

        # Create consensus from debate
        consensus = Consensus(
            participating_agents=[o.agent_id for o in agent_outputs],
            ticker=state["ticker"],
            final_recommendation=_determine_action(fused_signal.final_score),
            fused_score=fused_signal.final_score,
            calibrated_confidence=fused_signal.confidence,
            signal_contributions=fused_signal.signal_weights,
            supporting_evidence=fused_signal.top_evidence,
            conflicting_points=[c["description"] for c in conflicts],
            debate_rounds=1
        )

        return {
            "consensus": consensus,
            "workflow_log": [{
                "node": "debate",
                "status": "completed",
                "conflicts_resolved": len(conflicts),
                "consensus_action": consensus.final_recommendation
            }]
        }

    def recommendation_node(state: InvestmentAnalysisState) -> InvestmentAnalysisState:
        """Generate final recommendation."""
        logger.info(f"Recommendation node for {state['ticker']}")

        fused_signal = state.get("fused_signal")
        consensus = state.get("consensus")

        if consensus:
            # Recommendation from consensus (after debate)
            recommendation = {
                "ticker": state["ticker"],
                "action": consensus.final_recommendation,
                "confidence": consensus.calibrated_confidence,
                "fused_score": consensus.fused_score,
                "reasoning": f"Consensus from {len(consensus.participating_agents)} agents after debate",
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
                "debate_rounds": consensus.debate_rounds,
                "llm_summary": fused_signal.llm_summary if fused_signal else None
            }
        elif fused_signal:
            # Direct recommendation from fusion (no debate needed)
            action = _determine_action(fused_signal.final_score)
            recommendation = {
                "ticker": state["ticker"],
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
                ],
                "llm_summary": fused_signal.llm_summary
            }
        else:
            return {
                "errors": ["No fused signal or consensus for recommendation"],
                "workflow_log": [{
                    "node": "recommendation",
                    "status": "failed",
                    "reason": "no_signal_or_consensus"
                }]
            }

        return {
            "recommendation": recommendation,
            "workflow_log": [{
                "node": "recommendation",
                "status": "completed",
                "action": recommendation["action"],
                "confidence": recommendation["confidence"]
            }]
        }

    # Build graph
    workflow = StateGraph(InvestmentAnalysisState)

    # Add nodes
    workflow.add_node("financial_analysis", financial_node)
    workflow.add_node("qualitative_analysis", qualitative_node)
    workflow.add_node("context_analysis", context_node)
    workflow.add_node("workforce_analysis", workforce_node)
    workflow.add_node("market_intelligence", market_node)
    workflow.add_node("signal_fusion", fusion_node)
    workflow.add_node("debate", debate_node)
    workflow.add_node("recommendation", recommendation_node)

    # Set entry point
    workflow.set_entry_point("financial_analysis")

    # Add edges for parallel agent execution
    workflow.add_edge("financial_analysis", "qualitative_analysis")
    workflow.add_edge("qualitative_analysis", "context_analysis")
    workflow.add_edge("context_analysis", "workforce_analysis")
    workflow.add_edge("workforce_analysis", "market_intelligence")

    # After all agents, go to fusion
    workflow.add_edge("market_intelligence", "signal_fusion")

    # Conditional routing from fusion
    workflow.add_conditional_edges(
        "signal_fusion",
        debate_router,
        {
            "debate": "debate",
            "recommendation": "recommendation"
        }
    )

    # Debate leads to recommendation
    workflow.add_edge("debate", "recommendation")

    # Recommendation is the end
    workflow.add_edge("recommendation", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("LangGraph investment workflow compiled successfully")

    return app


def _determine_action(fused_score: float) -> str:
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
