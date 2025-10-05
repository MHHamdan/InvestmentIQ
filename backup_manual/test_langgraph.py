"""
Test LangGraph Integration for InvestmentIQ.

Tests the LangGraph-based workflow for multi-agent investment analysis.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.strategic_orchestrator import StrategicOrchestratorAgent
from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent
from agents.context_engine import ContextEngineAgent
from agents.workforce_intelligence import WorkforceIntelligenceAgent
from agents.market_intelligence import MarketIntelligenceAgent
from tools.data_tools import FinancialDataTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_langgraph_workflow():
    """Test complete LangGraph workflow."""
    logger.info("=" * 60)
    logger.info("Testing LangGraph Investment Analysis Workflow")
    logger.info("=" * 60)

    try:
        # Initialize agents
        logger.info("\n1. Initializing agents...")

        data_tool = FinancialDataTool(data_dir="data/samples")

        financial_agent = FinancialAnalystAgent(
            agent_id="financial_analyst",
            data_tool=data_tool
        )

        qualitative_agent = QualitativeSignalAgent(
            agent_id="qualitative_signal"
        )

        context_agent = ContextEngineAgent(
            agent_id="context_engine"
        )

        workforce_agent = WorkforceIntelligenceAgent(
            agent_id="workforce_intelligence"
        )

        market_agent = MarketIntelligenceAgent(
            agent_id="market_intelligence"
        )

        logger.info("✓ All 5 agents initialized")

        # Initialize orchestrator with LangGraph
        logger.info("\n2. Creating Strategic Orchestrator with LangGraph...")

        orchestrator = StrategicOrchestratorAgent(
            agent_id="strategic_orchestrator",
            financial_agent=financial_agent,
            qualitative_agent=qualitative_agent,
            context_agent=context_agent,
            workforce_agent=workforce_agent,
            market_agent=market_agent
        )

        logger.info("✓ Strategic Orchestrator created with LangGraph workflow")

        # Test workflow
        logger.info("\n3. Running LangGraph workflow for AAPL...")

        request = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "analysis_depth": "comprehensive"
        }

        response = await orchestrator.process(request)

        logger.info(f"\n4. Workflow completed with status: {response.status}")

        if response.status == "success":
            logger.info("\n✓ LangGraph Workflow SUCCESS")

            # Display results
            data = response.data
            recommendation = data.get("recommendation", {})

            logger.info("\n" + "=" * 60)
            logger.info("RECOMMENDATION")
            logger.info("=" * 60)
            logger.info(f"Ticker: {recommendation.get('ticker')}")
            logger.info(f"Action: {recommendation.get('action')}")
            logger.info(f"Confidence: {recommendation.get('confidence'):.2%}")
            logger.info(f"Fused Score: {recommendation.get('fused_score'):+.3f}")
            logger.info(f"Reasoning: {recommendation.get('reasoning')}")

            # Display workflow log
            workflow_log = data.get("langgraph_workflow_log", [])
            logger.info(f"\n✓ Workflow executed {len(workflow_log)} nodes:")
            for log_entry in workflow_log:
                node = log_entry.get("node", "unknown")
                status = log_entry.get("status", "unknown")
                logger.info(f"  - {node}: {status}")

            # Display agent contributions
            agent_outputs = data.get("agent_outputs", [])
            logger.info(f"\n✓ {len(agent_outputs)} agents participated:")
            for output in agent_outputs:
                agent_id = output.get("agent_id", "unknown")
                sentiment = output.get("sentiment", 0)
                confidence = output.get("confidence", 0)
                logger.info(f"  - {agent_id}: sentiment={sentiment:+.3f}, confidence={confidence:.2%}")

            # Check for conflicts
            conflicts_detected = data.get("conflicts_detected", False)
            if conflicts_detected:
                logger.info("\n⚠ Conflicts detected - debate mechanism activated")
            else:
                logger.info("\n✓ No major conflicts - agents in agreement")

            # Check metadata
            metadata = response.metadata
            logger.info(f"\n✓ Workflow Type: {metadata.get('workflow_type')}")
            logger.info(f"✓ Total Steps: {metadata.get('total_steps')}")
            logger.info(f"✓ Errors: {metadata.get('errors', 0)}")

            return True

        else:
            logger.error(f"\n✗ Workflow FAILED: {response.data}")
            return False

    except Exception as e:
        logger.error(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_visualization():
    """Test LangGraph visualization capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing LangGraph Visualization")
    logger.info("=" * 60)

    try:
        from agents.strategic_orchestrator import StrategicOrchestratorAgent
        from agents.financial_analyst import FinancialAnalystAgent
        from agents.qualitative_signal import QualitativeSignalAgent
        from agents.context_engine import ContextEngineAgent
        from tools.data_tools import FinancialDataTool

        # Create minimal orchestrator
        data_tool = FinancialDataTool(data_dir="data/samples")
        financial = FinancialAnalystAgent("financial", data_tool)
        qualitative = QualitativeSignalAgent("qualitative")
        context = ContextEngineAgent("context")

        orchestrator = StrategicOrchestratorAgent(
            "orchestrator",
            financial,
            qualitative,
            context
        )

        # Try to get graph visualization
        graph = orchestrator.graph

        logger.info("✓ LangGraph instance accessible")
        logger.info(f"  Graph type: {type(graph).__name__}")

        # List nodes
        try:
            nodes = list(graph.get_graph().nodes())
            logger.info(f"✓ Graph has {len(nodes)} nodes:")
            for node in nodes:
                logger.info(f"  - {node}")
        except Exception as e:
            logger.warning(f"  Could not list nodes: {e}")

        logger.info("\n✓ Graph structure verified")
        return True

    except Exception as e:
        logger.error(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("InvestmentIQ LangGraph Integration Tests")
    logger.info("=" * 60)

    results = []

    # Test 1: Full workflow
    logger.info("\nTest 1: Complete LangGraph Workflow")
    results.append(("Complete Workflow", await test_langgraph_workflow()))

    # Test 2: Graph visualization
    logger.info("\nTest 2: Graph Visualization")
    results.append(("Graph Visualization", await test_graph_visualization()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name:.<40} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return all(p for _, p in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
