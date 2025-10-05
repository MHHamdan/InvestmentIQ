"""
Unit tests for InvestmentIQ MVAS agents

Tests cover all agent functionality, A2A communication,
and workflow execution.
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.financial_analyst import FinancialAnalystAgent
from agents.qualitative_signal import QualitativeSignalAgent
from agents.context_engine import ContextEngineAgent
from agents.strategic_orchestrator import StrategicOrchestratorAgent
from tools.data_tools import (
    FinancialDataTool,
    QualitativeDataTool,
    ContextRuleTool
)
from config.settings import Settings


@pytest.fixture
def data_dir():
    """Fixture for test data directory"""
    return Settings.DATA_DIR


@pytest.fixture
def financial_tool(data_dir):
    """Fixture for financial data tool"""
    return FinancialDataTool(data_dir)


@pytest.fixture
def qualitative_tool(data_dir):
    """Fixture for qualitative data tool"""
    return QualitativeDataTool(data_dir)


@pytest.fixture
def context_tool(data_dir):
    """Fixture for context rule tool"""
    return ContextRuleTool(data_dir)


@pytest.fixture
def financial_agent(financial_tool):
    """Fixture for financial analyst agent"""
    return FinancialAnalystAgent(
        agent_id="test_financial_agent",
        data_tool=financial_tool
    )


@pytest.fixture
def qualitative_agent(qualitative_tool):
    """Fixture for qualitative signal agent"""
    return QualitativeSignalAgent(
        agent_id="test_qualitative_agent",
        data_tool=qualitative_tool
    )


@pytest.fixture
def context_agent(context_tool):
    """Fixture for context engine agent"""
    return ContextEngineAgent(
        agent_id="test_context_agent",
        rule_tool=context_tool
    )


@pytest.fixture
def orchestrator(financial_agent, qualitative_agent, context_agent):
    """Fixture for strategic orchestrator"""
    return StrategicOrchestratorAgent(
        agent_id="test_orchestrator",
        financial_agent=financial_agent,
        qualitative_agent=qualitative_agent,
        context_agent=context_agent
    )


class TestFinancialDataTool:
    """Test suite for Financial Data Tool"""

    @pytest.mark.asyncio
    async def test_read_financial_data(self, financial_tool):
        """Test reading financial data"""
        data = await financial_tool.read_financial_data("COMPANY_X")

        assert "company_id" in data
        assert data["company_id"] == "COMPANY_X"
        assert "revenue" in data
        assert data["revenue"] > 0

    @pytest.mark.asyncio
    async def test_get_financial_ratios(self, financial_tool):
        """Test financial ratio calculation"""
        result = await financial_tool.get_financial_ratios("COMPANY_X")

        assert "ratios" in result
        assert "gross_margin" in result["ratios"]
        assert result["ratios"]["gross_margin"] > 0

    @pytest.mark.asyncio
    async def test_invalid_company(self, financial_tool):
        """Test handling of invalid company ID"""
        result = await financial_tool.read_financial_data("INVALID_COMPANY")

        assert "error" in result


class TestQualitativeDataTool:
    """Test suite for Qualitative Data Tool"""

    @pytest.mark.asyncio
    async def test_process_unstructured_text(self, qualitative_tool):
        """Test unstructured text processing"""
        result = await qualitative_tool.process_unstructured_text("COMPANY_X")

        assert "sentiment_score" in result
        assert "sentiment_label" in result
        assert "key_themes" in result
        assert isinstance(result["key_themes"], list)

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, qualitative_tool):
        """Test sentiment scoring"""
        result = await qualitative_tool.process_unstructured_text("COMPANY_X")
        sentiment_score = result["sentiment_score"]

        assert -1.0 <= sentiment_score <= 1.0


class TestContextRuleTool:
    """Test suite for Context Rule Tool"""

    @pytest.mark.asyncio
    async def test_get_context_rule(self, context_tool):
        """Test context rule retrieval"""
        rule = await context_tool.get_context_rule(
            "contrarian_opportunity",
            "technology"
        )

        assert "rule_id" in rule
        assert "scenario_type" in rule
        assert rule["scenario_type"] == "contrarian_opportunity"

    @pytest.mark.asyncio
    async def test_invalid_scenario(self, context_tool):
        """Test handling of invalid scenario type"""
        rule = await context_tool.get_context_rule("invalid_scenario")

        assert "error" in rule


class TestFinancialAnalystAgent:
    """Test suite for Financial Analyst Agent"""

    @pytest.mark.asyncio
    async def test_process_request(self, financial_agent):
        """Test financial analysis processing"""
        response = await financial_agent.process({
            "company_id": "COMPANY_X",
            "analysis_type": "comprehensive"
        })

        assert response.status == "success"
        assert "financial_health" in response.data
        assert "key_metrics" in response.data

    @pytest.mark.asyncio
    async def test_missing_company_id(self, financial_agent):
        """Test error handling for missing company ID"""
        response = await financial_agent.process({})

        assert response.status == "error"

    @pytest.mark.asyncio
    async def test_financial_health_assessment(self, financial_agent):
        """Test financial health assessment logic"""
        response = await financial_agent.process({
            "company_id": "COMPANY_X"
        })

        health = response.data["financial_health"]
        assert health in ["Strong", "Moderate", "Weak"]


class TestQualitativeSignalAgent:
    """Test suite for Qualitative Signal Agent"""

    @pytest.mark.asyncio
    async def test_process_request(self, qualitative_agent):
        """Test qualitative analysis processing"""
        response = await qualitative_agent.process({
            "company_id": "COMPANY_X"
        })

        assert response.status == "success"
        assert "overall_sentiment" in response.data
        assert "risk_assessment" in response.data

    @pytest.mark.asyncio
    async def test_risk_assessment(self, qualitative_agent):
        """Test risk assessment logic"""
        response = await qualitative_agent.process({
            "company_id": "COMPANY_X"
        })

        risk = response.data["risk_assessment"]
        assert "risk_level" in risk
        assert risk["risk_level"] in ["low", "medium", "high"]


class TestContextEngineAgent:
    """Test suite for Context Engine Agent"""

    @pytest.mark.asyncio
    async def test_process_request(self, context_agent):
        """Test context rule application"""
        response = await context_agent.process({
            "scenario_type": "contrarian_opportunity",
            "context": {
                "financial_health": "Strong",
                "sentiment": "Very Negative",
                "sector": "technology",
                "gross_margin": 50
            }
        })

        assert response.status == "success"
        assert "recommendation" in response.data
        assert "confidence" in response.data

    @pytest.mark.asyncio
    async def test_condition_matching(self, context_agent):
        """Test condition matching logic"""
        response = await context_agent.process({
            "scenario_type": "contrarian_opportunity",
            "context": {
                "financial_health": "Strong",
                "sentiment": "Very Negative",
                "gross_margin": 50
            }
        })

        assert response.data["confidence"] > 0


class TestStrategicOrchestratorAgent:
    """Test suite for Strategic Orchestrator Agent"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, orchestrator):
        """Test complete workflow execution"""
        response = await orchestrator.process({
            "company_id": "COMPANY_X"
        })

        assert response.status == "success"
        assert "recommendation" in response.data
        assert "conflict_detected" in response.data

    @pytest.mark.asyncio
    async def test_conflict_detection(self, orchestrator):
        """Test conflict detection between agents"""
        response = await orchestrator.process({
            "company_id": "COMPANY_X"
        })

        conflict_detected = response.data["conflict_detected"]
        assert isinstance(conflict_detected, bool)

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, orchestrator):
        """Test final recommendation generation"""
        response = await orchestrator.process({
            "company_id": "COMPANY_X"
        })

        recommendation = response.data["recommendation"]
        assert "action" in recommendation
        assert "confidence" in recommendation
        assert recommendation["action"] in [
            "BUY", "SELL", "HOLD", "ACCUMULATE", "REDUCE"
        ]

    @pytest.mark.asyncio
    async def test_a2a_communication_logging(self, orchestrator):
        """Test A2A communication logging"""
        await orchestrator.process({"company_id": "COMPANY_X"})

        messages = orchestrator.get_a2a_message_history()
        assert len(messages) > 0
        assert all("sender" in msg for msg in messages)
        assert all("receiver" in msg for msg in messages)

    @pytest.mark.asyncio
    async def test_workflow_logging(self, orchestrator):
        """Test workflow step logging"""
        await orchestrator.process({"company_id": "COMPANY_X"})

        workflow_log = orchestrator.get_detailed_workflow_log()
        assert len(workflow_log) > 0
        assert all("step" in log for log in workflow_log)


class TestEndToEndWorkflow:
    """Integration tests for complete system"""

    @pytest.mark.asyncio
    async def test_contrarian_scenario(self, orchestrator):
        """Test complete contrarian opportunity scenario"""
        response = await orchestrator.process({
            "company_id": "COMPANY_X"
        })

        # Should detect conflict
        assert response.data["conflict_detected"] is True

        # Should recommend accumulate
        recommendation = response.data["recommendation"]
        assert recommendation["action"] == "ACCUMULATE"

        # Should have high confidence
        assert recommendation["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_workflow_completeness(self, orchestrator):
        """Test that workflow completes all steps"""
        response = await orchestrator.process({
            "company_id": "COMPANY_X"
        })

        workflow_log = orchestrator.get_detailed_workflow_log()

        # Check for key workflow steps
        step_names = [log["step"] for log in workflow_log]
        assert "workflow_initiated" in step_names
        assert "workflow_completed" in step_names
