"""
InvestmentIQ MVAS - Main Entry Point

This module serves as the main entry point for the Investment Intelligence
Minimal Viable Agent System (MVAS).
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

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
from utils.logger import setup_logging


class InvestmentIQSystem:
    """
    Main system orchestrator for InvestmentIQ MVAS.

    This class initializes all agents and tools, and manages the
    execution of investment analysis workflows.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        # Ensure required directories exist
        Settings.ensure_directories()

        # Setup logging
        self.logger = setup_logging(
            log_dir=Settings.LOG_DIR,
            log_level=Settings.LOG_LEVEL
        )

        # Set data directory
        self.data_dir = data_dir or Settings.DATA_DIR

        # Initialize tools
        self.logger.info("Initializing data access tools")
        self.financial_tool = FinancialDataTool(self.data_dir)
        self.qualitative_tool = QualitativeDataTool(self.data_dir)
        self.context_tool = ContextRuleTool(self.data_dir)

        # Initialize agents
        self.logger.info("Initializing agent system")
        self._initialize_agents()

        self.logger.info("InvestmentIQ MVAS initialized successfully")

    def _initialize_agents(self) -> None:
        """Initialize all system agents"""
        # Get agent configurations
        financial_config = Settings.get_agent_config("financial_analyst")
        qualitative_config = Settings.get_agent_config("qualitative_signal")
        context_config = Settings.get_agent_config("context_engine")
        orchestrator_config = Settings.get_agent_config("strategic_orchestrator")

        # Create agents
        self.financial_agent = FinancialAnalystAgent(
            agent_id=financial_config["agent_id"],
            data_tool=self.financial_tool
        )

        self.qualitative_agent = QualitativeSignalAgent(
            agent_id=qualitative_config["agent_id"],
            data_tool=self.qualitative_tool
        )

        self.context_agent = ContextEngineAgent(
            agent_id=context_config["agent_id"],
            rule_tool=self.context_tool
        )

        self.orchestrator = StrategicOrchestratorAgent(
            agent_id=orchestrator_config["agent_id"],
            financial_agent=self.financial_agent,
            qualitative_agent=self.qualitative_agent,
            context_agent=self.context_agent
        )

    async def analyze_investment(self, company_id: str) -> dict:
        """
        Run complete investment analysis for a company.

        Args:
            company_id: Unique identifier for the company

        Returns:
            Complete analysis results with recommendation
        """
        self.logger.info(
            f"Starting investment analysis",
            company_id=company_id
        )

        try:
            # Execute workflow through orchestrator
            response = await self.orchestrator.process({
                "company_id": company_id,
                "analysis_depth": "comprehensive"
            })

            # Log completion
            if response.status == "success":
                self.logger.info(
                    "Investment analysis completed successfully",
                    company_id=company_id,
                    recommendation=response.data["recommendation"]["action"]
                )
            else:
                self.logger.error(
                    "Investment analysis failed",
                    company_id=company_id,
                    error=response.data
                )

            return response.to_dict()

        except Exception as e:
            self.logger.error(
                f"Investment analysis error: {str(e)}",
                company_id=company_id
            )
            raise

    def get_detailed_logs(self) -> dict:
        """Get detailed execution logs"""
        return {
            "workflow_log": self.orchestrator.get_detailed_workflow_log(),
            "a2a_messages": self.orchestrator.get_a2a_message_history()
        }

    def save_results(
        self,
        results: dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save analysis results to file.

        Args:
            results: Analysis results to save
            output_file: Optional output file path

        Returns:
            Path to saved file
        """
        if not output_file:
            timestamp = asyncio.get_event_loop().time()
            output_file = Settings.OUTPUT_DIR / f"analysis_results_{timestamp}.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {output_file}")
        return output_file


async def run_contrarian_scenario():
    """
    Run the contrarian opportunity scenario.

    This demonstrates the complete workflow for detecting and resolving
    conflicts between strong financial data and negative sentiment.
    """
    print("\n" + "="*80)
    print("InvestmentIQ MVAS - Contrarian Opportunity Analysis")
    print("="*80 + "\n")

    # Initialize system
    system = InvestmentIQSystem()

    print("Initializing 4-Agent Multi-Agent System...")
    print("- Financial Analyst Agent: Ready")
    print("- Qualitative Signal Agent: Ready")
    print("- Context Engine Agent: Ready")
    print("- Strategic Orchestrator: Ready")
    print("\n" + "-"*80 + "\n")

    # Run analysis
    print("Executing Investment Analysis Workflow for COMPANY_X (TechVision Inc.)")
    print("\nStep 1: Gathering specialist agent inputs in parallel...")

    results = await system.analyze_investment("COMPANY_X")

    print("\nStep 2: Analyzing financial and qualitative signals...")
    print("\nStep 3: Conflict detection and resolution...")
    print("\nStep 4: Generating final recommendation...")

    print("\n" + "-"*80)
    print("\nANALYSIS RESULTS")
    print("-"*80 + "\n")

    # Extract key results
    data = results.get("data", {})
    recommendation = data.get("recommendation", {})
    conflict_detected = data.get("conflict_detected", False)

    print(f"Company: TechVision Inc. (COMPANY_X)")
    print(f"Conflict Detected: {'YES' if conflict_detected else 'NO'}")

    if conflict_detected:
        print(f"\nConflict Type: Strong Financials vs. Negative Sentiment")

    print(f"\nFINAL RECOMMENDATION:")
    print(f"  Action: {recommendation.get('action', 'N/A')}")
    print(f"  Position Size: {recommendation.get('position_size', 'N/A')}")
    print(f"  Confidence: {recommendation.get('confidence', 0)*100:.0f}%")
    print(f"\n  Reasoning: {recommendation.get('reasoning', 'N/A')}")

    supporting = recommendation.get('supporting_factors', {})
    if supporting:
        print(f"\n  Supporting Factors:")
        print(f"    - Financial Health: {supporting.get('financial_health', 'N/A')}")
        print(f"    - Sentiment: {supporting.get('sentiment', 'N/A')}")
        print(f"    - Risk Level: {supporting.get('risk_level', 'N/A')}")

    # Show context rule applied
    context_rule = recommendation.get('context_rule_applied', {})
    if context_rule:
        print(f"\n  Context Rule Applied:")
        print(f"    - Rule ID: {context_rule.get('rule_id', 'N/A')}")
        print(f"    - Historical Accuracy: {context_rule.get('historical_accuracy', 0)*100:.0f}%")
        print(f"    - Description: {context_rule.get('description', 'N/A')}")

    print("\n" + "-"*80)
    print("\nWORKFLOW EXECUTION LOG")
    print("-"*80 + "\n")

    # Get detailed logs
    logs = system.get_detailed_logs()
    workflow_log = logs.get("workflow_log", [])

    for i, step in enumerate(workflow_log, 1):
        print(f"{i}. {step.get('step', 'Unknown')}")
        print(f"   Timestamp: {step.get('timestamp', 'N/A')}")

    print("\n" + "-"*80)
    print("\nA2A COMMUNICATION LOG")
    print("-"*80 + "\n")

    a2a_messages = logs.get("a2a_messages", [])
    for i, msg in enumerate(a2a_messages, 1):
        print(f"{i}. {msg.get('sender')} -> {msg.get('receiver')}")
        print(f"   Type: {msg.get('message_type')}")
        print(f"   Timestamp: {msg.get('timestamp')}")

    # Save results
    output_file = system.save_results({
        "analysis_results": results,
        "detailed_logs": logs
    })

    print("\n" + "="*80)
    print(f"\nComplete results saved to: {output_file}")
    print("\nInvestmentIQ MVAS Analysis Complete!")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    asyncio.run(run_contrarian_scenario())


if __name__ == "__main__":
    main()
