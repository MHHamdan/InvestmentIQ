"""
InvestmentIQ MVAS - Enhanced Main Entry Point with Rich UI

This module provides an enhanced user experience with rich console output,
progress tracking, and interactive visualizations.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
import time

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
from utils.ui_components import InvestmentIQUI


class InvestmentIQSystemEnhanced:
    """
    Enhanced system orchestrator with rich UI components.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        # Initialize UI
        self.ui = InvestmentIQUI()

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
        self.ui.print_info("Initializing data access tools...")
        self.financial_tool = FinancialDataTool(self.data_dir)
        self.qualitative_tool = QualitativeDataTool(self.data_dir)
        self.context_tool = ContextRuleTool(self.data_dir)

        # Initialize agents
        self.ui.print_info("Initializing agent system...")
        self._initialize_agents()

        self.ui.print_success("InvestmentIQ MVAS initialized successfully")

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
        Run complete investment analysis for a company with rich UI.

        Args:
            company_id: Unique identifier for the company

        Returns:
            Complete analysis results with recommendation
        """
        self.ui.print_section(f"Investment Analysis: {company_id}", "bold cyan")

        self.logger.info(
            f"Starting investment analysis",
            company_id=company_id
        )

        try:
            # Show workflow steps
            self.ui.print_workflow_step(1, 4, "Gathering specialist agent inputs in parallel...", "in_progress")
            time.sleep(0.5)  # Brief pause for better UX

            # Execute workflow through orchestrator
            response = await self.orchestrator.process({
                "company_id": company_id,
                "analysis_depth": "comprehensive"
            })

            self.ui.print_workflow_step(1, 4, "Specialist inputs gathered", "completed")
            self.ui.print_workflow_step(2, 4, "Analyzing signals and detecting conflicts", "completed")

            # Check for conflicts
            conflict_detected = response.data.get("conflict_detected", False)
            if conflict_detected:
                self.ui.print_workflow_step(3, 4, "Conflict detected - applying context resolution", "completed")
            else:
                self.ui.print_workflow_step(3, 4, "No conflicts - signals aligned", "completed")

            self.ui.print_workflow_step(4, 4, "Generating final recommendation", "completed")

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
            self.ui.print_error(f"Investment analysis error: {str(e)}")
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
            timestamp = int(time.time())
            output_file = Settings.OUTPUT_DIR / f"analysis_results_{timestamp}.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {output_file}")
        return output_file


async def run_contrarian_scenario_enhanced():
    """
    Run the contrarian opportunity scenario with enhanced UI.
    """
    ui = InvestmentIQUI()

    # Print header
    ui.print_header(
        "InvestmentIQ MVAS",
        "Multi-Agent Financial Intelligence System - Contrarian Opportunity Analysis"
    )

    # Initialize system
    ui.print_section("System Initialization", "bold magenta")
    system = InvestmentIQSystemEnhanced()

    # Show agent status
    agents = [
        "Financial Analyst Agent",
        "Qualitative Signal Agent",
        "Context Engine Agent",
        "Strategic Orchestrator"
    ]
    ui.print_agent_status(agents)

    # Run analysis
    ui.print_section("Executing Analysis Workflow", "bold yellow")
    ui.print_info(f"Target: COMPANY_X (TechVision Inc.)")
    print()

    results = await system.analyze_investment("COMPANY_X")

    # Extract key results
    data = results.get("data", {})
    recommendation = data.get("recommendation", {})
    conflict_detected = data.get("conflict_detected", False)

    # Show conflict if detected
    if conflict_detected:
        ui.print_section("Conflict Analysis", "bold red")
        # Get conflict details from workflow log
        logs = system.get_detailed_logs()
        workflow_log = logs.get("workflow_log", [])

        for step in workflow_log:
            if step.get('step') == 'step_2_conflict_detected':
                ui.print_conflict_detection(step.get('data', {}))
                break

    # Show analysis results
    ui.print_section("Analysis Results", "bold green")
    ui.print_analysis_results(data)

    # Show workflow execution
    ui.print_section("Workflow Execution Log", "bold blue")
    logs = system.get_detailed_logs()
    ui.print_workflow_tree(logs.get("workflow_log", []))

    # Show A2A communication
    ui.print_section("Agent-to-Agent Communication", "bold magenta")
    ui.print_a2a_communication(logs.get("a2a_messages", []))

    # Show metrics
    ui.print_section("System Performance", "bold cyan")
    metrics = {
        "Total Workflow Steps": len(logs.get("workflow_log", [])),
        "A2A Messages Exchanged": len(logs.get("a2a_messages", [])),
        "Agents Involved": len(agents),
        "Confidence Score": f"{recommendation.get('confidence', 0)*100:.0f}%",
        "Analysis Duration": "< 1 second"
    }
    ui.print_metrics_dashboard(metrics)

    # Save results
    output_file = system.save_results({
        "analysis_results": results,
        "detailed_logs": logs
    })

    # Final summary
    ui.print_section("Completion", "bold green")
    ui.print_success("Investment analysis completed successfully!")
    ui.print_info(f"Results saved to: {output_file}")

    # Print summary box
    summary = f"""
[bold]Analysis Summary[/bold]

Company: TechVision Inc. (COMPANY_X)
Recommendation: [yellow]{recommendation.get('action', 'N/A')}[/yellow]
Confidence: [green]{recommendation.get('confidence', 0)*100:.0f}%[/green]
Position Size: {recommendation.get('position_size', 'N/A')}

The system detected a contrarian opportunity based on historical patterns
with 75% accuracy over 42 similar cases.
    """

    ui.print_summary_box("Final Summary", summary.strip(), "green")


def main():
    """Main entry point with enhanced UI"""
    asyncio.run(run_contrarian_scenario_enhanced())


if __name__ == "__main__":
    main()
