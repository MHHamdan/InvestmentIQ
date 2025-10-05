"""
Evaluation Runner for InvestmentIQ MVAS

Comprehensive evaluation runner with LangSmith integration.
Tests all agents with custom evaluators, LLM-as-judge, and summary evaluators.
"""

import asyncio
from langsmith.evaluation import evaluate
from evaluation.evaluators import get_evaluators
from evaluation.dataset_creator import create_evaluation_datasets
from main_enhanced import InvestmentIQSystemEnhanced
from utils.ui_components import InvestmentIQUI


class EvaluationRunner:
    """
    Runs comprehensive evaluations for InvestmentIQ MVAS.

    Evaluation Types:
    1. System-level evaluation (full workflow)
    2. Agent-specific evaluations
    3. Pairwise comparisons
    """

    def __init__(self):
        self.ui = InvestmentIQUI()
        self.evaluators = get_evaluators()
        self.system = None

    async def initialize_system(self):
        """Initialize InvestmentIQ system."""
        if not self.system:
            self.ui.print_info("Initializing InvestmentIQ system...")
            self.system = InvestmentIQSystemEnhanced()
            self.ui.print_success("System initialized successfully")

    async def target_function(self, inputs: dict) -> dict:
        """
        Target function for evaluation.

        Args:
            inputs: Dataset inputs (company_id, analysis_type)

        Returns:
            System analysis results
        """
        company_id = inputs.get("company_id", "COMPANY_X")
        return await self.system.analyze_investment(company_id)

    async def run_system_evaluation(self, dataset_name: str = "investment-iq-evaluation"):
        """
        Run complete system evaluation with all evaluators.

        Args:
            dataset_name: LangSmith dataset name
        """
        self.ui.print_section("Running System Evaluation", "bold cyan")

        await self.initialize_system()

        # Get all evaluators
        evaluators = [
            self.evaluators.financial_analysis_structure,
            self.evaluators.sentiment_detection_accuracy,
            self.evaluators.recommendation_consistency,
            self.evaluators.conflict_detection_accuracy,
            self.evaluators.investment_quality_judge
        ]

        # Get summary evaluators
        summary_evaluators = [
            self.evaluators.average_confidence_score,
            self.evaluators.conflict_detection_rate,
            self.evaluators.context_rule_effectiveness
        ]

        self.ui.print_info(f"Running evaluation on dataset: {dataset_name}")
        self.ui.print_info(f"Evaluators: {len(evaluators)} regular + {len(summary_evaluators)} summary")

        try:
            results = evaluate(
                self.target_function,
                data=dataset_name,
                evaluators=evaluators,
                summary_evaluators=summary_evaluators,
                experiment_prefix="investment-iq-system-eval",
                metadata={
                    "version": "1.0",
                    "component": "full_system",
                    "week": "4_enhancement"
                },
                max_concurrency=1  # Sequential for stability
            )

            self.ui.print_success("System evaluation completed!")
            self.ui.print_info("View detailed results in LangSmith UI")

            return results

        except Exception as e:
            self.ui.print_error(f"Evaluation failed: {e}")
            raise

    async def run_agent_evaluation(self, agent_name: str, dataset_name: str):
        """
        Run agent-specific evaluation.

        Args:
            agent_name: Agent to evaluate (financial_analyst, qualitative_signal, etc.)
            dataset_name: Dataset for this agent
        """
        self.ui.print_section(f"Evaluating {agent_name} Agent", "bold yellow")

        await self.initialize_system()

        # Create agent-specific target function
        async def agent_target(inputs: dict) -> dict:
            if agent_name == "financial_analyst":
                response = await self.system.financial_agent.process(inputs)
            elif agent_name == "qualitative_signal":
                response = await self.system.qualitative_agent.process(inputs)
            elif agent_name == "context_engine":
                response = await self.system.context_agent.process(inputs)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

            return response.to_dict()

        # Select appropriate evaluators
        if agent_name == "financial_analyst":
            evaluators = [self.evaluators.financial_analysis_structure]
        elif agent_name == "qualitative_signal":
            evaluators = [self.evaluators.sentiment_detection_accuracy]
        else:
            evaluators = []

        try:
            results = evaluate(
                agent_target,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=f"{agent_name}-eval",
                metadata={"agent": agent_name},
                max_concurrency=1
            )

            self.ui.print_success(f"{agent_name} evaluation completed!")
            return results

        except Exception as e:
            self.ui.print_error(f"Agent evaluation failed: {e}")
            raise

    def display_evaluation_summary(self):
        """Display evaluation summary and next steps."""
        summary = """
[bold cyan]Evaluation Complete![/bold cyan]

[bold]What was evaluated:[/bold]
✓ Financial analysis structure and quality
✓ Sentiment detection accuracy
✓ Recommendation consistency
✓ Conflict detection effectiveness
✓ Overall investment quality (LLM-as-judge)

[bold]Summary metrics computed:[/bold]
✓ Average confidence scores
✓ Conflict detection rates
✓ Context rule effectiveness

[bold]View results:[/bold]
Visit LangSmith UI at: https://smith.langchain.com
Project: investment-iq-production

[bold]Next steps:[/bold]
1. Review evaluation metrics in LangSmith
2. Identify improvement opportunities
3. Run pairwise evaluations for A/B testing
4. Collect user feedback on recommendations
        """

        self.ui.print_summary_box("Evaluation Summary", summary.strip(), "green")


async def main():
    """Main evaluation runner."""
    ui = InvestmentIQUI()

    ui.print_header(
        "InvestmentIQ MVAS - Evaluation Suite",
        "Comprehensive Agent Evaluation"
    )

    # Step 1: Create evaluation datasets
    ui.print_section("Step 1: Creating Evaluation Datasets", "bold magenta")

    try:
        datasets = create_evaluation_datasets()
        ui.print_success(f"Created {len(datasets)} evaluation datasets")
    except Exception as e:
        ui.print_warning(f"Dataset creation note: {e}")
        ui.print_info("Using existing datasets")

    # Step 2: Run system evaluation
    runner = EvaluationRunner()

    ui.print_section("Step 2: Running System Evaluation", "bold cyan")

    try:
        await runner.run_system_evaluation()
    except Exception as e:
        ui.print_error(f"System evaluation error: {e}")
        ui.print_info("Check your LangSmith API key and connection")

    # Step 3: Display summary
    ui.print_section("Step 3: Evaluation Summary", "bold green")
    runner.display_evaluation_summary()


if __name__ == "__main__":
    asyncio.run(main())
