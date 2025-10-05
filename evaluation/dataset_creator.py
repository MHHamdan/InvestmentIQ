"""
Evaluation Dataset Creator for InvestmentIQ MVAS

Advanced Feature: Creates and manages evaluation datasets for testing
multi-agent system performance.
"""

from typing import List, Dict, Any
from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()


class EvaluationDatasetCreator:
    """
    Creates evaluation datasets for InvestmentIQ MVAS testing.

    Datasets include:
    - Test scenarios (company IDs, analysis types)
    - Expected outputs (financial health, sentiment, recommendations)
    - Metadata (complexity, domain, regulatory requirements)
    """

    def __init__(self):
        self.client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        )

    def create_investment_analysis_dataset(
        self,
        dataset_name: str = "investment-iq-evaluation"
    ) -> str:
        """
        Create evaluation dataset for investment analysis.

        Returns:
            Dataset ID
        """

        # Evaluation examples covering different scenarios
        evaluation_examples = [
            {
                "inputs": {
                    "company_id": "COMPANY_X",
                    "analysis_type": "comprehensive"
                },
                "outputs": {
                    "expected_conflict": True,
                    "expected_action": "Strong BUY",
                    "expected_financial_health": "Strong",
                    "expected_sentiment": "Negative"
                },
                "metadata": {
                    "scenario": "contrarian_opportunity",
                    "complexity": "high",
                    "conflict_type": "financial_sentiment_mismatch"
                }
            },
            {
                "inputs": {
                    "company_id": "STABLE_TECH_CO",
                    "analysis_type": "comprehensive"
                },
                "outputs": {
                    "expected_conflict": False,
                    "expected_action": "BUY",
                    "expected_financial_health": "Strong",
                    "expected_sentiment": "Positive"
                },
                "metadata": {
                    "scenario": "aligned_signals",
                    "complexity": "medium",
                    "conflict_type": "none"
                }
            },
            {
                "inputs": {
                    "company_id": "RISKY_STARTUP_INC",
                    "analysis_type": "comprehensive"
                },
                "outputs": {
                    "expected_conflict": False,
                    "expected_action": "HOLD",
                    "expected_financial_health": "Weak",
                    "expected_sentiment": "Negative"
                },
                "metadata": {
                    "scenario": "high_risk",
                    "complexity": "medium",
                    "conflict_type": "none"
                }
            }
        ]

        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Investment analysis evaluation dataset for InvestmentIQ MVAS"
            )

            # Add examples
            self.client.create_examples(
                dataset_id=dataset.id,
                examples=evaluation_examples
            )

            print(f"✓ Dataset '{dataset_name}' created successfully!")
            print(f"  Dataset ID: {dataset.id}")
            print(f"  Examples: {len(evaluation_examples)}")

            return dataset.id

        except Exception as e:
            if "already exists" in str(e).lower():
                dataset = self.client.read_dataset(dataset_name=dataset_name)
                print(f"✓ Using existing dataset: {dataset.id}")
                return dataset.id
            else:
                raise Exception(f"Failed to create dataset: {e}")

    def create_agent_specific_datasets(self) -> Dict[str, str]:
        """
        Create agent-specific evaluation datasets.

        Returns:
            Dictionary mapping agent name to dataset ID
        """

        datasets = {}

        # Financial Analyst Dataset
        financial_examples = [
            {
                "inputs": {
                    "company_id": "COMPANY_X",
                    "analysis_type": "comprehensive"
                },
                "outputs": {
                    "expected_health": "Strong",
                    "expected_ratios": ["gross_margin", "debt_to_equity", "current_ratio"]
                },
                "metadata": {
                    "domain": "financial_analysis"
                }
            }
        ]

        try:
            dataset = self.client.create_dataset(
                dataset_name="financial-analyst-eval",
                description="Financial Analyst Agent evaluation dataset"
            )
            self.client.create_examples(dataset_id=dataset.id, examples=financial_examples)
            datasets["financial_analyst"] = dataset.id
            print(f"✓ Financial Analyst dataset created: {dataset.id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                dataset = self.client.read_dataset(dataset_name="financial-analyst-eval")
                datasets["financial_analyst"] = dataset.id
                print(f"✓ Using existing Financial Analyst dataset: {dataset.id}")

        # Qualitative Signal Dataset
        qualitative_examples = [
            {
                "inputs": {
                    "company_id": "COMPANY_X"
                },
                "outputs": {
                    "expected_sentiment": "Negative",
                    "expected_risk": "Medium",
                    "expected_themes": ["leadership_crisis", "employee_dissatisfaction"]
                },
                "metadata": {
                    "domain": "qualitative_analysis"
                }
            }
        ]

        try:
            dataset = self.client.create_dataset(
                dataset_name="qualitative-signal-eval",
                description="Qualitative Signal Agent evaluation dataset"
            )
            self.client.create_examples(dataset_id=dataset.id, examples=qualitative_examples)
            datasets["qualitative_signal"] = dataset.id
            print(f"✓ Qualitative Signal dataset created: {dataset.id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                dataset = self.client.read_dataset(dataset_name="qualitative-signal-eval")
                datasets["qualitative_signal"] = dataset.id
                print(f"✓ Using existing Qualitative Signal dataset: {dataset.id}")

        return datasets


def create_evaluation_datasets():
    """Convenience function to create all evaluation datasets."""
    creator = EvaluationDatasetCreator()

    print("Creating InvestmentIQ evaluation datasets...")
    print("=" * 80)

    # Create main dataset
    main_dataset_id = creator.create_investment_analysis_dataset()

    # Create agent-specific datasets
    agent_datasets = creator.create_agent_specific_datasets()

    print("=" * 80)
    print(f"✓ All datasets created successfully!")
    print(f"  Main dataset: {main_dataset_id}")
    print(f"  Agent datasets: {len(agent_datasets)}")

    return {
        "main": main_dataset_id,
        **agent_datasets
    }


if __name__ == "__main__":
    create_evaluation_datasets()
