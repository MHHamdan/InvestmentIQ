"""
Seed RAG Context Database with Historical Investment Cases

Populate Pinecone with historical investment scenarios for RAG-based context retrieval.
"""

from tools.rag_context_tool import get_rag_context_tool
from utils.ui_components import InvestmentIQUI


def get_historical_investment_cases():
    """
    Get historical investment cases for RAG indexing.

    In production, these would come from:
    - Historical investment database
    - Case study repositories
    - Investment outcome tracking systems
    """

    cases = [
        {
            "id": "case_001_contrarian_tech",
            "scenario": "Strong financial technology company with negative market sentiment due to leadership crisis",
            "outcome": "Success - Stock recovered 45% within 6 months after leadership stabilized",
            "recommendation": "Strong BUY",
            "accuracy": 0.85,
            "metadata": {
                "sector": "technology",
                "financial_health": "Strong",
                "sentiment": "Negative",
                "conflict_type": "contrarian_opportunity",
                "timeframe": "6_months",
                "roi": 0.45
            }
        },
        {
            "id": "case_002_contrarian_pharma",
            "scenario": "Pharmaceutical company with solid balance sheet but FDA approval delays causing negative press",
            "outcome": "Success - Gained 32% after FDA approval granted",
            "recommendation": "BUY",
            "accuracy": 0.78,
            "metadata": {
                "sector": "healthcare",
                "financial_health": "Strong",
                "sentiment": "Negative",
                "conflict_type": "contrarian_opportunity",
                "timeframe": "9_months",
                "roi": 0.32
            }
        },
        {
            "id": "case_003_aligned_growth",
            "scenario": "High-growth SaaS company with strong financials and positive market sentiment",
            "outcome": "Success - Continued growth of 28% annually",
            "recommendation": "BUY",
            "accuracy": 0.92,
            "metadata": {
                "sector": "technology",
                "financial_health": "Strong",
                "sentiment": "Positive",
                "conflict_type": "none",
                "timeframe": "12_months",
                "roi": 0.28
            }
        },
        {
            "id": "case_004_failed_turnaround",
            "scenario": "Retail company with weak financials and negative sentiment attempting turnaround",
            "outcome": "Failure - Continued decline, -22% over 12 months",
            "recommendation": "SELL",
            "accuracy": 0.88,
            "metadata": {
                "sector": "retail",
                "financial_health": "Weak",
                "sentiment": "Negative",
                "conflict_type": "none",
                "timeframe": "12_months",
                "roi": -0.22
            }
        },
        {
            "id": "case_005_contrarian_manufacturing",
            "scenario": "Manufacturing company with strong margins but supply chain concerns creating negative sentiment",
            "outcome": "Success - Resolved supply issues, +38% in 8 months",
            "recommendation": "BUY",
            "accuracy": 0.81,
            "metadata": {
                "sector": "manufacturing",
                "financial_health": "Strong",
                "sentiment": "Negative",
                "conflict_type": "contrarian_opportunity",
                "timeframe": "8_months",
                "roi": 0.38
            }
        },
        {
            "id": "case_006_false_positive",
            "scenario": "Tech startup with weak financials but strong marketing creating positive sentiment",
            "outcome": "Failure - Burned through cash, -45% before restructuring",
            "recommendation": "HOLD",
            "accuracy": 0.73,
            "metadata": {
                "sector": "technology",
                "financial_health": "Weak",
                "sentiment": "Positive",
                "conflict_type": "sentiment_financial_mismatch",
                "timeframe": "6_months",
                "roi": -0.45
            }
        },
        {
            "id": "case_007_contrarian_energy",
            "scenario": "Energy company with strong reserves and cash flow but negative environmental sentiment",
            "outcome": "Success - Transitioned to renewables, +29% in 18 months",
            "recommendation": "BUY",
            "accuracy": 0.76,
            "metadata": {
                "sector": "energy",
                "financial_health": "Strong",
                "sentiment": "Negative",
                "conflict_type": "contrarian_opportunity",
                "timeframe": "18_months",
                "roi": 0.29
            }
        },
        {
            "id": "case_008_stable_dividend",
            "scenario": "Utility company with consistent financials and neutral-to-positive sentiment",
            "outcome": "Success - Stable returns of 12% with dividends",
            "recommendation": "BUY",
            "accuracy": 0.95,
            "metadata": {
                "sector": "utilities",
                "financial_health": "Strong",
                "sentiment": "Positive",
                "conflict_type": "none",
                "timeframe": "24_months",
                "roi": 0.12
            }
        },
        {
            "id": "case_009_contrarian_automotive",
            "scenario": "Automotive manufacturer with strong EV technology but legacy industry negative sentiment",
            "outcome": "Success - EV line succeeded, +52% in 12 months",
            "recommendation": "Strong BUY",
            "accuracy": 0.87,
            "metadata": {
                "sector": "automotive",
                "financial_health": "Strong",
                "sentiment": "Negative",
                "conflict_type": "contrarian_opportunity",
                "timeframe": "12_months",
                "roi": 0.52
            }
        },
        {
            "id": "case_010_risky_biotech",
            "scenario": "Biotech startup with weak financials and negative sentiment, high risk drug pipeline",
            "outcome": "Failure - Clinical trials failed, -67% valuation loss",
            "recommendation": "SELL",
            "accuracy": 0.91,
            "metadata": {
                "sector": "biotechnology",
                "financial_health": "Weak",
                "sentiment": "Negative",
                "conflict_type": "none",
                "timeframe": "6_months",
                "roi": -0.67
            }
        }
    ]

    return cases


def main():
    """Seed RAG context database with historical cases."""
    ui = InvestmentIQUI()

    ui.print_header(
        "RAG Context Database Seeding",
        "Populating Pinecone with Historical Investment Cases"
    )

    ui.print_section("Step 1: Initializing RAG Context Tool", "bold cyan")

    try:
        rag_tool = get_rag_context_tool()

        # Check if RAG is properly configured
        stats = rag_tool.get_stats()

        if stats.get("status") == "disabled":
            ui.print_error("RAG Context Tool is not configured")
            ui.print_info("Please ensure PINECONE_API_KEY is set in .env file")
            return

        ui.print_success("RAG Context Tool initialized successfully")
        ui.print_info(f"Index: {stats.get('index_name', 'N/A')}")
        ui.print_info(f"Dimension: {stats.get('dimension', 'N/A')}")

    except Exception as e:
        ui.print_error(f"Failed to initialize RAG tool: {e}")
        return

    ui.print_section("Step 2: Loading Historical Investment Cases", "bold yellow")

    cases = get_historical_investment_cases()
    ui.print_success(f"Loaded {len(cases)} historical investment cases")

    # Display sample cases
    ui.print_info("Sample cases:")
    for case in cases[:3]:
        print(f"  • {case['id']}: {case['scenario'][:80]}...")

    ui.print_section("Step 3: Indexing Cases into Pinecone", "bold magenta")

    try:
        indexed_count = rag_tool.index_historical_cases(cases)

        if indexed_count > 0:
            ui.print_success(f"✓ Indexed {indexed_count} cases successfully!")

            # Get updated stats
            stats = rag_tool.get_stats()
            ui.print_info(f"Total vectors in index: {stats.get('total_vectors', 'N/A')}")

        else:
            ui.print_warning("No cases were indexed. Check configuration.")

    except Exception as e:
        ui.print_error(f"Failed to index cases: {e}")
        return

    ui.print_section("Step 4: Testing RAG Retrieval", "bold green")

    # Test retrieval with a sample scenario
    test_scenario = "Technology company with strong financials but negative sentiment due to management issues"

    ui.print_info(f"Test query: {test_scenario}")

    try:
        similar_cases = rag_tool.retrieve_similar_cases(test_scenario, top_k=3)

        if similar_cases:
            ui.print_success(f"Found {len(similar_cases)} similar cases:")

            for i, case in enumerate(similar_cases, 1):
                print(f"\n{i}. {case['case_id']} (similarity: {case['similarity_score']:.3f})")
                print(f"   Outcome: {case['outcome']}")
                print(f"   Recommendation: {case['recommendation']}")
                print(f"   Accuracy: {case['accuracy']:.2f}")

        else:
            ui.print_warning("No similar cases found. Try adjusting similarity threshold.")

    except Exception as e:
        ui.print_error(f"Failed to retrieve cases: {e}")
        return

    # Final summary
    ui.print_section("Summary", "bold cyan")

    summary = f"""
[bold green]✓ RAG Context Database Seeding Complete![/bold green]

[bold]Statistics:[/bold]
• Total cases indexed: {indexed_count}
• Vector dimension: {stats.get('dimension', 'N/A')}
• Index name: {stats.get('index_name', 'N/A')}

[bold]Coverage:[/bold]
• Sectors: Technology, Healthcare, Manufacturing, Retail, Energy, Utilities, Automotive, Biotech
• Scenario types: Contrarian opportunities, Aligned signals, High risk
• Historical accuracy: 73% - 95%

[bold]Next steps:[/bold]
1. RAG-enhanced context engine is now available
2. Run investment analysis to see historical precedents
3. Monitor RAG retrieval quality in evaluations
4. Add more historical cases as they become available
    """

    ui.print_summary_box("RAG Database Ready", summary.strip(), "green")


if __name__ == "__main__":
    main()
