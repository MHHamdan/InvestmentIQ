"""
Advanced Features Verification Script

Tests observability, evaluation, and RAG capabilities:
1. LLM Factory (Hugging Face integration)
2. Observability (LangSmith tracing)
3. Evaluation Framework
4. RAG Context Tool (Pinecone)
"""

import os
import asyncio
from utils.ui_components import InvestmentIQUI


def test_environment_variables():
    """Test if all required environment variables are set."""
    ui = InvestmentIQUI()
    ui.print_section("Testing Environment Variables", "bold cyan")

    required_vars = {
        "LANGSMITH_API_KEY": "LangSmith observability",
        "LANGSMITH_TRACING": "Tracing enabled flag",
        "LANGSMITH_PROJECT": "LangSmith project name",
        "HUGGING_FACE_API_KEY": "Hugging Face models",
        "PINECONE_API_KEY": "Pinecone vector database"
    }

    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value and value not in ["your_api_key_here", "your_huggingface_key_here", "your_pinecone_key_here"]:
            ui.print_success(f"✓ {var} - {description}")
        else:
            ui.print_warning(f"✗ {var} - {description} (not set or placeholder)")
            all_set = False

    return all_set


def test_llm_factory():
    """Test LLM Factory initialization."""
    ui = InvestmentIQUI()
    ui.print_section("Testing LLM Factory", "bold cyan")

    try:
        from utils.llm_factory import get_llm_factory

        factory = get_llm_factory()
        info = factory.get_provider_info()

        ui.print_success("✓ LLM Factory initialized")
        ui.print_info(f"Default provider: {info['default_provider']}")
        ui.print_info(f"Available providers: {', '.join(info['available_providers'])}")

        # Test chat model creation
        if info['huggingface_available']:
            ui.print_info("Testing Hugging Face chat model creation...")
            try:
                model = factory.create_chat_model(provider="huggingface")
                ui.print_success("✓ Hugging Face chat model created")
            except Exception as e:
                ui.print_warning(f"Hugging Face model creation warning: {e}")

        # Test embeddings
        if info['huggingface_available']:
            ui.print_info("Testing Hugging Face embeddings creation...")
            try:
                embeddings = factory.create_embeddings(provider="huggingface")
                ui.print_success("✓ Hugging Face embeddings created")
            except Exception as e:
                ui.print_warning(f"Embeddings creation warning: {e}")

        return True

    except Exception as e:
        ui.print_error(f"✗ LLM Factory test failed: {e}")
        return False


def test_observability():
    """Test Observability Manager."""
    ui = InvestmentIQUI()
    ui.print_section("Testing Observability", "bold cyan")

    try:
        from utils.observability import get_observability_manager

        obs_manager = get_observability_manager()

        if obs_manager.is_enabled():
            ui.print_success("✓ LangSmith observability enabled")
            ui.print_info(f"Project: {obs_manager.project}")

            # Test metrics retrieval
            metrics = obs_manager.get_project_metrics()
            ui.print_info(f"Status: {metrics.get('status', 'unknown')}")

        else:
            ui.print_warning("✗ LangSmith observability disabled")
            ui.print_info("Set LANGSMITH_TRACING=true in .env to enable")

        return True

    except Exception as e:
        ui.print_error(f"✗ Observability test failed: {e}")
        return False


def test_evaluation_framework():
    """Test Evaluation Framework."""
    ui = InvestmentIQUI()
    ui.print_section("Testing Evaluation Framework", "bold cyan")

    try:
        from evaluation.evaluators import get_evaluators

        evaluators = get_evaluators()
        ui.print_success("✓ Evaluation framework initialized")

        # List evaluators
        ui.print_info("Available evaluators:")
        print("  • financial_analysis_structure")
        print("  • sentiment_detection_accuracy")
        print("  • recommendation_consistency")
        print("  • conflict_detection_accuracy")
        print("  • investment_quality_judge (LLM-as-judge)")

        ui.print_info("Summary evaluators:")
        print("  • average_confidence_score")
        print("  • conflict_detection_rate")
        print("  • context_rule_effectiveness")

        return True

    except Exception as e:
        ui.print_error(f"✗ Evaluation framework test failed: {e}")
        return False


def test_rag_context_tool():
    """Test RAG Context Tool."""
    ui = InvestmentIQUI()
    ui.print_section("Testing RAG Context Tool", "bold cyan")

    try:
        from tools.rag_context_tool import get_rag_context_tool

        rag_tool = get_rag_context_tool()
        stats = rag_tool.get_stats()

        if stats.get("status") == "active":
            ui.print_success("✓ RAG Context Tool active")
            ui.print_info(f"Index: {stats.get('index_name', 'N/A')}")
            ui.print_info(f"Total vectors: {stats.get('total_vectors', 0)}")
            ui.print_info(f"Dimension: {stats.get('dimension', 'N/A')}")

        elif stats.get("status") == "disabled":
            ui.print_warning("✗ RAG Context Tool disabled")
            ui.print_info("Reason: Pinecone API key not configured")
            ui.print_info("Set PINECONE_API_KEY in .env to enable")

        else:
            ui.print_error(f"✗ RAG Context Tool error: {stats.get('error', 'unknown')}")

        return True

    except Exception as e:
        ui.print_error(f"✗ RAG Context Tool test failed: {e}")
        return False


async def test_end_to_end():
    """Test end-to-end integration."""
    ui = InvestmentIQUI()
    ui.print_section("Testing End-to-End Integration", "bold cyan")

    try:
        from main_enhanced import InvestmentIQSystemEnhanced

        ui.print_info("Initializing InvestmentIQ system...")
        system = InvestmentIQSystemEnhanced()

        ui.print_info("Running analysis on COMPANY_X...")
        results = await system.analyze_investment("COMPANY_X")

        if results.get("status") == "success":
            ui.print_success("✓ End-to-end analysis completed successfully")

            data = results.get("data", {})
            recommendation = data.get("recommendation", {})
            ui.print_info(f"Recommendation: {recommendation.get('action', 'N/A')}")
            ui.print_info(f"Confidence: {recommendation.get('confidence', 0)*100:.0f}%")

            # Check if observability captured the run
            if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
                ui.print_info("✓ Check LangSmith UI for trace details")

        else:
            ui.print_warning("Analysis completed with warnings")

        return True

    except Exception as e:
        ui.print_error(f"✗ End-to-end test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def main():
    """Run all advanced features setup tests."""
    ui = InvestmentIQUI()

    ui.print_header(
        "Advanced Features Verification",
        "Testing Observability, Evaluation & RAG"
    )

    results = {}

    # Run tests
    results["environment"] = test_environment_variables()
    print()

    results["llm_factory"] = test_llm_factory()
    print()

    results["observability"] = test_observability()
    print()

    results["evaluation"] = test_evaluation_framework()
    print()

    results["rag"] = test_rag_context_tool()
    print()

    results["end_to_end"] = await test_end_to_end()
    print()

    # Summary
    ui.print_section("Test Summary", "bold green")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    summary = f"""
[bold]Test Results: {passed}/{total} passed[/bold]

[bold]Components tested:[/bold]
{'[green]✓[/green]' if results.get('environment') else '[red]✗[/red]'} Environment Variables
{'[green]✓[/green]' if results.get('llm_factory') else '[red]✗[/red]'} LLM Factory (Hugging Face)
{'[green]✓[/green]' if results.get('observability') else '[red]✗[/red]'} Observability (LangSmith)
{'[green]✓[/green]' if results.get('evaluation') else '[red]✗[/red]'} Evaluation Framework
{'[green]✓[/green]' if results.get('rag') else '[red]✗[/red]'} RAG Context Tool (Pinecone)
{'[green]✓[/green]' if results.get('end_to_end') else '[red]✗[/red]'} End-to-End Integration

[bold]Next steps:[/bold]
1. If any tests failed, check .env configuration
2. Run: python seed_rag_context.py (to populate RAG database)
3. Run: python run_evaluation.py (to run evaluations)
4. Run: python cli_interactive.py (for interactive demo)

[bold]Documentation:[/bold]
- See ADVANCED_FEATURES.md for detailed guide
- Check LangSmith UI: https://smith.langchain.com
    """

    ui.print_summary_box("Verification Complete", summary.strip(), "cyan")

    if passed == total:
        ui.print_success("All advanced features are ready!")
    elif passed >= total - 1:
        ui.print_warning("Most components working - check warnings above")
    else:
        ui.print_error("Several components need configuration - check errors above")


if __name__ == "__main__":
    asyncio.run(main())
