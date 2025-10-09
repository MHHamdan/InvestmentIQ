"""
Test script for ADK Orchestrator

Tests the complete analysis pipeline with real data.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.adk_orchestrator import ADKOrchestrator


async def test_single_ticker(ticker: str = "AAPL"):
    """Test orchestrator with a single ticker."""
    print("=" * 80)
    print(f"🧪 Testing ADK Orchestrator with {ticker}")
    print("=" * 80)

    # Initialize orchestrator
    print("\n1️⃣ Initializing orchestrator...")
    orchestrator = ADKOrchestrator()

    # Run analysis
    print(f"\n2️⃣ Running analysis for {ticker}...")
    try:
        result = await orchestrator.analyze(ticker)

        # Display results
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)

        print(f"\n📊 Company: {result['company_name']}")
        print(f"🏢 Sector: {result['sector']}")
        print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")

        # Fused signal
        fused = result['fused_signal']
        print(f"\n🎯 Fused Score: {fused.final_score:.3f}")
        print(f"🔒 Confidence: {fused.confidence:.3f}")
        print(f"📈 Recommendation: {orchestrator.get_recommendation(fused)}")

        # Individual agent outputs
        print("\n" + "-" * 80)
        print("📋 INDIVIDUAL AGENT OUTPUTS")
        print("-" * 80)

        for agent_output in result['agent_outputs']:
            print(f"\n🤖 {agent_output.agent_id.upper()}")
            print(f"   Sentiment: {agent_output.sentiment:+.3f}")
            print(f"   Confidence: {agent_output.confidence:.3f}")
            print(f"   Signal Type: {agent_output.signal}")

            # Show actual metric values (first 5)
            print(f"   Metrics:")
            for i, (key, value) in enumerate(list(agent_output.metrics.items())[:5]):
                if isinstance(value, float):
                    print(f"      {key}: {value:.4f}")
                else:
                    print(f"      {key}: {value}")

            # Show reasoning and key factors (transparency!)
            if 'reasoning' in agent_output.metadata:
                print(f"   💡 Reasoning: {agent_output.metadata['reasoning'][:150]}...")
            if 'key_factors' in agent_output.metadata:
                print(f"   📌 Key Factors:")
                for factor in agent_output.metadata['key_factors']:
                    print(f"      • {factor}")

        # Signal weights
        print("\n" + "-" * 80)
        print("⚖️  FUSION WEIGHTS")
        print("-" * 80)
        for agent_id, weight in fused.signal_weights.items():
            contribution = fused.agent_signals[agent_id].sentiment * weight
            print(f"{agent_id:25s}: {weight:.2%} → contribution: {contribution:+.3f}")

        # Explanations
        print("\n" + "-" * 80)
        print("💡 EXPLANATIONS")
        print("-" * 80)
        for explanation in fused.explanations[:5]:  # Top 5 explanations
            print(f"   • {explanation}")

        # Save detailed results to JSON
        output_file = f"test_results_{ticker}.json"
        print(f"\n💾 Saving detailed results to {output_file}...")

        # Convert to serializable format
        output_data = {
            "ticker": result['ticker'],
            "company_name": result['company_name'],
            "sector": result['sector'],
            "timestamp": result['timestamp'],
            "execution_time": result['execution_time'],
            "fused_signal": {
                "final_score": fused.final_score,
                "confidence": fused.confidence,
                "recommendation": orchestrator.get_recommendation(fused),
                "fusion_method": fused.fusion_method,
                "explanations": fused.explanations
            },
            "agent_outputs": [
                {
                    "agent_id": ao.agent_id,
                    "signal_type": ao.signal,
                    "sentiment": ao.sentiment,
                    "confidence": ao.confidence,
                    "metrics": ao.metrics,
                    "metadata": ao.metadata
                }
                for ao in result['agent_outputs']
            ],
            "signal_weights": fused.signal_weights
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"✅ Results saved to {output_file}")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_multiple_tickers():
    """Test orchestrator with multiple tickers."""
    tickers = ["AAPL", "MSFT", "GOOGL"]

    print("=" * 80)
    print(f"🧪 Testing ADK Orchestrator with {len(tickers)} tickers")
    print("=" * 80)

    orchestrator = ADKOrchestrator()
    results = []

    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Analyzing {ticker}...")
        print('='*80)

        try:
            result = await orchestrator.analyze(ticker)
            fused = result['fused_signal']

            print(f"✅ {ticker}: Score={fused.final_score:+.3f}, Confidence={fused.confidence:.3f}")
            print(f"   Recommendation: {orchestrator.get_recommendation(fused)}")

            results.append({
                "ticker": ticker,
                "score": fused.final_score,
                "confidence": fused.confidence,
                "recommendation": orchestrator.get_recommendation(fused)
            })

        except Exception as e:
            print(f"❌ {ticker}: Failed - {e}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    for r in results:
        print(f"{r['ticker']:6s}: {r['score']:+.3f} | {r['confidence']:.3f} | {r['recommendation']}")

    return results


async def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Test ADK Orchestrator')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker to analyze')
    parser.add_argument('--multi', action='store_true', help='Test multiple tickers')

    args = parser.parse_args()

    if args.multi:
        await test_multiple_tickers()
    else:
        await test_single_ticker(args.ticker)


if __name__ == "__main__":
    asyncio.run(main())
