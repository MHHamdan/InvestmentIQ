#!/usr/bin/env python
"""
Quick test script for Market Intelligence FMP integration.
"""
import asyncio
import os

# Enable FMP
os.environ['USE_FMP_DATA'] = 'true'

from agents.market_intelligence import MarketIntelligenceAgent


async def main():
    print("Testing Market Intelligence Agent with FMP data...")
    print("=" * 60)

    agent = MarketIntelligenceAgent()

    # Test with AAPL
    print("\nAnalyzing AAPL...")
    result = await agent.analyze('AAPL', 'Apple Inc.', 'Technology')

    print(f"\n✅ Analysis Complete:")
    print(f"  Sentiment: {result.sentiment:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Data Source: {result.metadata.get('data_source', 'unknown')}")
    print(f"  Total Analysts: {result.metrics['analyst_ratings']['total_analysts']}")
    print(f"  Avg Price Target: ${result.metrics['analyst_ratings']['avg_price_target']:.2f}")
    print(f"  Bullish Ratio: {result.metrics['analyst_ratings']['bullish_ratio']:.1%}")
    print(f"  Bearish Ratio: {result.metrics['analyst_ratings']['bearish_ratio']:.1%}")
    print(f"  Upgrades: {result.metrics['analyst_ratings']['upgrades']}")
    print(f"  Downgrades: {result.metrics['analyst_ratings']['downgrades']}")


if __name__ == "__main__":
    asyncio.run(main())
