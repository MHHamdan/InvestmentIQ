"""
Test Hugging Face integration for InvestmentIQ.

Tests:
1. FinBERT sentiment analysis
2. LLM text generation
3. Evidence summarization
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.hf_client import get_hf_client
from tools.news_api_tool import NewsAPITool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hf_client():
    """Test HF client initialization."""
    logger.info("Testing HF Client...")

    try:
        client = get_hf_client()

        if not client.api_key or client.api_key == "your_huggingface_key_here":
            logger.error("Hugging Face not available - check HUGGING_FACE_API_KEY in .env")
            return False

        logger.info(f"HF client initialized with API key: {client.api_key[:10]}...")
        logger.info("HF Client test: PASSED")
        return True

    except Exception as e:
        logger.error(f"HF Client test failed: {e}")
        return False


def test_text_generation():
    """Test basic text generation."""
    logger.info("\nTesting Text Generation...")

    try:
        client = get_hf_client()

        prompt = "Explain why diversification is important in investing in one sentence."
        logger.info(f"Prompt: {prompt}")

        response = client.generate_text(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )

        if response:
            logger.info(f"Response: {response[:200]}...")
            logger.info("Text Generation test: PASSED")
            return True
        else:
            logger.warning("Empty response from HF API")
            return False

    except Exception as e:
        logger.error(f"Text Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finbert_sentiment():
    """Test FinBERT sentiment analysis."""
    logger.info("\nTesting FinBERT Sentiment Analysis...")

    try:
        # Create sample news data
        sample_news = {
            "company": "AAPL",
            "articles": [
                {
                    "title": "Apple Reports Record Revenue Growth",
                    "description": "Apple Inc. announced strong quarterly earnings with revenue up 15%",
                    "sentiment": 0.0,  # Will be replaced by FinBERT
                    "publishedAt": "2025-01-15T10:00:00Z"
                },
                {
                    "title": "iPhone Sales Decline in Key Markets",
                    "description": "Apple faces challenges as iPhone sales drop in major markets",
                    "sentiment": 0.0,
                    "publishedAt": "2025-01-14T14:30:00Z"
                }
            ],
            "timestamp": "2025-01-15T12:00:00Z"
        }

        # Test sentiment enhancement
        tool = NewsAPITool()

        if not tool.use_hf_sentiment:
            logger.warning("HF sentiment not enabled - check API key")
            return False

        enhanced_news = tool._enhance_with_hf_sentiment(sample_news)

        logger.info("Sentiment analysis results:")
        for article in enhanced_news['articles']:
            logger.info(f"  {article['title'][:50]}...")
            logger.info(f"    Sentiment: {article.get('sentiment', 'N/A')}")
            logger.info(f"    Source: {article.get('sentiment_source', 'N/A')}")
            logger.info(f"    Confidence: {article.get('sentiment_confidence', 'N/A')}")

        logger.info("FinBERT Sentiment test: PASSED")
        return True

    except Exception as e:
        logger.error(f"FinBERT Sentiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evidence_summarization():
    """Test LLM-based evidence summarization."""
    logger.info("\nTesting Evidence Summarization...")

    try:
        from core.signal_fusion import SignalFusion
        from core.agent_contracts import Evidence
        from datetime import datetime

        # Create sample evidence
        evidence = [
            Evidence(
                source="financial_analyst",
                value={"revenue_growth": 0.15, "margin_improvement": 0.05},
                timestamp=datetime.utcnow(),
                description="Strong revenue growth of 15% YoY with improving margins",
                confidence=0.85
            ),
            Evidence(
                source="market_intelligence",
                value={"rating": "Buy", "price_target": 220},
                timestamp=datetime.utcnow(),
                description="Analyst consensus rating upgraded to Buy with $220 target",
                confidence=0.75
            ),
            Evidence(
                source="sentiment_agent",
                value={"sentiment_score": 0.65},
                timestamp=datetime.utcnow(),
                description="Positive news sentiment driven by product launch success",
                confidence=0.70
            )
        ]

        # Test summarization
        fusion = SignalFusion(use_llm_summary=True)

        if not fusion.use_llm_summary:
            logger.warning("LLM summary not enabled")
            return False

        summary = fusion._generate_llm_evidence_summary("AAPL", evidence, 0.6)

        if summary:
            logger.info(f"Generated summary: {summary}")
            logger.info("Evidence Summarization test: PASSED")
            return True
        else:
            logger.warning("No summary generated")
            return False

    except Exception as e:
        logger.error(f"Evidence Summarization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("InvestmentIQ Hugging Face Integration Tests")
    logger.info("=" * 60)

    results = []

    # Test 1: HF Client
    results.append(("HF Client", test_hf_client()))

    # Test 2: Text Generation
    results.append(("Text Generation", test_text_generation()))

    # Test 3: FinBERT Sentiment
    results.append(("FinBERT Sentiment", test_finbert_sentiment()))

    # Test 4: Evidence Summarization
    results.append(("Evidence Summarization", test_evidence_summarization()))

    # Print summary
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
    success = main()
    sys.exit(0 if success else 1)
