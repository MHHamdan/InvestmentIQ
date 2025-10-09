"""
Agent Evaluation Suite with LangSmith Integration

Evaluates InvestmentIQ agents against ground truth dataset.
Measures accuracy, directional correctness, and confidence calibration.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.adk_orchestrator import ADKOrchestrator


class AgentEvaluator:
    """Evaluates agent performance against ground truth."""
    
    def __init__(self, dataset_path: str = "tests/eval_dataset.json"):
        self.dataset_path = dataset_path
        self.orchestrator = ADKOrchestrator()
        self.results = []
        
    def load_dataset(self) -> List[Dict]:
        """Load ground truth evaluation dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate evaluation metrics."""
        
        # Sentiment MAE (Mean Absolute Error)
        sentiment_errors = []
        directional_correct = 0
        
        for pred, truth in zip(predictions, ground_truth):
            # Sentiment error
            error = abs(pred['sentiment'] - truth['expected_sentiment'])
            sentiment_errors.append(error)
            
            # Directional accuracy (sign matching)
            pred_direction = 1 if pred['sentiment'] > 0 else (-1 if pred['sentiment'] < 0 else 0)
            truth_direction = 1 if truth['expected_sentiment'] > 0 else (-1 if truth['expected_sentiment'] < 0 else 0)
            
            if pred_direction == truth_direction:
                directional_correct += 1
        
        mae = sum(sentiment_errors) / len(sentiment_errors)
        directional_accuracy = directional_correct / len(predictions)
        
        # Recommendation agreement
        rec_correct = sum(1 for p, t in zip(predictions, ground_truth) 
                         if p['recommendation'] == t['expected_recommendation'])
        rec_accuracy = rec_correct / len(predictions)
        
        # Confidence calibration
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        
        return {
            'sentiment_mae': mae,
            'directional_accuracy': directional_accuracy,
            'recommendation_accuracy': rec_accuracy,
            'avg_confidence': avg_confidence,
            'total_cases': len(predictions)
        }
    
    def get_recommendation(self, sentiment: float) -> str:
        """Convert sentiment to recommendation."""
        if sentiment > 0.5:
            return "STRONG BUY"
        elif sentiment > 0.2:
            return "BUY"
        elif sentiment > -0.2:
            return "HOLD"
        elif sentiment > -0.5:
            return "SELL"
        else:
            return "STRONG SELL"
    
    async def run_evaluation(self) -> Dict:
        """Run full evaluation suite."""
        print("🔬 Starting Agent Evaluation\n")
        print("=" * 60)
        
        dataset = self.load_dataset()
        predictions = []
        
        for i, case in enumerate(dataset, 1):
            ticker = case['ticker']
            print(f"\n[{i}/{len(dataset)}] Evaluating {ticker}...")
            
            try:
                # Run analysis
                result = await self.orchestrator.analyze(ticker)
                
                # Extract prediction
                if isinstance(result['fused_signal'], dict):
                    sentiment = result['fused_signal']['final_score']
                    confidence = result['fused_signal']['confidence']
                else:
                    sentiment = result['fused_signal'].final_score
                    confidence = result['fused_signal'].confidence
                
                recommendation = self.get_recommendation(sentiment)
                
                prediction = {
                    'ticker': ticker,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'recommendation': recommendation
                }
                
                predictions.append(prediction)
                
                # Show comparison
                expected = case['expected_sentiment']
                error = abs(sentiment - expected)
                match = "✓" if recommendation == case['expected_recommendation'] else "✗"
                
                print(f"   Predicted: {sentiment:+.3f} ({recommendation})")
                print(f"   Expected:  {expected:+.3f} ({case['expected_recommendation']})")
                print(f"   Error:     {error:.3f}  {match}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                predictions.append({
                    'ticker': ticker,
                    'sentiment': 0.0,
                    'confidence': 0.0,
                    'recommendation': 'ERROR',
                    'error': str(e)
                })
        
        # Calculate metrics
        print("\n" + "=" * 60)
        print("\n📊 Evaluation Metrics\n")
        
        metrics = self.calculate_metrics(predictions, dataset)
        
        print(f"Sentiment MAE:           {metrics['sentiment_mae']:.3f}")
        print(f"Directional Accuracy:    {metrics['directional_accuracy']:.1%}")
        print(f"Recommendation Accuracy: {metrics['recommendation_accuracy']:.1%}")
        print(f"Average Confidence:      {metrics['avg_confidence']:.1%}")
        print(f"Total Test Cases:        {metrics['total_cases']}")
        
        # Detailed results table
        print("\n" + "=" * 60)
        print("\n📋 Detailed Results\n")
        print(f"{'Ticker':<8} {'Predicted':<12} {'Expected':<12} {'Error':<8} {'Match':<6}")
        print("-" * 60)
        
        for pred, truth in zip(predictions, dataset):
            error = abs(pred['sentiment'] - truth['expected_sentiment'])
            match = "✓" if pred['recommendation'] == truth['expected_recommendation'] else "✗"
            print(f"{pred['ticker']:<8} {pred['sentiment']:>+6.3f} ({pred['recommendation']:<10}) "
                  f"{truth['expected_sentiment']:>+6.3f} ({truth['expected_recommendation']:<10}) "
                  f"{error:>6.3f}   {match}")
        
        # Save results
        eval_results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': predictions,
            'dataset': dataset
        }
        
        output_path = "tests/eval_results.json"
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
        return eval_results


async def main():
    """Run evaluation."""
    evaluator = AgentEvaluator()
    results = await evaluator.run_evaluation()
    
    # Show pass/fail
    print("\n" + "=" * 60)
    print("\n🎯 Evaluation Summary\n")
    
    metrics = results['metrics']
    
    # Define thresholds
    mae_threshold = 0.3  # MAE < 0.3 is good
    dir_threshold = 0.7  # >70% directional accuracy
    rec_threshold = 0.6  # >60% recommendation match
    
    mae_pass = metrics['sentiment_mae'] < mae_threshold
    dir_pass = metrics['directional_accuracy'] >= dir_threshold
    rec_pass = metrics['recommendation_accuracy'] >= rec_threshold
    
    print(f"Sentiment MAE < {mae_threshold}:        {'✅ PASS' if mae_pass else '❌ FAIL'}")
    print(f"Directional Accuracy ≥ {dir_threshold:.0%}: {'✅ PASS' if dir_pass else '❌ FAIL'}")
    print(f"Recommendation Match ≥ {rec_threshold:.0%}: {'✅ PASS' if rec_pass else '❌ FAIL'}")
    
    overall = "✅ PASS" if (mae_pass and dir_pass and rec_pass) else "⚠️  NEEDS IMPROVEMENT"
    print(f"\nOverall: {overall}")


if __name__ == "__main__":
    asyncio.run(main())
