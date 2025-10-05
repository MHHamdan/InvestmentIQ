"""
Signal Fusion

Weighted ensemble fusion of agent signals with SHAP-like explanations.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from core.agent_contracts import AgentOutput, FusedSignal, Evidence, SignalType

logger = logging.getLogger(__name__)


class SignalFusion:
    """
    Fuse multiple agent signals into unified recommendation.

    Methods:
    - Weighted average fusion
    - Confidence-weighted fusion
    - SHAP-like contribution explanations
    """

    def __init__(self, method: str = "weighted_average"):
        """
        Initialize signal fusion.

        Args:
            method: Fusion method (weighted_average, confidence_weighted)
        """
        self.method = method
        logger.info(f"SignalFusion initialized with method: {method}")

    def fuse(
        self,
        ticker: str,
        agent_outputs: List[AgentOutput],
        weights: Dict[str, float] = None
    ) -> FusedSignal:
        """
        Fuse agent signals into unified score.

        Args:
            ticker: Stock ticker
            agent_outputs: List of agent outputs
            weights: Optional custom weights (defaults to equal weights)

        Returns:
            FusedSignal with final score and explanations
        """
        if not agent_outputs:
            raise ValueError("No agent outputs to fuse")

        # Build agent signals dict
        agent_signals = {
            output.agent_id: output for output in agent_outputs
        }

        # Determine weights
        if weights is None:
            weights = self._calculate_default_weights(agent_signals)

        # Normalize weights
        weights = self._normalize_weights(weights)

        # Calculate fused score
        if self.method == "weighted_average":
            final_score = self._weighted_average_fusion(agent_signals, weights)
        elif self.method == "confidence_weighted":
            final_score, weights = self._confidence_weighted_fusion(agent_signals)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        # Calculate confidence
        confidence = self._calculate_fused_confidence(agent_signals, weights)

        # Generate explanations
        explanations = self._generate_explanations(agent_signals, weights, final_score)

        # Collect top evidence
        top_evidence = self._collect_top_evidence(agent_signals)

        fused = FusedSignal(
            ticker=ticker,
            final_score=final_score,
            confidence=confidence,
            agent_signals=agent_signals,
            signal_weights=weights,
            explanations=explanations,
            top_evidence=top_evidence,
            fusion_method=self.method
        )

        logger.info(
            f"Fused {len(agent_outputs)} signals for {ticker}: "
            f"score={final_score:.3f}, confidence={confidence:.3f}"
        )

        return fused

    def _calculate_default_weights(
        self,
        agent_signals: Dict[str, AgentOutput]
    ) -> Dict[str, float]:
        """
        Calculate default weights based on agent types.

        Default hierarchy:
        - Financial: 0.30 (hard metrics)
        - Market Intelligence: 0.25 (analyst consensus)
        - Sentiment: 0.20 (news/social)
        - Workforce: 0.15 (employee signals)
        - Context: 0.10 (historical patterns)
        """
        default_weights = {
            SignalType.FINANCIAL: 0.30,
            SignalType.MARKET_INTELLIGENCE: 0.25,
            SignalType.SENTIMENT: 0.20,
            SignalType.WORKFORCE: 0.15,
            SignalType.CONTEXT: 0.10
        }

        weights = {}
        for agent_id, output in agent_signals.items():
            signal_type = output.signal
            weights[agent_id] = default_weights.get(signal_type, 0.10)

        return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            # Equal weights fallback
            n = len(weights)
            return {k: 1.0 / n for k in weights.keys()}

        return {k: v / total for k, v in weights.items()}

    def _weighted_average_fusion(
        self,
        agent_signals: Dict[str, AgentOutput],
        weights: Dict[str, float]
    ) -> float:
        """
        Weighted average of agent sentiments.

        Returns:
            Fused score in [-1, 1]
        """
        weighted_sum = sum(
            output.sentiment * weights[agent_id]
            for agent_id, output in agent_signals.items()
        )

        return round(weighted_sum, 3)

    def _confidence_weighted_fusion(
        self,
        agent_signals: Dict[str, AgentOutput]
    ) -> tuple[float, Dict[str, float]]:
        """
        Fusion weighted by agent confidence.

        Returns:
            Tuple of (fused_score, adjusted_weights)
        """
        # Weight by confidence
        confidence_weights = {
            agent_id: output.confidence
            for agent_id, output in agent_signals.items()
        }

        # Normalize
        total_confidence = sum(confidence_weights.values())
        if total_confidence == 0:
            # Fallback to equal weights
            n = len(confidence_weights)
            confidence_weights = {k: 1.0 / n for k in confidence_weights.keys()}
        else:
            confidence_weights = {
                k: v / total_confidence
                for k, v in confidence_weights.items()
            }

        # Weighted sum
        weighted_sum = sum(
            output.sentiment * confidence_weights[agent_id]
            for agent_id, output in agent_signals.items()
        )

        return round(weighted_sum, 3), confidence_weights

    def _calculate_fused_confidence(
        self,
        agent_signals: Dict[str, AgentOutput],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate confidence in fused signal.

        Factors:
        - Individual agent confidences
        - Agreement between agents (low variance = high confidence)
        """
        # Weighted average of confidences
        avg_confidence = sum(
            output.confidence * weights[agent_id]
            for agent_id, output in agent_signals.items()
        )

        # Signal agreement (variance of sentiments)
        sentiments = [output.sentiment for output in agent_signals.values()]
        mean_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)

        # Agreement score (inverse of variance)
        # Max variance for sentiment in [-1, 1] is ~1.0
        agreement = 1.0 - min(1.0, variance / 0.5)

        # Combine: 70% individual confidence, 30% agreement
        fused_confidence = 0.7 * avg_confidence + 0.3 * agreement

        return round(fused_confidence, 3)

    def _generate_explanations(
        self,
        agent_signals: Dict[str, AgentOutput],
        weights: Dict[str, float],
        final_score: float
    ) -> List[str]:
        """
        Generate SHAP-like contribution explanations.

        Returns:
            List of explanation strings
        """
        explanations = []

        # Sort agents by absolute contribution
        contributions = [
            (agent_id, output.sentiment * weights[agent_id], output)
            for agent_id, output in agent_signals.items()
        ]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Generate explanation for each agent
        for agent_id, contribution, output in contributions:
            direction = "positive" if contribution > 0 else "negative"
            strength = abs(contribution)

            if strength > 0.1:
                impact = "strong"
            elif strength > 0.05:
                impact = "moderate"
            else:
                impact = "weak"

            explanation = (
                f"{agent_id}: {impact} {direction} contribution ({contribution:+.3f}), "
                f"sentiment={output.sentiment:.3f}, confidence={output.confidence:.3f}"
            )
            explanations.append(explanation)

        # Overall summary
        summary = (
            f"Final fused score: {final_score:.3f} "
            f"({'bullish' if final_score > 0.2 else 'bearish' if final_score < -0.2 else 'neutral'})"
        )
        explanations.insert(0, summary)

        return explanations

    def _collect_top_evidence(
        self,
        agent_signals: Dict[str, AgentOutput],
        max_evidence: int = 10
    ) -> List[Evidence]:
        """
        Collect top evidence from all agents.

        Args:
            agent_signals: Agent outputs
            max_evidence: Maximum evidence items to return

        Returns:
            Top evidence sorted by confidence
        """
        all_evidence = []

        for output in agent_signals.values():
            all_evidence.extend(output.evidence)

        # Sort by confidence
        all_evidence.sort(key=lambda e: e.confidence, reverse=True)

        return all_evidence[:max_evidence]

    def calculate_contribution_scores(
        self,
        agent_signals: Dict[str, AgentOutput],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate SHAP-like contribution score for each agent.

        Returns:
            Dictionary mapping agent_id to contribution score
        """
        contributions = {}

        for agent_id, output in agent_signals.items():
            contribution = output.sentiment * weights[agent_id]
            contributions[agent_id] = round(contribution, 3)

        return contributions

    def detect_conflicts(
        self,
        agent_signals: Dict[str, AgentOutput],
        threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicting signals between agents.

        Args:
            agent_signals: Agent outputs
            threshold: Minimum sentiment difference to flag as conflict

        Returns:
            List of conflict descriptions
        """
        conflicts = []
        agent_list = list(agent_signals.items())

        for i, (agent_id_1, output_1) in enumerate(agent_list):
            for agent_id_2, output_2 in agent_list[i + 1:]:
                diff = abs(output_1.sentiment - output_2.sentiment)

                if diff >= threshold:
                    conflicts.append({
                        "agent_1": agent_id_1,
                        "agent_2": agent_id_2,
                        "sentiment_1": output_1.sentiment,
                        "sentiment_2": output_2.sentiment,
                        "difference": round(diff, 3),
                        "description": (
                            f"{agent_id_1} ({output_1.sentiment:+.2f}) conflicts with "
                            f"{agent_id_2} ({output_2.sentiment:+.2f}), "
                            f"difference: {diff:.2f}"
                        )
                    })

        return conflicts
