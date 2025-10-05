"""
Pattern Miner

Discover correlations between signals and outcomes for context learning.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class Pattern:
    """Represents a discovered pattern."""

    def __init__(
        self,
        pattern_id: str,
        description: str,
        conditions: Dict[str, Any],
        support: float,
        confidence: float,
        lift: float
    ):
        self.pattern_id = pattern_id
        self.description = description
        self.conditions = conditions
        self.support = support  # Frequency in dataset
        self.confidence = confidence  # P(outcome|conditions)
        self.lift = lift  # How much better than random
        self.discovered_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "conditions": self.conditions,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "discovered_at": self.discovered_at.isoformat()
        }


class PatternMiner:
    """
    Mine correlations between signals and outcomes.

    Discovers patterns like:
    - "When Glassdoor < 3.2 AND hiring_freeze, then earnings_miss (confidence=0.78)"
    - "When analyst_downgrades >= 2 AND news_negative > 0.6, then stock_decline (confidence=0.82)"
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.6,
        min_lift: float = 1.2
    ):
        """
        Initialize pattern miner.

        Args:
            min_support: Minimum frequency for pattern (0-1)
            min_confidence: Minimum confidence for pattern (0-1)
            min_lift: Minimum lift over random baseline
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.patterns: List[Pattern] = []

        logger.info(
            f"PatternMiner initialized: "
            f"support>={min_support}, confidence>={min_confidence}, lift>={min_lift}"
        )

    def mine_patterns(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """
        Mine patterns from historical signal-outcome pairs.

        Args:
            historical_data: List of dictionaries with:
                - signals: Dict of signal values
                - outcome: Binary outcome (1=positive, 0=negative)
                - ticker: Stock ticker
                - date: Date of observation

        Returns:
            List of discovered patterns
        """
        if len(historical_data) < 10:
            logger.warning(
                f"Insufficient data for pattern mining: {len(historical_data)} samples. "
                f"Recommend at least 50 samples."
            )
            return []

        logger.info(f"Mining patterns from {len(historical_data)} historical samples")

        # Discretize continuous signals
        discretized = self._discretize_signals(historical_data)

        # Mine single-condition patterns
        single_patterns = self._mine_single_conditions(discretized)

        # Mine two-condition patterns
        two_patterns = self._mine_two_conditions(discretized)

        # Combine and filter
        all_patterns = single_patterns + two_patterns
        self.patterns = [
            p for p in all_patterns
            if p.support >= self.min_support
            and p.confidence >= self.min_confidence
            and p.lift >= self.min_lift
        ]

        # Sort by confidence * lift
        self.patterns.sort(
            key=lambda p: p.confidence * p.lift,
            reverse=True
        )

        logger.info(f"Discovered {len(self.patterns)} patterns meeting thresholds")

        return self.patterns

    def _discretize_signals(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Discretize continuous signals into categorical bins.

        Returns:
            List of discretized data
        """
        discretized = []

        for sample in historical_data:
            signals = sample.get("signals", {})
            disc_signals = {}

            for key, value in signals.items():
                if isinstance(value, (int, float)):
                    # Discretize numeric values
                    if "sentiment" in key or "score" in key:
                        # Sentiment/score in [-1, 1] or [0, 1]
                        if value > 0.3:
                            disc_signals[key] = "high"
                        elif value < -0.3:
                            disc_signals[key] = "low"
                        else:
                            disc_signals[key] = "neutral"

                    elif "rating" in key:
                        # Ratings (e.g., Glassdoor 1-5)
                        if value >= 4.0:
                            disc_signals[key] = "high"
                        elif value <= 3.0:
                            disc_signals[key] = "low"
                        else:
                            disc_signals[key] = "medium"

                    elif "ratio" in key or "percent" in key:
                        # Ratios/percentages
                        if value > 0.6:
                            disc_signals[key] = "high"
                        elif value < 0.4:
                            disc_signals[key] = "low"
                        else:
                            disc_signals[key] = "medium"

                    else:
                        # Generic numeric
                        median = np.median([
                            s.get("signals", {}).get(key, 0)
                            for s in historical_data
                            if key in s.get("signals", {})
                        ])

                        disc_signals[key] = "high" if value > median else "low"

                else:
                    # Keep categorical as-is
                    disc_signals[key] = value

            discretized.append({
                "signals": disc_signals,
                "outcome": sample.get("outcome", 0),
                "ticker": sample.get("ticker", ""),
                "date": sample.get("date", "")
            })

        return discretized

    def _mine_single_conditions(
        self,
        discretized_data: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Mine single-condition patterns."""
        patterns = []
        n_total = len(discretized_data)
        n_positive = sum(d["outcome"] for d in discretized_data)

        # Baseline probability
        p_outcome = n_positive / n_total if n_total > 0 else 0.0

        # Count signal-outcome co-occurrences
        signal_counts = defaultdict(lambda: {"total": 0, "positive": 0})

        for sample in discretized_data:
            outcome = sample["outcome"]
            for signal_name, signal_value in sample["signals"].items():
                key = f"{signal_name}={signal_value}"
                signal_counts[key]["total"] += 1
                if outcome:
                    signal_counts[key]["positive"] += 1

        # Calculate metrics for each condition
        for condition, counts in signal_counts.items():
            total = counts["total"]
            positive = counts["positive"]

            support = total / n_total
            confidence = positive / total if total > 0 else 0.0
            lift = confidence / p_outcome if p_outcome > 0 else 1.0

            if support >= self.min_support * 0.5:  # Relaxed for single conditions
                signal_name, signal_value = condition.split("=", 1)
                patterns.append(Pattern(
                    pattern_id=f"single_{condition.replace('=', '_')}",
                    description=f"When {signal_name} is {signal_value}, outcome is positive",
                    conditions={signal_name: signal_value},
                    support=support,
                    confidence=confidence,
                    lift=lift
                ))

        return patterns

    def _mine_two_conditions(
        self,
        discretized_data: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Mine two-condition patterns."""
        patterns = []
        n_total = len(discretized_data)
        n_positive = sum(d["outcome"] for d in discretized_data)

        p_outcome = n_positive / n_total if n_total > 0 else 0.0

        # Get unique signals
        all_signals = set()
        for sample in discretized_data:
            for signal_name, signal_value in sample["signals"].items():
                all_signals.add((signal_name, signal_value))

        all_signals = list(all_signals)

        # Mine pairs
        for i, (sig1_name, sig1_val) in enumerate(all_signals):
            for sig2_name, sig2_val in all_signals[i + 1:]:
                # Count co-occurrences
                total = 0
                positive = 0

                for sample in discretized_data:
                    signals = sample["signals"]
                    if (signals.get(sig1_name) == sig1_val and
                        signals.get(sig2_name) == sig2_val):
                        total += 1
                        if sample["outcome"]:
                            positive += 1

                if total == 0:
                    continue

                support = total / n_total
                confidence = positive / total
                lift = confidence / p_outcome if p_outcome > 0 else 1.0

                if support >= self.min_support:
                    patterns.append(Pattern(
                        pattern_id=f"two_{sig1_name}_{sig2_name}",
                        description=(
                            f"When {sig1_name}={sig1_val} AND {sig2_name}={sig2_val}, "
                            f"outcome is positive"
                        ),
                        conditions={sig1_name: sig1_val, sig2_name: sig2_val},
                        support=support,
                        confidence=confidence,
                        lift=lift
                    ))

        return patterns

    def get_top_patterns(self, n: int = 10) -> List[Pattern]:
        """
        Get top N patterns by confidence * lift.

        Args:
            n: Number of patterns to return

        Returns:
            Top N patterns
        """
        return self.patterns[:n]

    def match_pattern(
        self,
        current_signals: Dict[str, Any]
    ) -> List[Tuple[Pattern, float]]:
        """
        Match current signals against discovered patterns.

        Args:
            current_signals: Current signal values

        Returns:
            List of (Pattern, match_score) tuples
        """
        matches = []

        # Discretize current signals
        disc_signals = {}
        for key, value in current_signals.items():
            if isinstance(value, (int, float)):
                if "sentiment" in key or "score" in key:
                    if value > 0.3:
                        disc_signals[key] = "high"
                    elif value < -0.3:
                        disc_signals[key] = "low"
                    else:
                        disc_signals[key] = "neutral"
                elif "rating" in key:
                    if value >= 4.0:
                        disc_signals[key] = "high"
                    elif value <= 3.0:
                        disc_signals[key] = "low"
                    else:
                        disc_signals[key] = "medium"
                else:
                    disc_signals[key] = value
            else:
                disc_signals[key] = value

        # Check each pattern
        for pattern in self.patterns:
            conditions_met = 0
            total_conditions = len(pattern.conditions)

            for cond_signal, cond_value in pattern.conditions.items():
                if disc_signals.get(cond_signal) == cond_value:
                    conditions_met += 1

            if conditions_met > 0:
                match_score = conditions_met / total_conditions
                if match_score >= 0.5:  # At least 50% match
                    matches.append((pattern, match_score))

        # Sort by match_score * pattern_confidence
        matches.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)

        return matches

    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of discovered patterns.

        Returns:
            Summary dictionary
        """
        if not self.patterns:
            return {
                "total_patterns": 0,
                "avg_confidence": 0.0,
                "avg_lift": 0.0,
                "top_patterns": []
            }

        confidences = [p.confidence for p in self.patterns]
        lifts = [p.lift for p in self.patterns]

        return {
            "total_patterns": len(self.patterns),
            "avg_confidence": round(np.mean(confidences), 3),
            "avg_lift": round(np.mean(lifts), 3),
            "max_confidence": round(max(confidences), 3),
            "max_lift": round(max(lifts), 3),
            "top_patterns": [
                {
                    "description": p.description,
                    "confidence": p.confidence,
                    "lift": p.lift,
                    "support": p.support
                }
                for p in self.get_top_patterns(5)
            ]
        }
