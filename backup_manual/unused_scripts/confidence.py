"""
Confidence Calibration

Calibrate confidence scores using isotonic regression and Platt scaling.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Calibrate confidence scores to match empirical accuracy.

    Methods:
    - Isotonic regression (non-parametric, monotonic)
    - Platt scaling (logistic regression)
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.

        Args:
            method: Calibration method (isotonic, platt)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False

        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        logger.info(f"ConfidenceCalibrator initialized with method: {method}")

    def fit(
        self,
        confidences: List[float],
        outcomes: List[bool]
    ):
        """
        Fit calibration model on historical data.

        Args:
            confidences: List of raw confidence scores [0, 1]
            outcomes: List of binary outcomes (True=correct, False=incorrect)
        """
        if len(confidences) != len(outcomes):
            raise ValueError("Confidences and outcomes must have same length")

        if len(confidences) < 10:
            logger.warning(
                f"Insufficient data for calibration: {len(confidences)} samples. "
                f"Recommend at least 50 samples."
            )

        confidences_array = np.array(confidences).reshape(-1, 1)
        outcomes_array = np.array(outcomes, dtype=int)

        if self.method == "isotonic":
            self.calibrator.fit(confidences, outcomes_array)
        elif self.method == "platt":
            self.calibrator.fit(confidences_array, outcomes_array)

        self.is_fitted = True
        logger.info(f"Calibrator fitted on {len(confidences)} samples")

    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.

        Args:
            confidence: Raw confidence score [0, 1]

        Returns:
            Calibrated confidence score [0, 1]
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw confidence")
            return confidence

        if self.method == "isotonic":
            calibrated = self.calibrator.predict([confidence])[0]
        elif self.method == "platt":
            calibrated = self.calibrator.predict_proba([[confidence]])[0][1]

        return float(np.clip(calibrated, 0.0, 1.0))

    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """
        Calibrate multiple confidence scores.

        Args:
            confidences: List of raw confidence scores

        Returns:
            List of calibrated confidence scores
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw confidences")
            return confidences

        confidences_array = np.array(confidences)

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(confidences_array)
        elif self.method == "platt":
            calibrated = self.calibrator.predict_proba(
                confidences_array.reshape(-1, 1)
            )[:, 1]

        return np.clip(calibrated, 0.0, 1.0).tolist()

    def evaluate_calibration(
        self,
        confidences: List[float],
        outcomes: List[bool],
        num_bins: int = 10
    ) -> dict:
        """
        Evaluate calibration quality using binned accuracy.

        Args:
            confidences: Confidence scores
            outcomes: True outcomes
            num_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration metrics
        """
        if len(confidences) != len(outcomes):
            raise ValueError("Confidences and outcomes must have same length")

        confidences_array = np.array(confidences)
        outcomes_array = np.array(outcomes, dtype=float)

        # Create bins
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences_array, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Calculate metrics per bin
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = confidences_array[mask].mean()
                bin_acc = outcomes_array[mask].mean()
                bin_count = mask.sum()

                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)

        # Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(confidences)

        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            ece += (count / total_samples) * abs(conf - acc)

        # Maximum Calibration Error (MCE)
        if bin_confidences:
            mce = max(
                abs(conf - acc)
                for conf, acc in zip(bin_confidences, bin_accuracies)
            )
        else:
            mce = 0.0

        return {
            "expected_calibration_error": round(ece, 4),
            "max_calibration_error": round(mce, 4),
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "total_samples": total_samples
        }


class EnsembleConfidence:
    """
    Calculate ensemble confidence from multiple models/agents.
    """

    @staticmethod
    def variance_based(
        predictions: List[float],
        confidences: List[float]
    ) -> float:
        """
        Calculate ensemble confidence based on prediction variance.

        Low variance (agreement) = high confidence
        High variance (disagreement) = low confidence

        Args:
            predictions: List of predictions from different models
            confidences: List of individual confidences

        Returns:
            Ensemble confidence [0, 1]
        """
        if not predictions or not confidences:
            return 0.0

        # Average individual confidences
        avg_confidence = np.mean(confidences)

        # Calculate prediction variance
        variance = np.var(predictions)

        # Agreement score (inverse of variance, normalized)
        # Assume predictions in [-1, 1], max variance ~1.0
        agreement = 1.0 - min(1.0, variance / 0.5)

        # Combine: 60% individual confidence, 40% agreement
        ensemble_conf = 0.6 * avg_confidence + 0.4 * agreement

        return float(np.clip(ensemble_conf, 0.0, 1.0))

    @staticmethod
    def entropy_based(probabilities: List[float]) -> float:
        """
        Calculate confidence based on prediction entropy.

        Low entropy (certain) = high confidence
        High entropy (uncertain) = low confidence

        Args:
            probabilities: List of class probabilities

        Returns:
            Confidence [0, 1]
        """
        if not probabilities:
            return 0.0

        probs = np.array(probabilities)
        probs = probs / probs.sum()  # Normalize

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Max entropy for N classes: log(N)
        max_entropy = np.log(len(probabilities))

        # Normalized entropy in [0, 1]
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0

        # Confidence = 1 - entropy
        confidence = 1.0 - normalized_entropy

        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def weighted_average(
        confidences: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """
        Weighted average of individual confidences.

        Args:
            confidences: List of confidence scores
            weights: Optional weights (defaults to equal)

        Returns:
            Weighted average confidence
        """
        if not confidences:
            return 0.0

        if weights is None:
            weights = [1.0] * len(confidences)

        if len(confidences) != len(weights):
            raise ValueError("Confidences and weights must have same length")

        confidences_array = np.array(confidences)
        weights_array = np.array(weights)

        # Normalize weights
        weights_normalized = weights_array / weights_array.sum()

        # Weighted average
        weighted_conf = np.sum(confidences_array * weights_normalized)

        return float(np.clip(weighted_conf, 0.0, 1.0))


def calculate_confidence_interval(
    predictions: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for ensemble predictions.

    Args:
        predictions: List of predictions
        confidence_level: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not predictions:
        return (0.0, 0.0)

    predictions_array = np.array(predictions)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)

    # Z-score for confidence level (approximation)
    if confidence_level == 0.95:
        z_score = 1.96
    elif confidence_level == 0.99:
        z_score = 2.576
    elif confidence_level == 0.90:
        z_score = 1.645
    else:
        # Use normal approximation
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

    margin = z_score * std / np.sqrt(len(predictions))

    lower = mean - margin
    upper = mean + margin

    return (float(lower), float(upper))
