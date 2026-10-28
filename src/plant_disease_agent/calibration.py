"""Calibration and selective prediction metrics (ECE, Brier Score, NLL, Temperature Scaling)."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 29,
) -> float:
    """Computes multiclass Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2."""
    n = probabilities.shape[0]
    one_hot = np.zeros((n, num_classes), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Computes Negative Log Likelihood (Cross-Entropy)."""
    n = probabilities.shape[0]
    clipped = np.clip(probabilities[np.arange(n), labels], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def compute_selective_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    abstention_mask: np.ndarray,
) -> Dict[str, float]:
    """Computes coverage, selective accuracy, and selective risk after abstention."""
    total = len(labels)
    accepted_mask = ~abstention_mask
    n_accepted = int(np.sum(accepted_mask))

    coverage = float(n_accepted / total) if total > 0 else 0.0
    if n_accepted == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "selective_accuracy": 0.0,
            "selective_risk": 0.0,
        }

    correct_accepted = np.sum(predictions[accepted_mask] == labels[accepted_mask])
    selective_acc = float(correct_accepted / n_accepted)
    selective_risk = float(1.0 - selective_acc)

    return {
        "coverage": round(coverage, 4),
        "abstention_rate": round(1.0 - coverage, 4),
        "selective_accuracy": round(selective_acc, 4),
        "selective_risk": round(selective_risk, 4),
    }


class TemperatureScaler:
    """Applies temperature scaling to uncalibrated logits."""
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(float(temperature), 1e-4)

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
