"""Uncertainty quantification: entropy, margin, confidence, and inter-model disagreement."""

from typing import Dict, List, Tuple
import numpy as np


def validate_probability_vector(probs: List[float], expected_dim: int = 29) -> np.ndarray:
    """Validates that probabilities form a proper probability distribution."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValueError(f"Expected 1D probability array of length {expected_dim}, got shape {arr.shape}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Probability vector contains NaN or Inf values.")
    if np.any(arr < 0.0):
        # Clip small numerical artifacts
        arr = np.clip(arr, 0.0, None)
    total = np.sum(arr)
    if total <= 0:
        raise ValueError("Sum of probability vector must be strictly positive.")
    return arr / total


def compute_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Shannon entropy in nats: -sum(p * ln(p)). Maximum for 29 classes is ln(29) ≈ 3.367."""
    p = np.clip(probabilities, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def compute_top2_margin(probabilities: np.ndarray) -> float:
    """Computes margin between highest and second-highest class probabilities."""
    sorted_probs = np.sort(probabilities)[::-1]
    if len(sorted_probs) < 2:
        return float(sorted_probs[0])
    return float(sorted_probs[0] - sorted_probs[1])


def compute_model_disagreement(
    ensemble_pred_idx: int,
    model_predictions: Dict[str, int]
) -> float:
    """Calculates the ratio of participating models that disagree with the ensemble decision."""
    if not model_predictions:
        return 1.0
    disagree_count = sum(1 for m_idx in model_predictions.values() if m_idx != ensemble_pred_idx)
    return float(disagree_count / len(model_predictions))


def compute_distribution_variance(model_prob_matrix: np.ndarray) -> float:
    """Computes mean variance across class probabilities among models: (num_models, num_classes)."""
    if model_prob_matrix.shape[0] <= 1:
        return 0.0
    # Variance along model axis, averaged over classes
    return float(np.mean(np.var(model_prob_matrix, axis=0, ddof=1)))
