"""Tests for predictive entropy, margin, and probability vector validation."""

import numpy as np
import pytest

from plant_disease_agent.uncertainty import (
    compute_model_disagreement,
    compute_predictive_entropy,
    compute_top2_margin,
    validate_probability_vector,
)


def test_predictive_entropy():
    # Deterministic 1.0 distribution has entropy 0.0
    p_det = np.zeros(29)
    p_det[0] = 1.0
    assert pytest.approx(compute_predictive_entropy(p_det), abs=1e-5) == 0.0

    # Uniform distribution over 29 classes has entropy ln(29) ≈ 3.367
    p_uni = np.ones(29) / 29.0
    expected_max = float(np.log(29.0))
    assert pytest.approx(compute_predictive_entropy(p_uni), rel=1e-3) == expected_max


def test_top2_margin():
    probs = np.zeros(29)
    probs[3] = 0.85
    probs[5] = 0.10
    probs[8] = 0.05
    assert pytest.approx(compute_top2_margin(probs), rel=1e-4) == 0.75


def test_validate_probability_vector():
    # Valid vector
    v = [1.0 / 29.0] * 29
    res = validate_probability_vector(v, expected_dim=29)
    assert len(res) == 29
    assert pytest.approx(float(np.sum(res)), abs=1e-6) == 1.0

    # Wrong dimension
    with pytest.raises(ValueError):
        validate_probability_vector([0.5, 0.5], expected_dim=29)

    # NaN / Inf rejection
    with pytest.raises(ValueError):
        validate_probability_vector([float("nan")] * 29, expected_dim=29)
