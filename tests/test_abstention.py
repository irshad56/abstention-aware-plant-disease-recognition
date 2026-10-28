"""Tests for calibrated abstention policies and decision states."""

import pytest
from plant_disease_agent.abstention import AbstentionPolicy
from plant_disease_agent.schemas import (
    AbstentionState,
    EnsembleResult,
    SuitabilityAssessment,
)


def test_abstention_on_invalid_image():
    policy = AbstentionPolicy()
    unsuitable = SuitabilityAssessment(
        is_suitable=False,
        reasons=["Severely blurry", "Too dark"]
    )
    decision = policy.evaluate(unsuitable, None)
    assert decision.state == AbstentionState.INVALID_IMAGE
    assert decision.should_abstain is True
    assert len(decision.triggered_criteria) == 2


def test_abstention_on_insufficient_models():
    policy = AbstentionPolicy(min_participating_models=3)
    suitable = SuitabilityAssessment(is_suitable=True)
    degraded_ensemble = EnsembleResult(
        predicted_idx=0,
        predicted_class="Apple__Healthy",
        confidence=0.95,
        margin=0.90,
        predictive_entropy=0.1,
        model_disagreement=0.0,
        participating_models=["m1", "m2"],  # Only 2 models
    )
    decision = policy.evaluate(suitable, degraded_ensemble)
    assert decision.state == AbstentionState.MODEL_UNAVAILABLE
    assert decision.should_abstain is True


def test_abstention_on_low_confidence():
    # Override with explicit strict threshold to keep the test self-contained
    policy = AbstentionPolicy(min_confidence=0.70, max_predictive_entropy=1.85, min_margin=0.08)
    suitable = SuitabilityAssessment(is_suitable=True)
    low_conf_ensemble = EnsembleResult(
        predicted_idx=0,
        predicted_class="Apple__Apple Scab",
        confidence=0.30,   # Clearly below any reasonable threshold
        margin=0.25,
        predictive_entropy=0.8,
        model_disagreement=0.0,
        participating_models=["m1", "m2", "m3"],
    )
    decision = policy.evaluate(suitable, low_conf_ensemble)
    assert decision.state == AbstentionState.ABSTAINED
    assert decision.should_abstain is True
    assert any("Confidence" in c for c in decision.triggered_criteria)


def test_acceptance_when_confident_and_agreed():
    policy = AbstentionPolicy()
    suitable = SuitabilityAssessment(is_suitable=True)
    confident_ensemble = EnsembleResult(
        predicted_idx=0,
        predicted_class="Tomato__Early Blight",
        confidence=0.92,
        margin=0.85,
        predictive_entropy=0.35,
        model_disagreement=0.0,
        participating_models=["m1", "m2", "m3", "m4", "m5"],
    )
    decision = policy.evaluate(suitable, confident_ensemble)
    assert decision.state == AbstentionState.ACCEPTED
    assert decision.should_abstain is False
