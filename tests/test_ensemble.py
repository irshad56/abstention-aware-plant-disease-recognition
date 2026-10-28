"""Tests for deep ensemble aggregation and probability normalization."""

import numpy as np
import pytest

from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.schemas import ModelPrediction


def test_ensemble_aggregation_equal_weights(mock_class_names):
    aggregator = DeepEnsembleAggregator(class_names=mock_class_names)

    # 3 models voting for class 0 with 90% confidence
    p1 = [0.90] + [0.10 / 28] * 28
    p2 = [0.80] + [0.20 / 28] * 28
    p3 = [0.70] + [0.30 / 28] * 28

    preds = {
        "m1": ModelPrediction(model_id="m1", display_name="M1", predicted_idx=0, predicted_class=mock_class_names[0], confidence=0.90, probabilities=p1),
        "m2": ModelPrediction(model_id="m2", display_name="M2", predicted_idx=0, predicted_class=mock_class_names[0], confidence=0.80, probabilities=p2),
        "m3": ModelPrediction(model_id="m3", display_name="M3", predicted_idx=0, predicted_class=mock_class_names[0], confidence=0.70, probabilities=p3),
    }

    res = aggregator.aggregate(preds)
    assert res.predicted_idx == 0
    assert res.predicted_class == mock_class_names[0]
    # Expected mean: (0.9 + 0.8 + 0.7) / 3 = 0.80
    assert pytest.approx(res.confidence, rel=1e-3) == 0.80
    assert res.model_disagreement == 0.0
    assert len(res.ensemble_probabilities) == 29
    assert pytest.approx(sum(res.ensemble_probabilities), abs=1e-4) == 1.0


def test_ensemble_disagreement(mock_class_names):
    aggregator = DeepEnsembleAggregator(class_names=mock_class_names)

    # m1 and m2 vote class 0, m3 votes class 1
    p1 = [0.90] + [0.10 / 28] * 28
    p2 = [0.90] + [0.10 / 28] * 28
    p3 = [0.0] + [0.90] + [0.10 / 27] * 27

    preds = {
        "m1": ModelPrediction(model_id="m1", display_name="M1", predicted_idx=0, predicted_class=mock_class_names[0], confidence=0.90, probabilities=p1),
        "m2": ModelPrediction(model_id="m2", display_name="M2", predicted_idx=0, predicted_class=mock_class_names[0], confidence=0.90, probabilities=p2),
        "m3": ModelPrediction(model_id="m3", display_name="M3", predicted_idx=1, predicted_class=mock_class_names[1], confidence=0.90, probabilities=p3),
    }

    res = aggregator.aggregate(preds)
    assert res.predicted_idx == 0
    # 1 out of 3 models disagreed -> disagreement = 1/3 = 0.333
    assert pytest.approx(res.model_disagreement, rel=1e-2) == 0.333
