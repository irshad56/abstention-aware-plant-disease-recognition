"""Tests for partial model failure recovery and offline LLM fallback."""

import pytest
from plant_disease_agent.agents.tools import LLMSynthesizer, generate_deterministic_explanation
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.inference import InferenceManager
from plant_disease_agent.schemas import EnsembleResult, EvidenceRecord
from conftest import MockModelAdapter


def test_partial_model_failure_recovery(mock_class_names, valid_leaf_image):
    # 2 out of 5 models fail during forward pass
    adapters = {
        "m1": MockModelAdapter("m1", "M1", target_idx=0, should_fail=False),
        "m2": MockModelAdapter("m2", "M2", target_idx=0, should_fail=False),
        "m3": MockModelAdapter("m3", "M3", target_idx=0, should_fail=False),
        "m4": MockModelAdapter("m4", "M4", should_fail=True),
        "m5": MockModelAdapter("m5", "M5", should_fail=True),
    }

    inf_manager = InferenceManager(adapters)
    predictions = inf_manager.run_all(valid_leaf_image)

    assert predictions["m1"].status == "success"
    assert predictions["m4"].status in ("failed", "unloaded")
    assert predictions["m5"].status in ("failed", "unloaded")

    aggregator = DeepEnsembleAggregator(class_names=mock_class_names, min_participating_models=3)
    res = aggregator.aggregate(predictions)

    # 3 models succeeded -> meets min_participating_models
    assert len(res.participating_models) == 3
    assert res.predicted_idx == 0


def test_deterministic_fallback_when_no_llm_key(mock_class_names):
    synthesizer = LLMSynthesizer(api_key="")
    assert synthesizer.client is None

    ens = EnsembleResult(
        predicted_idx=0,
        predicted_class=mock_class_names[0],
        confidence=0.91,
        margin=0.85,
        predictive_entropy=0.4,
        model_disagreement=0.0,
        participating_models=["m1", "m2", "m3"],
    )

    ev_rec = EvidenceRecord(
        citation_id="CIT-1",
        title="Apple Scab Factsheet",
        publisher="Penn State Extension",
        url="https://extension.psu.edu/apple-scab",
        source_type="extension_service",
        excerpt="Symptoms: Olive green velvety spots | Management: Prune dormant canopies",
    )

    explanation = synthesizer.synthesize(ens, [ev_rec])

    assert explanation is not None
    assert explanation.predicted_class == mock_class_names[0]
    assert len(explanation.citations) == 1
    assert "Apple Scab" in explanation.summary or "Apple__Apple Scab" in explanation.summary
    assert len(explanation.verified_symptoms) > 0
    assert len(explanation.recommended_actions) > 0
