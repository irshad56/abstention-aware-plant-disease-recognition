"""Tests for LangGraph agent workflow routing and node execution."""

import pytest
from plant_disease_agent.agents.graph import build_plant_disease_graph
from plant_disease_agent.agents.state import AgentState
from plant_disease_agent.agents.tools import LLMSynthesizer
from plant_disease_agent.abstention import AbstentionPolicy
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.image_validation import ImageValidator
from plant_disease_agent.inference import InferenceManager
from plant_disease_agent.schemas import AbstentionState


def test_graph_routes_corrupt_image_to_abstention(mock_adapters, mock_class_names):
    validator = ImageValidator()
    inf_manager = InferenceManager(mock_adapters)
    aggregator = DeepEnsembleAggregator(class_names=mock_class_names)
    policy = AbstentionPolicy()
    retriever = EvidenceRetriever()
    ev_validator = EvidenceValidator()
    synthesizer = LLMSynthesizer()

    graph = build_plant_disease_graph(
        validator=validator,
        inference_manager=inf_manager,
        aggregator=aggregator,
        policy=policy,
        retriever=retriever,
        evidence_validator=ev_validator,
        synthesizer=synthesizer,
    )

    state: AgentState = {
        "image_bytes": b"INVALID_GARBAGE_BYTES",
        "trace": [],
    }

    final_state = graph.invoke(state)

    assert final_state["abstention_decision"].state == AbstentionState.INVALID_IMAGE
    assert final_state["abstention_decision"].should_abstain is True
    # Trace recorded validation step
    node_names = [s.node_name for s in final_state["trace"]]
    assert "image_validation" in node_names
    assert "abstention_handler" in node_names
    # Model inference should have been bypassed by conditional router
    assert "model_inference" not in node_names


def test_graph_executes_full_retrieval_flow(mock_adapters, mock_class_names, valid_image_bytes):
    validator = ImageValidator()
    inf_manager = InferenceManager(mock_adapters)
    aggregator = DeepEnsembleAggregator(class_names=mock_class_names)
    policy = AbstentionPolicy()
    retriever = EvidenceRetriever()
    ev_validator = EvidenceValidator()
    synthesizer = LLMSynthesizer()

    graph = build_plant_disease_graph(
        validator=validator,
        inference_manager=inf_manager,
        aggregator=aggregator,
        policy=policy,
        retriever=retriever,
        evidence_validator=ev_validator,
        synthesizer=synthesizer,
    )

    state: AgentState = {
        "image_bytes": valid_image_bytes,
        "trace": [],
    }

    final_state = graph.invoke(state)

    assert final_state["abstention_decision"].state == AbstentionState.ACCEPTED
    assert final_state["abstention_decision"].should_abstain is False
    assert final_state["explanation"] is not None
    assert final_state["explanation"].predicted_class == mock_class_names[0]

    node_names = [s.node_name for s in final_state["trace"]]
    assert "image_validation" in node_names
    assert "model_inference" in node_names
    assert "ensemble_uncertainty" in node_names
    assert "query_planning" in node_names
    assert "evidence_retrieval" in node_names
    assert "evidence_validation" in node_names
    assert "grounded_explanation" in node_names
    assert "response_assembly" in node_names
