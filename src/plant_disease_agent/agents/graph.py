"""LangGraph workflow definition for Abstention-Aware Plant Disease Recognition."""

from functools import partial
from typing import Dict, Literal
from langgraph.graph import END, StateGraph

from plant_disease_agent.agents.nodes import (
    node_abstention_handler,
    node_ensemble_uncertainty,
    node_evidence_retrieval,
    node_evidence_validation,
    node_grounded_explanation,
    node_image_validation,
    node_model_inference,
    node_query_planning,
    node_response_assembly,
)
from plant_disease_agent.agents.state import AgentState
from plant_disease_agent.agents.tools import LLMSynthesizer
from plant_disease_agent.abstention import AbstentionPolicy
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.image_validation import ImageValidator
from plant_disease_agent.inference import InferenceManager


def route_after_validation(state: AgentState) -> Literal["model_inference", "abstention_handler"]:
    """Conditional Edge 1: Routes invalid images directly to abstention."""
    suitability = state.get("suitability")
    if suitability is not None and suitability.is_suitable:
        return "model_inference"
    return "abstention_handler"


def route_after_ensemble(state: AgentState) -> Literal["query_planning", "abstention_handler"]:
    """Conditional Edge 2: Routes high uncertainty or abstention decisions away from explanation."""
    decision = state.get("abstention_decision")
    if decision is not None and not decision.should_abstain:
        return "query_planning"
    return "abstention_handler"


def build_plant_disease_graph(
    validator: ImageValidator,
    inference_manager: InferenceManager,
    aggregator: DeepEnsembleAggregator,
    policy: AbstentionPolicy,
    retriever: EvidenceRetriever,
    evidence_validator: EvidenceValidator,
    synthesizer: LLMSynthesizer,
) -> StateGraph:
    """Constructs and compiles the full LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # Register Nodes with dependency injection
    workflow.add_node(
        "image_validation",
        partial(node_image_validation, validator=validator),
    )
    workflow.add_node(
        "model_inference",
        partial(node_model_inference, inference_manager=inference_manager),
    )
    workflow.add_node(
        "ensemble_uncertainty",
        partial(node_ensemble_uncertainty, aggregator=aggregator, policy=policy),
    )
    workflow.add_node(
        "abstention_handler",
        node_abstention_handler,
    )
    workflow.add_node(
        "query_planning",
        node_query_planning,
    )
    workflow.add_node(
        "evidence_retrieval",
        partial(node_evidence_retrieval, retriever=retriever),
    )
    workflow.add_node(
        "evidence_validation",
        partial(node_evidence_validation, validator=evidence_validator),
    )
    workflow.add_node(
        "grounded_explanation",
        partial(node_grounded_explanation, synthesizer=synthesizer),
    )
    workflow.add_node(
        "response_assembly",
        node_response_assembly,
    )

    # Set Entry Point
    workflow.set_entry_point("image_validation")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "image_validation",
        route_after_validation,
        {
            "model_inference": "model_inference",
            "abstention_handler": "abstention_handler",
        },
    )

    workflow.add_edge("model_inference", "ensemble_uncertainty")

    workflow.add_conditional_edges(
        "ensemble_uncertainty",
        route_after_ensemble,
        {
            "query_planning": "query_planning",
            "abstention_handler": "abstention_handler",
        },
    )

    workflow.add_edge("abstention_handler", "response_assembly")
    workflow.add_edge("query_planning", "evidence_retrieval")
    workflow.add_edge("evidence_retrieval", "evidence_validation")
    workflow.add_edge("evidence_validation", "grounded_explanation")
    workflow.add_edge("grounded_explanation", "response_assembly")
    workflow.add_edge("response_assembly", END)

    return workflow.compile()
