"""LangGraph execution nodes for abstention-aware plant disease recognition."""

import time
from datetime import datetime
from typing import Any, Dict, List

from plant_disease_agent.agents.state import AgentState
from plant_disease_agent.agents.tools import LLMSynthesizer, generate_deterministic_explanation
from plant_disease_agent.abstention import AbstentionPolicy
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.image_validation import ImageValidator
from plant_disease_agent.inference import InferenceManager
from plant_disease_agent.schemas import (
    AbstentionDecision,
    AbstentionState,
    AgentTraceStep,
    EnsembleResult,
    GroundedExplanation,
    SuitabilityAssessment,
)
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


def record_step(state: AgentState, node_name: str, status: str, duration_ms: float, details: Dict[str, Any]):
    """Appends an execution step to the agent trace."""
    if "trace" not in state or state["trace"] is None:
        state["trace"] = []

    step = AgentTraceStep(
        node_name=node_name,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        status=status,
        duration_ms=round(duration_ms, 2),
        details=details,
    )
    state["trace"].append(step)


def node_image_validation(
    state: AgentState,
    validator: ImageValidator,
) -> AgentState:
    """Node A: Image Intake and Quality Validation."""
    start = time.perf_counter()
    image_input = state.get("image_bytes") or state.get("image_path")

    if not image_input:
        suitability = SuitabilityAssessment(
            is_suitable=False,
            reasons=["No image payload or file path provided."]
        )
        state["suitability"] = suitability
        state["image_obj"] = None
        duration = (time.perf_counter() - start) * 1000.0
        record_step(state, "image_validation", "failed", duration, {"reasons": suitability.reasons})
        return state

    pil_img, suitability = validator.load_and_validate(image_input)
    state["image_obj"] = pil_img
    state["suitability"] = suitability

    duration = (time.perf_counter() - start) * 1000.0
    status = "passed" if suitability.is_suitable else "rejected"
    record_step(
        state,
        "image_validation",
        status,
        duration,
        {
            "is_suitable": suitability.is_suitable,
            "sharpness": suitability.sharpness_variance,
            "mean_intensity": suitability.mean_intensity,
            "reasons": suitability.reasons,
        },
    )
    return state


def node_model_inference(
    state: AgentState,
    inference_manager: InferenceManager,
) -> AgentState:
    """Node B: Multi-Model Inference."""
    start = time.perf_counter()
    pil_img = state.get("image_obj")

    if pil_img is None:
        duration = (time.perf_counter() - start) * 1000.0
        record_step(state, "model_inference", "skipped", duration, {"reason": "Image object is null"})
        state["predictions"] = {}
        return state

    predictions = inference_manager.run_all(pil_img)
    state["predictions"] = predictions

    duration = (time.perf_counter() - start) * 1000.0
    summary = {
        m_id: {
            "status": p.status,
            "pred": p.predicted_class if p.status == "success" else None,
            "conf": p.confidence,
            "latency_ms": p.latency_ms,
        }
        for m_id, p in predictions.items()
    }
    record_step(state, "model_inference", "completed", duration, {"models": summary})
    return state


def node_ensemble_uncertainty(
    state: AgentState,
    aggregator: DeepEnsembleAggregator,
    policy: AbstentionPolicy,
) -> AgentState:
    """Node C: Ensemble Aggregation & Uncertainty-Aware Abstention Decision."""
    start = time.perf_counter()
    preds = state.get("predictions", {})
    suitability = state.get("suitability")

    ensemble_result = aggregator.aggregate(preds)
    state["ensemble_result"] = ensemble_result

    abstention_decision = policy.evaluate(suitability, ensemble_result)
    state["abstention_decision"] = abstention_decision

    duration = (time.perf_counter() - start) * 1000.0
    record_step(
        state,
        "ensemble_uncertainty",
        abstention_decision.state.value,
        duration,
        {
            "decision": abstention_decision.state.value,
            "confidence": ensemble_result.confidence,
            "entropy": ensemble_result.predictive_entropy,
            "margin": ensemble_result.margin,
            "disagreement": ensemble_result.model_disagreement,
            "active_models": len(ensemble_result.participating_models),
        },
    )
    return state


def node_abstention_handler(state: AgentState) -> AgentState:
    """Conditional Route 1: Safe Abstention Handler."""
    start = time.perf_counter()
    decision = state.get("abstention_decision")
    
    if decision is None:
        suitability = state.get("suitability")
        if suitability and not suitability.is_suitable:
            reasons = suitability.reasons
            decision = AbstentionDecision(
                state=AbstentionState.INVALID_IMAGE,
                should_abstain=True,
                primary_reason=f"Input image rejected by quality heuristics: {'; '.join(reasons)}",
                triggered_criteria=reasons,
                confidence_assessment="Invalid Image",
                recommendation="Please submit a well-lit, in-focus, close-up photograph of the affected plant leaf.",
            )
        else:
            decision = AbstentionDecision(
                state=AbstentionState.ABSTAINED,
                should_abstain=True,
                primary_reason="General diagnostic uncertainty or model unavailability.",
                triggered_criteria=["uncertainty"],
                confidence_assessment="Uncertain",
                recommendation="Please verify image and model configuration.",
            )
        state["abstention_decision"] = decision

    reasons = decision.triggered_criteria if decision else ["Unspecified diagnostic uncertainty"]

    # Provide safe explanation refusing diagnosis
    explanation = GroundedExplanation(
        predicted_class="ABSTAINED",
        plant_species="Unknown / Unverified",
        disease_name="Diagnosis Withheld",
        confidence_percentage=0.0,
        summary=(
            f"The system has actively abstained from providing a plant disease diagnosis. "
            f"Primary trigger: {decision.primary_reason if decision else 'Low confidence'}."
        ),
        verified_symptoms=[],
        recommended_actions=[
            "Retake photograph under diffuse, even daylight without glare or heavy shadows.",
            "Ensure the leaf is flat, centered, and sharply focused.",
            "Consult an agricultural extension specialist or accredited laboratory for conclusive diagnosis.",
        ],
        preventative_measures=[],
        citations=[],
    )
    state["explanation"] = explanation
    duration = (time.perf_counter() - start) * 1000.0
    record_step(
        state,
        "abstention_handler",
        "handled",
        duration,
        {"primary_reason": decision.primary_reason if decision else "Abstained"},
    )
    return state


def node_query_planning(state: AgentState) -> AgentState:
    """Node D: Evidence Query Planning."""
    start = time.perf_counter()
    ensemble = state.get("ensemble_result")
    pred_class = ensemble.predicted_class if ensemble else "Unknown"

    parts = pred_class.split("__")
    plant = parts[0].strip()
    disease = parts[1].strip() if len(parts) > 1 else pred_class

    plan = {
        "predicted_class": pred_class,
        "plant_species": plant,
        "disease_name": disease,
        "search_query": f"{plant} {disease} symptoms management prevention extension",
        "timestamp": datetime.utcnow().isoformat(),
    }
    state["query_plan"] = plan
    duration = (time.perf_counter() - start) * 1000.0
    record_step(state, "query_planning", "planned", duration, plan)
    return state


def node_evidence_retrieval(
    state: AgentState,
    retriever: EvidenceRetriever,
) -> AgentState:
    """Node E: Evidence Retrieval Tool."""
    start = time.perf_counter()
    query_plan = state.get("query_plan", {})
    target_class = query_plan.get("predicted_class", "")

    records = retriever.retrieve(target_class, top_k=4)
    state["raw_evidence"] = records

    duration = (time.perf_counter() - start) * 1000.0
    record_step(
        state,
        "evidence_retrieval",
        "retrieved",
        duration,
        {"retrieved_count": len(records), "titles": [r.title for r in records]},
    )
    return state


def node_evidence_validation(
    state: AgentState,
    validator: EvidenceValidator,
) -> AgentState:
    """Node F: Evidence Validation & Citation Filtering."""
    start = time.perf_counter()
    raw = state.get("raw_evidence", [])
    query_plan = state.get("query_plan", {})
    plant = query_plan.get("plant_species", "")
    disease = query_plan.get("disease_name", "")

    validated = validator.filter_records(raw, plant, disease)
    state["validated_evidence"] = validated

    duration = (time.perf_counter() - start) * 1000.0
    record_step(
        state,
        "evidence_validation",
        "validated",
        duration,
        {"raw_count": len(raw), "validated_count": len(validated)},
    )
    return state


def node_grounded_explanation(
    state: AgentState,
    synthesizer: LLMSynthesizer,
) -> AgentState:
    """Node G: Grounded Explanation Generation."""
    start = time.perf_counter()
    ensemble = state.get("ensemble_result")
    validated_evidence = state.get("validated_evidence", [])

    explanation = synthesizer.synthesize(ensemble, validated_evidence)
    state["explanation"] = explanation

    duration = (time.perf_counter() - start) * 1000.0
    record_step(
        state,
        "grounded_explanation",
        "generated",
        duration,
        {"citations_included": len(explanation.citations)},
    )
    return state


def node_response_assembly(state: AgentState) -> AgentState:
    """Node H: Final Response Packaging."""
    start = time.perf_counter()
    duration = (time.perf_counter() - start) * 1000.0
    record_step(state, "response_assembly", "completed", duration, {"status": "ready"})
    return state
