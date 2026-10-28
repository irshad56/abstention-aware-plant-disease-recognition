"""Typed schemas for image validation, model inference, ensemble uncertainty,
abstention, evidence retrieval, and agent orchestration.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AbstentionState(str, Enum):
    ACCEPTED = "ACCEPTED"
    ABSTAINED = "ABSTAINED"
    INVALID_IMAGE = "INVALID_IMAGE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"


class SuitabilityAssessment(BaseModel):
    is_suitable: bool = Field(description="Whether the image meets minimal diagnostic quality standards")
    reasons: List[str] = Field(default_factory=list, description="Quality diagnostics and warnings")
    width: int = Field(default=0)
    height: int = Field(default=0)
    sharpness_variance: float = Field(default=0.0, description="Laplacian variance sharpness heuristic")
    mean_intensity: float = Field(default=0.0, description="Normalized brightness across RGB channels (0.0 - 1.0)")
    aspect_ratio: float = Field(default=1.0)


class ModelPrediction(BaseModel):
    model_id: str
    display_name: str
    predicted_idx: int
    predicted_class: str
    confidence: float = Field(description="Softmax probability of the predicted top class")
    probabilities: List[float] = Field(description="Full 29-class probability vector")
    latency_ms: float = Field(default=0.0)
    status: str = Field(default="success", description="'success', 'failed', or 'unloaded'")
    error_message: Optional[str] = None


class EnsembleResult(BaseModel):
    predicted_idx: int
    predicted_class: str
    confidence: float = Field(description="Mean probability of the winning class across models")
    margin: float = Field(description="Difference between top-1 and top-2 ensemble probabilities")
    predictive_entropy: float = Field(description="Predictive entropy in nats (-sum(p * ln(p)))")
    model_disagreement: float = Field(description="Fraction of participating models whose argmax differs from ensemble")
    participating_models: List[str] = Field(default_factory=list)
    ensemble_probabilities: List[float] = Field(default_factory=list)
    per_model_predictions: Dict[str, ModelPrediction] = Field(default_factory=dict)


class AbstentionDecision(BaseModel):
    state: AbstentionState
    should_abstain: bool
    primary_reason: str
    triggered_criteria: List[str] = Field(default_factory=list)
    confidence_assessment: str = Field(default="Uncertain")
    recommendation: str = Field(default="Consult an agricultural expert.")


class EvidenceRecord(BaseModel):
    citation_id: str
    title: str
    publisher: str
    url: str
    source_type: str = Field(default="extension_service")
    excerpt: str
    relevance_score: float = Field(default=1.0)
    is_validated: bool = Field(default=True)
    validation_notes: Optional[str] = None


class GroundedExplanation(BaseModel):
    predicted_class: str
    plant_species: str
    disease_name: str
    confidence_percentage: float
    summary: str
    verified_symptoms: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    preventative_measures: List[str] = Field(default_factory=list)
    scientific_literature_grounding: Optional[str] = Field(
        default=None,
        description="Grounded scientific synthesis linking pathological observations to peer-reviewed literature and DOI citations."
    )
    citations: List[EvidenceRecord] = Field(default_factory=list)
    disclaimer: str = (
        "Agricultural Diagnostic Notice: Image-based plant disease recognition is an assistive research "
        "tool and cannot substitute for on-site agronomical inspection or laboratory assays."
    )


class AgentTraceStep(BaseModel):
    node_name: str
    timestamp: str
    status: str
    duration_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


class FinalResponse(BaseModel):
    decision_state: AbstentionState
    suitability: SuitabilityAssessment
    ensemble_result: Optional[EnsembleResult] = None
    abstention_decision: AbstentionDecision
    evidence_records: List[EvidenceRecord] = Field(default_factory=list)
    explanation: Optional[GroundedExplanation] = None
    trace: List[AgentTraceStep] = Field(default_factory=list)
    total_latency_ms: float = 0.0
