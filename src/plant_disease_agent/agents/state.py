"""LangGraph agent workflow shared state definition."""

import sys
if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict

from typing import Any, Dict, List, Optional
from PIL import Image

from plant_disease_agent.schemas import (
    AbstentionDecision,
    AgentTraceStep,
    EnsembleResult,
    EvidenceRecord,
    GroundedExplanation,
    ModelPrediction,
    SuitabilityAssessment,
)


class AgentState(TypedDict):
    """Shared immutable state passed across LangGraph nodes."""
    image_bytes: NotRequired[Optional[bytes]]
    image_path: NotRequired[Optional[str]]
    image_obj: NotRequired[Optional[Image.Image]]
    suitability: NotRequired[Optional[SuitabilityAssessment]]
    predictions: NotRequired[Dict[str, ModelPrediction]]
    ensemble_result: NotRequired[Optional[EnsembleResult]]
    abstention_decision: NotRequired[Optional[AbstentionDecision]]
    query_plan: NotRequired[Optional[Dict[str, Any]]]
    raw_evidence: NotRequired[List[EvidenceRecord]]
    validated_evidence: NotRequired[List[EvidenceRecord]]
    explanation: NotRequired[Optional[GroundedExplanation]]
    trace: NotRequired[List[AgentTraceStep]]
    error: NotRequired[Optional[str]]
