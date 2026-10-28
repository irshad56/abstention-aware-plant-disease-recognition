"""Abstention policy and decision engine for uncertain or out-of-scope inputs."""

from typing import List, Optional
from plant_disease_agent.schemas import (
    AbstentionDecision,
    AbstentionState,
    EnsembleResult,
    SuitabilityAssessment,
)
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class AbstentionPolicy:
    """
    Abstention policy calibrated for a 29-class plant disease classifier.

    Entropy scale reference:
      - Maximum entropy (uniform over 29 classes): ln(29) ≈ 3.367 nats
      - Default threshold 1.85 nats  ≈ 55 % of maximum entropy
      - At uniform 2-class split the entropy is only ~0.69 nats, so real disease
        images with one dominant model prediction easily stay below 1.85 nats.
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        max_predictive_entropy: float = 1.85,
        min_margin: float = 0.08,
        max_model_disagreement: float = 0.60,
        min_participating_models: int = 3,
        abstain_on_quality_failure: bool = True,
    ):
        self.min_confidence = min_confidence
        self.max_predictive_entropy = max_predictive_entropy
        self.min_margin = min_margin
        self.max_model_disagreement = max_model_disagreement
        self.min_participating_models = min_participating_models
        self.abstain_on_quality_failure = abstain_on_quality_failure

    def evaluate(
        self,
        suitability: SuitabilityAssessment,
        ensemble_result: Optional[EnsembleResult],
    ) -> AbstentionDecision:
        """Evaluates whether the system has sufficient evidence to emit a diagnosis or must abstain."""
        triggered: List[str] = []

        # 1. Image Quality / Suitability Check
        if self.abstain_on_quality_failure and not suitability.is_suitable:
            reasons_str = "; ".join(suitability.reasons)
            return AbstentionDecision(
                state=AbstentionState.INVALID_IMAGE,
                should_abstain=True,
                primary_reason=f"Input image rejected by quality heuristics: {reasons_str}",
                triggered_criteria=suitability.reasons,
                confidence_assessment="Invalid Image",
                recommendation="Please submit a well-lit, in-focus, close-up photograph of the affected plant leaf.",
            )

        # 2. Model Availability Check
        if ensemble_result is None or len(ensemble_result.participating_models) < self.min_participating_models:
            active_count = len(ensemble_result.participating_models) if ensemble_result else 0
            return AbstentionDecision(
                state=AbstentionState.MODEL_UNAVAILABLE,
                should_abstain=True,
                primary_reason=(
                    f"Insufficient active models for safe ensemble inference "
                    f"({active_count} active < {self.min_participating_models} required)."
                ),
                triggered_criteria=[f"active_models={active_count}"],
                confidence_assessment="Degraded Ensemble",
                recommendation="Ensure model weights are placed in the configured artifact path and re-run.",
            )

        # 3. Confidence Threshold
        if ensemble_result.confidence < self.min_confidence:
            triggered.append(
                f"Confidence {ensemble_result.confidence:.3f} below threshold {self.min_confidence:.3f}"
            )

        # 4. Predictive Entropy Threshold
        if ensemble_result.predictive_entropy > self.max_predictive_entropy:
            triggered.append(
                f"Predictive entropy {ensemble_result.predictive_entropy:.3f} nats exceeds threshold {self.max_predictive_entropy:.3f}"
            )

        # 5. Top-2 Margin Threshold
        if ensemble_result.margin < self.min_margin:
            triggered.append(
                f"Prediction margin {ensemble_result.margin:.3f} below threshold {self.min_margin:.3f}"
            )

        # 6. Inter-Model Disagreement
        if ensemble_result.model_disagreement > self.max_model_disagreement:
            triggered.append(
                f"Model disagreement {ensemble_result.model_disagreement:.1%} exceeds threshold {self.max_model_disagreement:.1%}"
            )

        if triggered:
            primary = " | ".join(triggered)
            logger.info(f"Abstention triggered: {primary}")
            return AbstentionDecision(
                state=AbstentionState.ABSTAINED,
                should_abstain=True,
                primary_reason=f"High diagnostic uncertainty: {primary}",
                triggered_criteria=triggered,
                confidence_assessment="Uncertain / Potential Out-of-Distribution",
                recommendation=(
                    "The system refused diagnosis due to statistical uncertainty or model disagreement. "
                    "Submit additional canopy photos or consult an agricultural extension specialist."
                ),
            )

        # Sufficient confidence and agreement
        return AbstentionDecision(
            state=AbstentionState.ACCEPTED,
            should_abstain=False,
            primary_reason="Confidence, entropy, margin, and model agreement meet all acceptance criteria.",
            triggered_criteria=[],
            confidence_assessment="High Confidence",
            recommendation="Review the grounded evidence and recommended management guidelines below.",
        )
