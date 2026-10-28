"""Ensemble aggregation over deep learning model prediction vectors."""

from typing import Dict, List, Optional
import numpy as np

from plant_disease_agent.schemas import EnsembleResult, ModelPrediction
from plant_disease_agent.uncertainty import (
    compute_model_disagreement,
    compute_predictive_entropy,
    compute_top2_margin,
    validate_probability_vector,
)
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class DeepEnsembleAggregator:
    def __init__(
        self,
        class_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        min_participating_models: int = 3,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.weights = weights or {}
        self.min_participating_models = min_participating_models

    def aggregate(
        self,
        predictions: Dict[str, ModelPrediction]
    ) -> EnsembleResult:
        """Aggregates valid model prediction vectors into a calibrated ensemble distribution."""
        valid_predictions: Dict[str, ModelPrediction] = {
            m_id: pred
            for m_id, pred in predictions.items()
            if pred.status == "success" and len(pred.probabilities) == self.num_classes
        }

        if not valid_predictions:
            # Fallback for complete failure
            uniform = [1.0 / self.num_classes] * self.num_classes
            return EnsembleResult(
                predicted_idx=0,
                predicted_class=self.class_names[0],
                confidence=float(1.0 / self.num_classes),
                margin=0.0,
                predictive_entropy=float(np.log(self.num_classes)),
                model_disagreement=1.0,
                participating_models=[],
                ensemble_probabilities=uniform,
                per_model_predictions=predictions,
            )

        participating_keys = list(valid_predictions.keys())
        prob_matrix = []
        weight_list = []

        for m_id in participating_keys:
            raw_p = valid_predictions[m_id].probabilities
            validated_p = validate_probability_vector(raw_p, expected_dim=self.num_classes)
            prob_matrix.append(validated_p)

            # Assign weight
            w = float(self.weights.get(m_id, 1.0))
            if w <= 0:
                logger.warning(f"Invalid non-positive weight for model {m_id}: {w}. Reverting to 1.0.")
                w = 1.0
            weight_list.append(w)

        prob_matrix = np.array(prob_matrix, dtype=np.float64)  # (M, C)
        weights_arr = np.array(weight_list, dtype=np.float64)[:, np.newaxis]  # (M, 1)

        # Weighted or equal average
        ensemble_p = np.sum(prob_matrix * weights_arr, axis=0) / np.sum(weights_arr)
        ensemble_p = validate_probability_vector(ensemble_p.tolist(), expected_dim=self.num_classes)

        pred_idx = int(np.argmax(ensemble_p))
        pred_class = self.class_names[pred_idx]
        confidence = float(ensemble_p[pred_idx])
        margin = compute_top2_margin(ensemble_p)
        entropy = compute_predictive_entropy(ensemble_p)

        model_votes = {
            m_id: valid_predictions[m_id].predicted_idx
            for m_id in participating_keys
        }
        disagreement = compute_model_disagreement(pred_idx, model_votes)

        return EnsembleResult(
            predicted_idx=pred_idx,
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            predictive_entropy=round(entropy, 4),
            model_disagreement=round(disagreement, 4),
            participating_models=participating_keys,
            ensemble_probabilities=[round(float(v), 6) for v in ensemble_p],
            per_model_predictions=predictions,
        )
