"""Inference engine running parallel/sequential model evaluation."""

from typing import Dict
from PIL import Image

from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.schemas import ModelPrediction
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class InferenceManager:
    def __init__(self, adapters: Dict[str, BaseModelAdapter]):
        self.adapters = adapters

    def run_all(self, image: Image.Image) -> Dict[str, ModelPrediction]:
        """Runs inference sequentially across all registered model adapters."""
        predictions: Dict[str, ModelPrediction] = {}

        for m_id, adapter in self.adapters.items():
            try:
                pred = adapter.predict(image)
                predictions[m_id] = pred
            except Exception as e:
                logger.error(f"Unexpected exception during {adapter.display_name} inference: {e}", exc_info=True)
                predictions[m_id] = ModelPrediction(
                    model_id=m_id,
                    display_name=adapter.display_name,
                    predicted_idx=0,
                    predicted_class=adapter.class_names[0],
                    confidence=0.0,
                    probabilities=[0.0] * adapter.num_classes,
                    latency_ms=0.0,
                    status="failed",
                    error_message=str(e),
                )

        return predictions
