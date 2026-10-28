"""Abstract base class for all 5 deep learning model adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from plant_disease_agent.schemas import ModelPrediction
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class BaseModelAdapter(ABC):
    def __init__(
        self,
        model_id: str,
        display_name: str,
        num_classes: int = 29,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.display_name = display_name
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        from plant_disease_agent.utils import resolve_device
        self.device = resolve_device(device)
        self.model: Any = None
        self.is_loaded = False
        self.checkpoint_path: Optional[Path] = None
        self.load_error: Optional[str] = None

    @abstractmethod
    def load(self, checkpoint_path: Path) -> bool:
        """Loads weights, validates state dict keys and dimensions, and places in eval mode."""
        pass

    @abstractmethod
    def _forward(self, image: Image.Image) -> np.ndarray:
        """Performs model inference and returns a 1D probability distribution of shape (num_classes,)."""
        pass

    def predict(self, image: Image.Image) -> ModelPrediction:
        """Executes forward pass with latency tracking and exception isolation."""
        if not self.is_loaded or self.model is None:
            return ModelPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                predicted_idx=0,
                predicted_class=self.class_names[0],
                confidence=0.0,
                probabilities=[0.0] * self.num_classes,
                latency_ms=0.0,
                status="unloaded",
                error_message=f"Model {self.display_name} is not loaded. Error: {self.load_error}",
            )

        start_time = time.perf_counter()
        try:
            probs = self._forward(image)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if len(probs) != self.num_classes:
                raise ValueError(f"Output shape mismatch: expected {self.num_classes}, got {len(probs)}")

            pred_idx = int(np.argmax(probs))
            pred_class = self.class_names[pred_idx]
            confidence = float(probs[pred_idx])

            return ModelPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                predicted_idx=pred_idx,
                predicted_class=pred_class,
                confidence=round(confidence, 4),
                probabilities=[round(float(p), 6) for p in probs],
                latency_ms=round(latency_ms, 2),
                status="success",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Prediction failed for model {self.display_name}: {e}", exc_info=True)
            return ModelPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                predicted_idx=0,
                predicted_class=self.class_names[0],
                confidence=0.0,
                probabilities=[0.0] * self.num_classes,
                latency_ms=round(latency_ms, 2),
                status="failed",
                error_message=str(e),
            )

    def unload(self):
        """Releases model from memory."""
        self.model = None
        self.is_loaded = False
        self.load_error = None
