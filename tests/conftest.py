"""Pytest fixtures for unit and integration testing without requiring GPU or large weights."""

import io
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
from PIL import Image

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plant_disease_agent.config import AppConfig
from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.schemas import ModelPrediction


class MockModelAdapter(BaseModelAdapter):
    """Deterministic mock adapter for fast, reproducible unit tests."""
    def __init__(
        self,
        model_id: str,
        display_name: str,
        num_classes: int = 29,
        target_idx: int = 0,
        target_confidence: float = 0.95,
        should_fail: bool = False,
    ):
        super().__init__(
            model_id=model_id,
            display_name=display_name,
            num_classes=num_classes,
        )
        self.target_idx = target_idx
        self.target_confidence = target_confidence
        self.should_fail = should_fail
        self.is_loaded = not should_fail
        self.model = "mock_model_loaded" if not should_fail else None

    def load(self, checkpoint_path: Path) -> bool:
        if self.should_fail:
            self.is_loaded = False
            self.load_error = "Mock load failure."
            return False
        self.is_loaded = True
        return True

    def _forward(self, image: Image.Image) -> np.ndarray:
        if self.should_fail:
            raise RuntimeError("Mock forward execution failure.")

        probs = np.full(self.num_classes, (1.0 - self.target_confidence) / (self.num_classes - 1), dtype=np.float32)
        probs[self.target_idx] = self.target_confidence
        return probs


@pytest.fixture
def mock_class_names() -> List[str]:
    return [
        "Apple__Apple Scab",
        "Apple__Black Rot",
        "Apple__Cedar Apple Rust",
        "Apple__Healthy",
        "Bell Pepper__Bacterial Spot",
        "Bell Pepper__Healthy",
        "Cherry__Healthy",
        "Cherry__Powdery Mildew",
        "Corn (Maize)__Cercospora Leaf Spot",
        "Corn (Maize)__Common Rust ",
        "Corn (Maize)__Healthy",
        "Corn (Maize)__Northern Leaf Blight",
        "Grape__Black Rot",
        "Grape__Esca (Black Measles)",
        "Grape__Healthy",
        "Grape__Leaf Blight",
        "Peach__Bacterial Spot",
        "Peach__Healthy",
        "Potato__Early Blight",
        "Potato__Healthy",
        "Potato__Late Blight",
        "Strawberry__Healthy",
        "Strawberry__Leaf Scorch",
        "Tomato__Bacterial Spot",
        "Tomato__Early Blight",
        "Tomato__Healthy",
        "Tomato__Late Blight",
        "Tomato__Septoria Leaf Spot",
        "Tomato__Yellow Leaf Curl Virus",
    ]


@pytest.fixture
def mock_adapters(mock_class_names) -> Dict[str, BaseModelAdapter]:
    return {
        "convnext_tiny": MockModelAdapter("convnext_tiny", "ConvNeXt-Tiny", target_idx=0, target_confidence=0.92),
        "efficientnetv2_b0": MockModelAdapter("efficientnetv2_b0", "EfficientNetV2-B0", target_idx=0, target_confidence=0.94),
        "dinov2_linear": MockModelAdapter("dinov2_linear", "DINOv2-Small (Linear)", target_idx=0, target_confidence=0.88),
        "dinov2_finetuned": MockModelAdapter("dinov2_finetuned", "DINOv2-Small (Fine-Tuned)", target_idx=0, target_confidence=0.96),
        "clip_vit_b32": MockModelAdapter("clip_vit_b32", "CLIP ViT-B/32 (MLP)", target_idx=0, target_confidence=0.90),
    }


@pytest.fixture
def valid_leaf_image() -> Image.Image:
    # 224x224 green image with sufficient contrast/texture
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 160  # Green channel
    # Add simulated texture variation
    arr[::4, ::4, 0] = 50
    arr[1::4, 1::4, 2] = 40
    return Image.fromarray(arr)


@pytest.fixture
def valid_image_bytes(valid_leaf_image) -> bytes:
    buf = io.BytesIO()
    valid_leaf_image.save(buf, format="JPEG")
    return buf.getvalue()
