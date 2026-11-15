"""Authoritative Four-Level Inference Wrapper and Agent Safety Bridge Adapter.

Integrates the validated Level 1-4 research inference release:
  Level 1: Image Quality Assessment (IQA Gate)
  Level 2: Leaf vs Non-Leaf Verification (ResNet18 Domain-Adapted)
  Level 3: Supported Taxonomy & Out-of-Distribution Verification (LogisticRegression Probability Gate)
  Level 4: Diverse Five-Model Disease Ensemble (ConvNeXtV2_Tiny, DINOv2_Small, EfficientNetV2_S, MaxViT_Tiny, Swin_Tiny)
Followed by strict temperature calibration and Agent Safety Bridge routing to Groq scientific evidence retrieval.
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image

# Ensure release path is importable
RELEASE_DIR = Path(__file__).resolve().parent.parent.parent / "authoritative_release" / "final_inference_release"
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))

try:
    from plant_disease_inference import PlantDiseaseInference
    from agent_safety_bridge import prepare_agent_request
    AUTHORITATIVE_AVAILABLE = True
except Exception as e:
    PlantDiseaseInference = None
    prepare_agent_request = None
    AUTHORITATIVE_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class AuthoritativeInferenceWrapper:
    """Singleton wrapper around the authoritative PlantDiseaseInference engine."""

    _instance: Optional["AuthoritativeInferenceWrapper"] = None
    _engine: Optional[Any] = None

    def __init__(self, release_dir: Optional[Path] = None, device: Optional[str] = None):
        self.release_dir = Path(release_dir or RELEASE_DIR)
        self.device = device
        self.engine = None
        self.load_status = {}
        self._initialize()

    def _initialize(self):
        if not AUTHORITATIVE_AVAILABLE:
            self.load_status = {
                "engine": "FAILED",
                "error": f"Could not import authoritative release: {_IMPORT_ERROR}"
            }
            return

        checkpoints_dir = self.release_dir / "checkpoints"
        expected_models = [
            ("ConvNeXtV2_Tiny", "ConvNeXtV2_Tiny_best.pt"),
            ("DINOv2_Small", "DINOv2_Small_best.pt"),
            ("EfficientNetV2_S", "EfficientNetV2_S_best.pt"),
            ("MaxViT_Tiny", "MaxViT_Tiny_best.pt"),
            ("Swin_Tiny", "Swin_Tiny_best.pt"),
            ("leaf_verifier", "leaf_verifier_resnet18_domain_adapted.pth"),
        ]

        # Check file existence
        missing = []
        for name, ckpt in expected_models:
            p = checkpoints_dir / ckpt
            if not p.exists():
                missing.append(ckpt)

        if missing:
            self.load_status = {
                "engine": "NOT_FOUND",
                "error": f"Missing checkpoints in {checkpoints_dir}: {missing}"
            }
            return

        try:
            self.engine = PlantDiseaseInference(
                release_dir=str(self.release_dir),
                device=self.device
            )
            self.load_status = {
                "engine": "LOADED",
                "models": [m["name"] for m in self.engine.models],
                "joint_classes": len(self.engine.joint_classes),
                "adapter_classes": len(self.engine.adapter_classes),
                "taxonomy_threshold": self.engine.taxonomy_threshold,
                "disease_threshold": self.engine.disease_threshold,
                "leaf_threshold": self.engine.leaf_threshold,
                "temperature": self.engine.temperature,
            }
        except Exception as e:
            self.load_status = {
                "engine": "LOADED_FAILED",
                "error": str(e)
            }

    @property
    def is_ready(self) -> bool:
        return self.engine is not None and self.load_status.get("engine") == "LOADED"

    def predict(self, image: Union[Image.Image, str, bytes, Path]) -> Dict[str, Any]:
        """Executes full 4-level inference pipeline."""
        if not self.is_ready:
            return {
                "status": "ABSTAIN",
                "stage": "MODEL_UNAVAILABLE",
                "reason": ["AUTHORITATIVE_MODELS_NOT_LOADED", self.load_status.get("error", "")],
                "disease_prediction": None,
                "disease_specific_evidence_allowed": False,
            }

        if isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, (str, Path)):
            image = Image.open(image)

        # Run 4-level engine
        raw_result = self.engine.predict(image)

        # Apply strict Agent Safety Bridge
        bridge_decision = prepare_agent_request(raw_result)
        raw_result["agent_bridge"] = bridge_decision

        return raw_result
