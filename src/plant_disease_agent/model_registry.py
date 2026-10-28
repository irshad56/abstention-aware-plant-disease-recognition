"""Model Registry: discovers candidate checkpoint paths and manages adapter lifecycles."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from plant_disease_agent.config import AppConfig
from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.model_adapters.convnext import ConvNeXtAdapter
from plant_disease_agent.model_adapters.efficientnetv2 import EfficientNetV2Adapter
from plant_disease_agent.model_adapters.dinov2_linear import DINOv2LinearAdapter
from plant_disease_agent.model_adapters.dinov2_finetuned import DINOv2FineTunedAdapter
from plant_disease_agent.model_adapters.clip_mlp import CLIPMLPAdapter
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelRegistry:
    def __init__(self, config: AppConfig):
        self.config = config
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        class_names = self.config.class_names
        num_classes = len(class_names)
        from plant_disease_agent.utils import resolve_device
        device = resolve_device(self.config.device)

        self.adapters["convnext_tiny"] = ConvNeXtAdapter(
            num_classes=num_classes, class_names=class_names, device=device
        )
        self.adapters["efficientnetv2_b0"] = EfficientNetV2Adapter(
            num_classes=num_classes, class_names=class_names, device=device
        )
        self.adapters["dinov2_linear"] = DINOv2LinearAdapter(
            num_classes=num_classes, class_names=class_names, device=device
        )
        self.adapters["dinov2_finetuned"] = DINOv2FineTunedAdapter(
            num_classes=num_classes, class_names=class_names, device=device
        )
        self.adapters["clip_vit_b32"] = CLIPMLPAdapter(
            num_classes=num_classes, class_names=class_names, device=device
        )

    def discover_and_load_all(self) -> Dict[str, Dict[str, Any]]:
        """Scans candidate paths for all 5 models and attempts to load found checkpoints."""
        status_report = {}

        for model_id, adapter in self.adapters.items():
            cfg = self.config.get_model_config(model_id)
            candidates = cfg.get("candidate_paths", [])

            found_path: Optional[Path] = None
            # Check configured model root first
            root = Path(self.config.model_root)
            for c in candidates:
                cand_path = root / c
                if cand_path.exists():
                    found_path = cand_path
                    break
                # Also check relative to current working directory
                direct_path = Path(c)
                if direct_path.exists():
                    found_path = direct_path
                    break

            if found_path is None:
                status_report[model_id] = {
                    "status": "NOT_FOUND",
                    "path": None,
                    "error": "No matching checkpoint found in candidate paths.",
                    "display_name": adapter.display_name,
                }
                continue

            # Attempt loading
            success = adapter.load(found_path)
            status_report[model_id] = {
                "status": "LOADED" if success else "FAILED",
                "path": str(found_path),
                "error": adapter.load_error,
                "display_name": adapter.display_name,
            }

        return status_report

    def get_loaded_adapters(self) -> Dict[str, BaseModelAdapter]:
        return {m_id: adapter for m_id, adapter in self.adapters.items() if adapter.is_loaded}
