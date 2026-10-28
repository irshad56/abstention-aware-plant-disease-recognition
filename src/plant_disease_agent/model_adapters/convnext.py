"""ConvNeXt-Tiny model adapter."""

from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision.models import convnext_tiny
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.preprocessing import get_torch_transform, preprocess_numpy
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class ConvNeXtAdapter(BaseModelAdapter):
    def __init__(
        self,
        num_classes: int = 29,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        super().__init__(
            model_id="convnext_tiny",
            display_name="ConvNeXt-Tiny",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.transform = get_torch_transform(
            resize_size=236,
            crop_size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            interpolation="bicubic",
        )

    def load(self, checkpoint_path: Path) -> bool:
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch / torchvision is not installed."
            return False

        p = Path(checkpoint_path)
        if not p.exists():
            self.load_error = f"Checkpoint file does not exist: {p}"
            return False

        try:
            # Instantiate architecture without pretrained weights
            model = convnext_tiny(weights=None)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, self.num_classes)

            # Load checkpoint safely
            state = torch.load(p, map_location=self.device)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state_dict = state["model_state_dict"]
                elif "state_dict" in state:
                    state_dict = state["state_dict"]
                elif "model" in state:
                    state_dict = state["model"]
                else:
                    state_dict = state
            else:
                state_dict = state

            # Strip module. prefix if trained under DataParallel
            cleaned_state = {}
            for k, v in state_dict.items():
                new_k = k[7:] if k.startswith("module.") else k
                cleaned_state[new_k] = v

            model.load_state_dict(cleaned_state, strict=True)
            model.to(self.device)
            model.eval()

            self.model = model
            self.is_loaded = True
            self.checkpoint_path = p
            self.load_error = None
            logger.info(f"Loaded ConvNeXt-Tiny strictly from {p} on {self.device}")
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed to load ConvNeXt-Tiny from {p}: {e}", exc_info=True)
            return False

    def _forward(self, image: Image.Image) -> np.ndarray:
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("PyTorch is required for ConvNeXt inference.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs
