"""DINOv2-Small Frozen Backbone with Trained Linear Classifier."""

from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.preprocessing import get_torch_transform
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class DINOv2LinearAdapter(BaseModelAdapter):
    def __init__(
        self,
        num_classes: int = 29,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        backbone_id: str = "facebook/dinov2-small",
    ):
        super().__init__(
            model_id="dinov2_linear",
            display_name="DINOv2-Small (Linear Head)",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.backbone_id = backbone_id
        self.feature_dim = 384
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.classifier_head: Optional[nn.Linear] = None

        self.transform = get_torch_transform(
            resize_size=518,
            crop_size=518,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            interpolation="bicubic",
        )

    def load(self, checkpoint_path: Path) -> bool:
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch / Transformers is not installed."
            return False

        p = Path(checkpoint_path)
        if not p.exists():
            self.load_error = f"Checkpoint file does not exist: {p}"
            return False

        try:
            # Load checkpoint
            ckpt = torch.load(p, map_location="cpu")

            # Checkpoint can either be:
            # 1) dict with model_state_dict, feature_mean, feature_std
            # 2) direct state_dict of the linear layer (weight (29, 384), bias (29))
            if isinstance(ckpt, dict) and "feature_mean" in ckpt and "feature_std" in ckpt:
                self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
                self.feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
                # Safeguard against zero std
                self.feature_std = np.where(self.feature_std <= 1e-7, 1.0, self.feature_std)
                head_state = ckpt.get("model_state_dict", ckpt)
            else:
                self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
                self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
                head_state = ckpt

            # Linear classifier head (384 -> 29)
            head = nn.Linear(self.feature_dim, self.num_classes)
            if isinstance(head_state, dict):
                # Check if keys are weight/bias or classifier.weight/classifier.bias
                cleaned_head = {}
                for k, v in head_state.items():
                    k_clean = k.replace("classifier.", "").replace("head.", "").replace("fc.", "")
                    if k_clean in ("weight", "bias"):
                        cleaned_head[k_clean] = v
                if "weight" in cleaned_head:
                    head.load_state_dict(cleaned_head, strict=True)
                else:
                    head.load_state_dict(head_state, strict=False)

            head.to(self.device)
            head.eval()
            self.classifier_head = head

            # Instantiate DINOv2 backbone
            try:
                backbone = AutoModel.from_pretrained(self.backbone_id)
                backbone.to(self.device)
                backbone.eval()
                self.model = backbone
            except Exception as b_err:
                logger.warning(f"Could not load online backbone '{self.backbone_id}': {b_err}. Head is loaded.")
                self.model = None

            self.is_loaded = True
            self.checkpoint_path = p
            self.load_error = None
            logger.info(f"Loaded DINOv2 Linear head from {p}")
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed loading DINOv2 Linear model from {p}: {e}", exc_info=True)
            return False

    def _forward(self, image: Image.Image) -> np.ndarray:
        if not TORCH_AVAILABLE or self.classifier_head is None:
            raise RuntimeError("Classifier head is not loaded.")

        if self.model is None:
            raise RuntimeError(
                f"Backbone '{self.backbone_id}' is not loaded. Ensure network access to HuggingFace or cached weights."
            )

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            # CLS token is at index 0 of last_hidden_state
            if hasattr(outputs, "last_hidden_state"):
                cls_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_feature = outputs.pooler_output.cpu().numpy()
            else:
                cls_feature = outputs[0][:, 0, :].cpu().numpy()

            # Standardize feature
            if self.feature_mean is not None and self.feature_std is not None:
                cls_feature = (cls_feature - self.feature_mean) / self.feature_std

            feat_tensor = torch.tensor(cls_feature, dtype=torch.float32, device=self.device)
            logits = self.classifier_head(feat_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return probs
