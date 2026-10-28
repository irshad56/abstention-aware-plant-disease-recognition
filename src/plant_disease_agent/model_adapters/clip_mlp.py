"""CLIP ViT-B/32 Visual Encoder with Custom Trained MLP Adapter."""

from pathlib import Path
from typing import List, Optional
import zipfile
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.preprocessing import get_torch_transform
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class DiseaseMLP(nn.Module):
    def __init__(self, in_features: int = 512, hidden_features: int = 256, num_classes: int = 29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPMLPAdapter(BaseModelAdapter):
    def __init__(
        self,
        num_classes: int = 29,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        backbone_id: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__(
            model_id="clip_vit_b32",
            display_name="CLIP ViT-B/32 (MLP)",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.backbone_id = backbone_id
        self.mlp: Optional[DiseaseMLP] = None
        self.transform = get_torch_transform(
            resize_size=224,
            crop_size=224,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
            interpolation="bicubic",
        )

    def load(self, checkpoint_path: Path) -> bool:
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch / Transformers is not installed."
            return False

        p = Path(checkpoint_path)
        if not p.exists():
            self.load_error = f"Checkpoint file or archive does not exist: {p}"
            return False

        try:
            mlp = DiseaseMLP(in_features=512, hidden_features=256, num_classes=self.num_classes)

            # Check if checkpoint_path is a ZIP archive
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p, "r") as z:
                    # Search for mlp weights in zip
                    target_file = None
                    for name in z.namelist():
                        if "clip_mlp_weights.pth" in name or "adapted_mlp_weights.pth" in name:
                            target_file = name
                            break
                    if not target_file:
                        raise FileNotFoundError(f"Could not find MLP weights in zip archive: {p}")

                    with z.open(target_file) as f:
                        state = torch.load(f, map_location="cpu")
            else:
                state = torch.load(p, map_location="cpu")

            if isinstance(state, dict):
                # Clean up nested wrappers
                if "net" in state or any(k.startswith("net.") for k in state.keys()):
                    mlp_state = {k.replace("net.", ""): v for k, v in state.items()}
                    mlp.net.load_state_dict(mlp_state, strict=True)
                else:
                    mlp.load_state_dict(state, strict=False)
            else:
                mlp.load_state_dict(state)

            mlp.to(self.device).eval()
            self.mlp = mlp

            # Load vision backbone (check local extracted paths first)
            local_backbone_candidates = [
                Path("artifacts/models/clip_vit_b32/clip_final_artifacts/clip_frozen_mlp/clip_backbone"),
                Path("artifacts/models/clip_vit_b32/clip_frozen_mlp/clip_backbone"),
                Path("artifacts/models/clip_vit_b32/backbone"),
            ]
            backbone_path = self.backbone_id
            for cand in local_backbone_candidates:
                if (cand / "config.json").exists():
                    backbone_path = str(cand.resolve())
                    logger.info(f"Using local offline CLIP backbone from {backbone_path}")
                    break

            try:
                backbone = CLIPVisionModelWithProjection.from_pretrained(backbone_path)
                backbone.to(self.device).eval()
                self.model = backbone
            except Exception as b_err:
                logger.warning(f"Could not load CLIP backbone '{backbone_path}': {b_err}")
                self.model = None

            self.is_loaded = True
            self.checkpoint_path = p
            self.load_error = None
            logger.info(f"Loaded CLIP MLP from {p}")
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed loading CLIP MLP adapter from {p}: {e}", exc_info=True)
            return False

    def _forward(self, image: Image.Image) -> np.ndarray:
        if not TORCH_AVAILABLE or self.mlp is None:
            raise RuntimeError("CLIP MLP model is not loaded.")

        if self.model is None:
            raise RuntimeError(f"CLIP Vision backbone '{self.backbone_id}' is not loaded.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

            # Robust handling of Transformers return types (tensor or BaseModelOutputWithPooling)
            if hasattr(outputs, "image_embeds"):
                embeds = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                embeds = outputs
            else:
                # Extract pooled representation and apply visual projection
                pooled = getattr(outputs, "pooler_output", outputs[1] if len(outputs) > 1 else outputs[0])
                if hasattr(self.model, "visual_projection"):
                    embeds = self.model.visual_projection(pooled)
                else:
                    embeds = pooled

            # L2 Normalization
            normalized_embeds = F.normalize(embeds, p=2, dim=-1)

            logits = self.mlp(normalized_embeds)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return probs
