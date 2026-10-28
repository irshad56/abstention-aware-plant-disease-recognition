"""DINOv2-Small Fine-Tuned Backbone with Classifier."""

from pathlib import Path
from typing import Dict, List, Optional
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


def remap_dinov2_attention_keys(
    state_dict: Dict[str, torch.Tensor],
    target_keys: List[str]
) -> Dict[str, torch.Tensor]:
    """Dynamically aligns attention keys between HuggingFace transformers and timm/torch formats."""
    remapped = {}
    target_set = set(target_keys)

    # Common renamings for 96 attention projections (12 layers * 4 projections [q, k, v, o] * 2 [weight, bias])
    key_aliases = [
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".q_proj.", ".attention.attention.query."),
        (".k_proj.", ".attention.attention.key."),
        (".v_proj.", ".attention.attention.value."),
        (".o_proj.", ".attention.output.dense."),
    ]

    for k, v in state_dict.items():
        k_clean = k[7:] if k.startswith("module.") else k
        if k_clean in target_set:
            remapped[k_clean] = v
            continue

        matched = False
        for src_sub, tgt_sub in key_aliases:
            candidate = k_clean.replace(src_sub, tgt_sub)
            if candidate in target_set:
                remapped[candidate] = v
                matched = True
                break

        if not matched:
            remapped[k_clean] = v

    return remapped


class DINOv2FineTunedAdapter(BaseModelAdapter):
    def __init__(
        self,
        num_classes: int = 29,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        backbone_id: str = "facebook/dinov2-small",
    ):
        super().__init__(
            model_id="dinov2_finetuned",
            display_name="DINOv2-Small (Fine-Tuned)",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.backbone_id = backbone_id
        self.feature_dim = 384
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
            ckpt = torch.load(p, map_location="cpu")
            backbone = AutoModel.from_pretrained(self.backbone_id)
            target_keys = list(backbone.state_dict().keys())

            head = nn.Linear(self.feature_dim, self.num_classes)

            # Check structure
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                b_state = ckpt["model_state_dict"]
                c_state = ckpt.get("classifier_state_dict", {})
            else:
                b_state = ckpt
                c_state = {}

            # Remap attention keys
            aligned_b_state = remap_dinov2_attention_keys(b_state, target_keys)
            missing, unexpected = backbone.load_state_dict(aligned_b_state, strict=False)
            logger.info(f"DINOv2 Fine-Tuned loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

            # Load classifier
            if c_state:
                cleaned_c = {k.replace("classifier.", ""): v for k, v in c_state.items()}
                head.load_state_dict(cleaned_c, strict=False)
            else:
                # Check if head weights are inside b_state
                head_dict = {
                    "weight": b_state.get("classifier.weight", head.weight),
                    "bias": b_state.get("classifier.bias", head.bias),
                }
                head.load_state_dict(head_dict, strict=False)

            backbone.to(self.device).eval()
            head.to(self.device).eval()

            self.model = backbone
            self.classifier_head = head
            self.is_loaded = True
            self.checkpoint_path = p
            self.load_error = None
            logger.info(f"Loaded DINOv2 Fine-Tuned strictly/aligned from {p}")
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed loading DINOv2 Fine-Tuned from {p}: {e}", exc_info=True)
            return False

    def _forward(self, image: Image.Image) -> np.ndarray:
        if not TORCH_AVAILABLE or self.model is None or self.classifier_head is None:
            raise RuntimeError("DINOv2 Fine-Tuned model or head is not loaded.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state[:, 0, :]
            else:
                feat = outputs[0][:, 0, :]

            logits = self.classifier_head(feat)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return probs
