"""Model 5: CLIP ViT-B/32 image encoder (frozen) + saved MLP classifier.

Feature pipeline (explicit, independent of get_image_features() behaviour):

    CLIP image processor -> vision_model(...).pooler_output
        -> visual_projection -> L2 normalize -> MLP -> logits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from plant_disease.config import get_logger
from plant_disease.models.base import (
    ModelAdapter,
    ModelLoadError,
    extract_state_dict,
    load_state_dict_strict,
    read_checkpoint,
)

log = get_logger("models.clip")

WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin")
MLP_STATE_KEYS = (
    "mlp_state_dict", "classifier_state_dict", "head_state_dict",
    "model_state_dict", "state_dict", "mlp",
)
MLP_PREFIXES = ("", "mlp.", "head.", "classifier.", "net.", "model.", "module.")
DROPOUT = 0.20   # inactive in eval mode


def build_mlp(embed_dim: int, hidden: int, num_classes: int) -> nn.Sequential:
    """Layout described in the project notes. Strict loading confirms it."""
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(hidden, num_classes),
    )


def find_mlp_file(mlp_dir: Path) -> Path:
    """Locate the MLP checkpoint written by `setup_artifacts.py extract`."""
    mlp_dir = Path(mlp_dir)
    summary = mlp_dir.parent / "metadata" / "extraction_summary.json"
    if summary.is_file():
        try:
            name = json.loads(summary.read_text(encoding="utf-8")).get("mlp_file")
        except (OSError, json.JSONDecodeError):
            name = None
        if name and (mlp_dir / name).is_file():
            return mlp_dir / name
    files = (
        sorted(p for p in mlp_dir.glob("*") if p.suffix in WEIGHT_SUFFIXES)
        if mlp_dir.is_dir() else []
    )
    if len(files) == 1:
        return files[0]
    raise ModelLoadError(
        f"Could not identify the CLIP MLP checkpoint in {mlp_dir} "
        f"({len(files)} candidate files). Run: python scripts\\setup_artifacts.py extract"
    )


def read_mlp_state_dict(path: Path) -> Dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelLoadError("safetensors is not installed.") from exc
        return dict(load_file(str(path), device="cpu"))
    return extract_state_dict(read_checkpoint(path), "CLIP MLP", keys=MLP_STATE_KEYS)


def infer_hidden_dim(state_dict: Dict[str, Any], embed_dim: int, num_classes: int) -> int:
    """Hidden width = the single (hidden, embed_dim) weight that is not the output layer."""
    candidates = [
        int(v.shape[0]) for v in state_dict.values()
        if v.ndim == 2 and v.shape[1] == embed_dim and v.shape[0] != num_classes
    ]
    if len(candidates) != 1:
        shapes = {k: list(v.shape) for k, v in list(state_dict.items())[:12]}
        raise ModelLoadError(
            f"CLIP MLP: cannot infer hidden size from tensors: {shapes}. "
            "Run: python scripts\\setup_artifacts.py inspect"
        )
    return candidates[0]


def align_mlp_keys(state_dict: Dict[str, Any], target_keys: List[str]) -> Tuple[Dict[str, Any], str]:
    """Strip a wrapper prefix (e.g. 'mlp.') if that makes keys match the MLP."""
    target = set(target_keys)
    best, best_prefix, best_score = state_dict, "", -1
    for prefix in MLP_PREFIXES:
        if prefix and not all(k.startswith(prefix) for k in state_dict):
            continue
        stripped = {k[len(prefix):]: v for k, v in state_dict.items()}
        score = sum(1 for k in stripped if k in target)
        if score > best_score:
            best, best_prefix, best_score = stripped, prefix, score
    return best, best_prefix


class ClipMlpClassifier(nn.Module):
    """CLIP vision tower + projection + L2 norm + MLP."""

    def __init__(self, clip: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.clip = clip
        self.mlp = mlp

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.clip.vision_model(pixel_values=pixel_values).pooler_output
        projected = self.clip.visual_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(pixel_values))


class ClipMlpAdapter(ModelAdapter):
    """Loads the CLIP backbone (bundled or cached, never downloaded here) and the MLP."""

    def _backbone_sources(self) -> List[Tuple[str, Dict[str, Any], str]]:
        sources: List[Tuple[str, Dict[str, Any], str]] = []
        bundled = self.spec.get("backbone_dir")
        if bundled and (Path(bundled) / "config.json").is_file():
            sources.append((str(bundled), {}, "bundled with the CLIP archive"))
        repo = self.spec["backbone"]
        if self.cache_dir:
            sources.append((repo, {"cache_dir": str(self.cache_dir)}, "project cache"))
        sources.append((repo, {}, "default Hugging Face cache"))
        return sources

    def _load(self) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise ModelLoadError("transformers is not installed.") from exc

        clip = None
        origin = ""
        source_used: Optional[Tuple[str, Dict[str, Any]]] = None
        errors: List[str] = []
        for source, kwargs, label in self._backbone_sources():
            try:
                clip = CLIPModel.from_pretrained(source, local_files_only=True, **kwargs)
                origin, source_used = label, (source, kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:120]}")
        if clip is None or source_used is None:
            raise ModelLoadError(
                "CLIP backbone not available locally.\n"
                "Run:  python scripts\\download_backbones.py\n"
                f"Tried: {errors}"
            )
        embed_dim = int(clip.config.projection_dim)

        try:
            self._processor = CLIPImageProcessor.from_pretrained(
                source_used[0], local_files_only=True, **source_used[1]
            )
            processor_note = f"CLIP processor loaded from {origin}"
        except Exception:  # noqa: BLE001
            self._processor = CLIPImageProcessor()
            processor_note = "CLIP processor files not found; library default settings used"

        mlp_path = find_mlp_file(self.spec["mlp_dir"])
        mlp_sd = read_mlp_state_dict(mlp_path)
        hidden = infer_hidden_dim(mlp_sd, embed_dim, self.num_classes)
        mlp = build_mlp(embed_dim, hidden, self.num_classes)
        aligned, prefix = align_mlp_keys(mlp_sd, list(mlp.state_dict().keys()))
        load_state_dict_strict(mlp, aligned, f"{self.display_name} MLP")

        self.model = self._finalize(ClipMlpClassifier(clip, mlp))
        self.notes.append(f"backbone from {origin}; MLP {embed_dim}->{hidden}->{self.num_classes}")
        self.notes.append(processor_note)
        if prefix:
            self.notes.append(f"MLP keys had prefix '{prefix}' (stripped)")
        self.notes.append("training/inference preprocessing parity not verified")

    def _prepare_batch(self, images: List[Image.Image]) -> Any:
        pixel_values = self._processor(
            images=[img.convert("RGB") for img in images], return_tensors="pt"
        )["pixel_values"]
        return pixel_values.to(self.device)

    def _forward(self, images: List[Image.Image]) -> Any:
        return self.model(self._prepare_batch(images))
