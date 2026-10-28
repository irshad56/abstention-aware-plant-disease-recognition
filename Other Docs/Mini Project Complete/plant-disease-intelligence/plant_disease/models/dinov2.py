"""DINOv2-Small adapters: (3) frozen backbone + standardized linear head,
(4) fine-tuned backbone + separate classifier head.

The two models are different and are kept in separate classes:

  DinoV2LinearAdapter     image -> backbone (pretrained, from local cache)
                          -> CLS feature -> (feature - mean) / std -> Linear
  DinoV2FinetunedAdapter  image -> backbone (fine-tuned weights from the
                          checkpoint) -> CLS feature -> Linear

The CLS feature is last_hidden_state[:, 0] (after the backbone's final
layer norm), identical across Transformers versions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from plant_disease.config import get_logger
from plant_disease.models.base import (
    ModelAdapter,
    ModelLoadError,
    load_state_dict_strict,
    read_checkpoint,
    strip_prefix,
)

log = get_logger("models.dinov2")

HIDDEN_DIM = 384                 # dinov2-small
EXPECTED_ATTENTION_REMAPS = 96   # 12 layers x (q,k,v,o) x (weight,bias), per project notes
STAT_KEYS = ("feature_mean", "feature_std")

# Attention layer names differ between Transformers versions:
#   older: encoder.layer.N.attention.attention.query / key / value, attention.output.dense
#   newer: encoder.layer.N.attention.q_proj / k_proj / v_proj / o_proj
REMAP_RULES: List[Tuple[str, str]] = [
    ("attention.attention.query", "attention.q_proj"),
    ("attention.attention.key", "attention.k_proj"),
    ("attention.attention.value", "attention.v_proj"),
    ("attention.output.dense", "attention.o_proj"),
]
PREFIX_CANDIDATES = ("", "module.", "dinov2.", "backbone.", "model.", "base_model.", "encoder_model.")


# ==========================================================================
# Key alignment (explicit, testable remapping)
# ==========================================================================
def _rename_key(key: str, reverse: bool) -> str:
    for old, new in REMAP_RULES:
        src, dst = (new, old) if reverse else (old, new)
        if src in key:
            return key.replace(src, dst)
    return key


def align_state_dict(
    state_dict: Dict[str, Any], target_keys: Iterable[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rename checkpoint keys so they match the installed model's keys.

    Tries stripping a common prefix and renaming attention layers in either
    direction (old->new or new->old), and keeps the option that matches the
    most target keys. The target model's own keys decide, so this works
    whichever Transformers version is installed.

    Returns (aligned_state_dict, info). info has: prefix_stripped, direction
    ('none' | 'old_to_new' | 'new_to_old'), remapped_keys, matched, total.
    """
    target = set(target_keys)
    best: Optional[Tuple[Tuple[int, int], Dict[str, Any], Dict[str, Any]]] = None

    for prefix in PREFIX_CANDIDATES:
        if prefix and not all(k.startswith(prefix) for k in state_dict):
            continue
        stripped = {k[len(prefix):]: v for k, v in state_dict.items()}
        for direction, reverse in (("none", None), ("old_to_new", False), ("new_to_old", True)):
            renamed: Dict[str, Any] = {}
            remapped = 0
            for key, value in stripped.items():
                new_key = key
                if reverse is not None and key not in target:
                    new_key = _rename_key(key, reverse)
                if new_key != key:
                    remapped += 1
                renamed[new_key] = value
            if len(renamed) != len(stripped):      # name collision: reject this option
                continue
            matched = sum(1 for k in renamed if k in target)
            score = (matched, -remapped)
            info = {
                "prefix_stripped": prefix,
                "direction": direction,
                "remapped_keys": remapped,
                "matched": matched,
                "total": len(renamed),
            }
            if best is None or score > best[0]:
                best = (score, renamed, info)

    if best is None:  # pragma: no cover - the "" prefix always qualifies
        raise ModelLoadError("Could not align checkpoint keys with the model.")
    return best[1], best[2]


# ==========================================================================
# Loading helpers
# ==========================================================================
def load_dinov2_backbone(repo_id: str, cache_dir: Optional[Any]) -> nn.Module:
    """Load the pretrained DINOv2 backbone from local files only (no network)."""
    try:
        from transformers import Dinov2Model
    except ImportError as exc:
        raise ModelLoadError("transformers is not installed.") from exc

    attempts: List[Dict[str, Any]] = []
    if cache_dir:
        attempts.append({"cache_dir": str(cache_dir)})
    attempts.append({})                          # default Hugging Face cache
    errors: List[str] = []
    for kwargs in attempts:
        try:
            backbone = Dinov2Model.from_pretrained(repo_id, local_files_only=True, **kwargs)
            if backbone.config.hidden_size != HIDDEN_DIM:
                raise ModelLoadError(
                    f"{repo_id} has hidden size {backbone.config.hidden_size}, "
                    f"expected {HIDDEN_DIM}."
                )
            return backbone
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
    raise ModelLoadError(
        f"DINOv2 backbone '{repo_id}' is not available locally.\n"
        "Download it once with:  python scripts\\download_backbones.py\n"
        f"Details: {errors[-1] if errors else 'n/a'}"
    )


def find_linear_head(
    state_dict: Dict[str, Any], in_dim: int, num_classes: int, label: str
) -> Tuple[str, str]:
    """Locate the (num_classes, in_dim) weight and (num_classes,) bias by shape.

    Fails if the head is ambiguous or the state dict holds anything else, so
    unexpected content (for example backbone weights) is never ignored.
    """
    weight_keys = [
        k for k, v in state_dict.items()
        if v.ndim == 2 and tuple(v.shape) == (num_classes, in_dim)
    ]
    bias_keys = [
        k for k, v in state_dict.items()
        if v.ndim == 1 and tuple(v.shape) == (num_classes,)
    ]
    if len(weight_keys) != 1 or len(bias_keys) != 1:
        shapes = {k: list(v.shape) for k, v in list(state_dict.items())[:12]}
        raise ModelLoadError(
            f"{label}: expected one ({num_classes}, {in_dim}) weight and one "
            f"({num_classes},) bias; found weights={weight_keys}, biases={bias_keys}.\n"
            f"Tensors present: {shapes}"
        )
    w_key, b_key = weight_keys[0], bias_keys[0]
    if w_key.rpartition(".")[0] != b_key.rpartition(".")[0]:
        raise ModelLoadError(f"{label}: weight '{w_key}' and bias '{b_key}' belong to different layers.")
    extras = [k for k in state_dict if k not in (w_key, b_key) and k not in STAT_KEYS]
    if extras:
        raise ModelLoadError(
            f"{label}: unexpected extra tensors besides the linear head: {extras[:8]}"
            f" ({len(extras)} total). Run: python scripts\\setup_artifacts.py inspect"
        )
    return w_key, b_key


def _as_vector(value: Any, dim: int, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    tensor = tensor.detach().float().flatten()
    if tensor.numel() != dim:
        raise ModelLoadError(f"{name} has {tensor.numel()} values, expected {dim}.")
    if not torch.isfinite(tensor).all():
        raise ModelLoadError(f"{name} contains non-finite values.")
    return tensor


def _build_linear(state_dict: Dict[str, Any], w_key: str, b_key: str, num_classes: int) -> nn.Linear:
    head = nn.Linear(HIDDEN_DIM, num_classes)
    load_state_dict_strict(head, {"weight": state_dict[w_key], "bias": state_dict[b_key]}, "linear head")
    return head


def _historical_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in checkpoint.items():
        name = str(key).lower()
        if any(w in name for w in ("epoch", "acc", "loss")) and isinstance(value, (int, float, str)):
            out[str(key)] = value
    return out


# ==========================================================================
# Combined module
# ==========================================================================
class DinoClassifier(nn.Module):
    """backbone -> CLS feature -> [feature standardization] -> linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.standardize = feature_mean is not None and feature_std is not None
        if self.standardize:
            self.register_buffer("feature_mean", feature_mean.view(1, -1))
            self.register_buffer("feature_std", feature_std.view(1, -1))

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.features(pixel_values)
        if self.standardize:                      # feature standardization, NOT image normalization
            feats = (feats - self.feature_mean) / self.feature_std
        return self.head(feats)


# ==========================================================================
# Adapters
# ==========================================================================
class DinoV2LinearAdapter(ModelAdapter):
    """Model 3: pretrained DINOv2-Small + standardized linear classifier."""

    def _load(self) -> None:
        backbone = load_dinov2_backbone(self.spec["backbone"], self.cache_dir)
        checkpoint = read_checkpoint(self.spec["checkpoint"])
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            keys = list(checkpoint)[:10] if isinstance(checkpoint, dict) else type(checkpoint).__name__
            raise ModelLoadError(f"{self.display_name}: 'model_state_dict' missing. Found: {keys}")
        state_dict = strip_prefix(checkpoint["model_state_dict"])

        mean = checkpoint.get("feature_mean", state_dict.get("feature_mean"))
        std = checkpoint.get("feature_std", state_dict.get("feature_std"))
        if mean is None or std is None:
            raise ModelLoadError(
                f"{self.display_name}: feature_mean/feature_std missing from the checkpoint. "
                "Refusing to run without feature standardization."
            )
        mean_t = _as_vector(mean, HIDDEN_DIM, "feature_mean")
        std_t = _as_vector(std, HIDDEN_DIM, "feature_std")
        if not bool((std_t > 0).all()):
            raise ModelLoadError(f"{self.display_name}: feature_std has zero or negative entries.")

        w_key, b_key = find_linear_head(state_dict, HIDDEN_DIM, self.num_classes, self.display_name)
        head = _build_linear(state_dict, w_key, b_key, self.num_classes)

        self.model = self._finalize(DinoClassifier(backbone, head, mean_t, std_t))
        self.notes.append("feature standardization applied: (feature - mean) / std from checkpoint")
        self.notes.append(
            "CLS feature = last_hidden_state[:, 0]; exact training feature path assumed, not proven"
        )

    def _forward(self, images: List[Image.Image]) -> Any:
        return self.model(self._prepare_batch(images))


class DinoV2FinetunedAdapter(ModelAdapter):
    """Model 4: fine-tuned DINOv2-Small backbone + separate classifier head."""

    def _load(self) -> None:
        backbone = load_dinov2_backbone(self.spec["backbone"], self.cache_dir)
        checkpoint = read_checkpoint(self.spec["checkpoint"])
        if not isinstance(checkpoint, dict) or not {"model_state_dict", "classifier_state_dict"} <= set(checkpoint):
            keys = list(checkpoint)[:10] if isinstance(checkpoint, dict) else type(checkpoint).__name__
            raise ModelLoadError(
                f"{self.display_name}: needs 'model_state_dict' and 'classifier_state_dict'. "
                f"Found: {keys}"
            )

        # Backbone: align key names to the installed Transformers version, then load strictly
        aligned, info = align_state_dict(checkpoint["model_state_dict"], backbone.state_dict().keys())
        load_state_dict_strict(backbone, aligned, f"{self.display_name} backbone")
        self.notes.append(
            f"backbone keys aligned: prefix '{info['prefix_stripped']}', "
            f"{info['direction']}, {info['remapped_keys']} remapped"
        )
        if info["remapped_keys"] and info["remapped_keys"] != EXPECTED_ATTENTION_REMAPS:
            self.notes.append(
                f"note: {info['remapped_keys']} keys remapped; project notes recorded "
                f"{EXPECTED_ATTENTION_REMAPS}"
            )

        # Head
        head_sd = strip_prefix(checkpoint["classifier_state_dict"])
        w_key, b_key = find_linear_head(head_sd, HIDDEN_DIM, self.num_classes, f"{self.display_name} head")
        head = _build_linear(head_sd, w_key, b_key, self.num_classes)

        self.model = self._finalize(DinoClassifier(backbone, head))
        meta = _historical_metadata(checkpoint)
        if meta:
            self.notes.append(f"checkpoint metadata (historical, not verified here): {meta}")
        self.notes.append(
            "CLS feature = last_hidden_state[:, 0] feeds the head; "
            "preprocessing assumed same as the linear model, not confirmed"
        )

    def _forward(self, images: List[Image.Image]) -> Any:
        return self.model(self._prepare_batch(images))
