"""Common interface and helpers for all five model adapters.

Every adapter:
  * is built from its entry in configs/config.yaml,
  * loads its checkpoint strictly (any missing/unexpected key is an error),
  * checks that it outputs one logit per class in the shared class mapping,
  * returns logits/probabilities as NumPy arrays aligned to class_mapping.json.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from plant_disease.config import ClassMapping, ConfigError, get_logger
from plant_disease.imaging import get_preprocessor

log = get_logger("models")

TRUST_ENV_VAR = "PD_TRUST_CHECKPOINTS"
STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model", "net", "weights")


class ModelLoadError(Exception):
    """A model could not be built or its checkpoint could not be loaded."""


class ModelInferenceError(Exception):
    """A loaded model failed or returned invalid output."""


# ==========================================================================
# Checkpoint helpers (shared by all adapters)
# ==========================================================================
def read_checkpoint(path: Path | str) -> Any:
    """Load a torch checkpoint on CPU.

    Uses weights_only=True first. Files that need full unpickling are loaded
    only if you set the environment variable PD_TRUST_CHECKPOINTS=1, because
    unpickling can run arbitrary code (safe only for files you created).
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:  # noqa: BLE001
        if os.environ.get(TRUST_ENV_VAR, "").strip().lower() in ("1", "true", "yes"):
            log.warning("Loading %s with weights_only=False (%s=1).", path.name, TRUST_ENV_VAR)
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(f"Could not read {path.name}: {exc}") from exc
        raise ModelLoadError(
            f"Could not read {path.name} with weights_only=True "
            f"({type(safe_error).__name__}: {str(safe_error)[:200]}).\n"
            f"If you created this file yourself, set {TRUST_ENV_VAR}=1 and retry."
        ) from safe_error


def _is_state_dict(value: Any) -> bool:
    import torch

    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(isinstance(v, torch.Tensor) for v in value.values())
    )


def strip_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """Remove a DataParallel-style prefix if every key has it."""
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(
    checkpoint: Any, label: str = "checkpoint", keys: Sequence[str] = STATE_DICT_KEYS
) -> Dict[str, Any]:
    """Find the weights inside a checkpoint (plain state dict or a wrapper dict)."""
    if _is_state_dict(checkpoint):
        return strip_prefix(checkpoint)
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint and _is_state_dict(checkpoint[key]):
                return strip_prefix(checkpoint[key])
        raise ModelLoadError(
            f"{label}: no state dict found. Top-level keys: {list(checkpoint)[:15]}"
        )
    if hasattr(checkpoint, "state_dict") and callable(checkpoint.state_dict):
        return strip_prefix(checkpoint.state_dict())
    raise ModelLoadError(f"{label}: unsupported checkpoint type {type(checkpoint).__name__}.")


def load_state_dict_strict(module: Any, state_dict: Dict[str, Any], label: str) -> None:
    """Load weights and fail loudly on any missing/unexpected key or shape mismatch."""
    try:
        result = module.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise ModelLoadError(f"{label}: tensor shape mismatch.\n{str(exc)[:800]}") from exc
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise ModelLoadError(
            f"{label}: checkpoint does not match the model.\n"
            f"  missing keys    ({len(missing)}): {missing[:8]}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}"
        )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (float64)."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ==========================================================================
# Adapter base class
# ==========================================================================
class ModelAdapter(ABC):
    """Base class for one model. Subclasses implement _load() and _forward()."""

    def __init__(
        self,
        name: str,
        spec: Dict[str, Any],
        class_mapping: ClassMapping,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.spec = spec
        self.display_name: str = spec.get("display_name", name)
        self.class_mapping = class_mapping
        self.num_classes = class_mapping.num_classes
        self.device = device
        self.cache_dir = cache_dir
        self.model: Any = None
        self.load_seconds: Optional[float] = None
        self.last_batch_seconds: Optional[float] = None
        self.notes: List[str] = []
        self._preprocessor = None

    # ---- to implement --------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """Build the network, load the checkpoint, set self.model."""

    @abstractmethod
    def _forward(self, images: List[Image.Image]) -> Any:
        """Return a logits tensor of shape [len(images), num_classes]."""

    # ---- shared behaviour ------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def preprocessing_verified(self) -> bool:
        return bool(self.spec.get("preprocessing", {}).get("verified", False))

    def _finalize(self, module: Any) -> Any:
        """Move to device, switch to eval mode and freeze parameters."""
        module = module.to(self.device).eval()
        for param in module.parameters():
            param.requires_grad_(False)
        return module

    def _prepare_batch(self, images: List[Image.Image]) -> Any:
        """Default preprocessing: PIL images -> normalized tensor batch on device."""
        import torch

        if self._preprocessor is None:
            self._preprocessor = get_preprocessor(self.spec["preprocessing"])
        batch = torch.stack([self._preprocessor(img) for img in images])
        return batch.to(self.device)

    def load(self) -> None:
        """Load the model and verify its output shape. Raises ModelLoadError."""
        start = time.perf_counter()
        try:
            self._load()
            self._validate_output()
        except ModelLoadError:
            self.model = None
            raise
        except Exception as exc:  # noqa: BLE001
            self.model = None
            raise ModelLoadError(f"{self.display_name}: {type(exc).__name__}: {exc}") from exc
        self.load_seconds = time.perf_counter() - start
        log.info("Loaded %s on %s in %.2fs", self.display_name, self.device, self.load_seconds)

    def _validate_output(self) -> None:
        probe = Image.new("RGB", (256, 256), (128, 128, 128))
        logits = self._run([probe])
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ModelLoadError(f"{self.display_name}: unexpected output shape {logits.shape}.")
        try:
            self.class_mapping.validate_output_dim(int(logits.shape[1]), self.display_name)
        except ConfigError as exc:
            raise ModelLoadError(str(exc)) from exc
        if not np.isfinite(logits).all():
            raise ModelLoadError(f"{self.display_name}: non-finite logits on probe image.")

    def _run(self, images: List[Image.Image]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            logits = self._forward(images)
        return logits.detach().float().cpu().numpy()

    def predict_logits(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Logits as float32 array [N, num_classes], ordered like class_mapping.json."""
        if not self.loaded:
            raise ModelInferenceError(f"{self.display_name} is not loaded.")
        if len(images) == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)

        outputs: List[np.ndarray] = []
        start = time.perf_counter()
        for i in range(0, len(images), batch_size):
            chunk = list(images[i:i + batch_size])
            try:
                out = self._run(chunk)
            except Exception as exc:  # noqa: BLE001
                raise ModelInferenceError(
                    f"{self.display_name} failed during inference: {type(exc).__name__}: {exc}"
                ) from exc
            if out.shape != (len(chunk), self.num_classes):
                raise ModelInferenceError(
                    f"{self.display_name} returned shape {out.shape}, "
                    f"expected {(len(chunk), self.num_classes)}."
                )
            if not np.isfinite(out).all():
                raise ModelInferenceError(f"{self.display_name} produced non-finite logits.")
            outputs.append(out)
        self.last_batch_seconds = time.perf_counter() - start
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Softmax scores [N, num_classes]. These are model scores, not calibrated probabilities."""
        return softmax_np(self.predict_logits(images, batch_size))

    def describe(self) -> Dict[str, Any]:
        """Diagnostics for the UI and reports."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "device": self.device,
            "loaded": self.loaded,
            "load_seconds": None if self.load_seconds is None else round(self.load_seconds, 3),
            "preprocessing_verified": self.preprocessing_verified,
            "notes": list(self.notes),
        }
