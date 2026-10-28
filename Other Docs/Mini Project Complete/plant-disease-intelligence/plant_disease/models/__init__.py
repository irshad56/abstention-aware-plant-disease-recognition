"""Model registry: build, load and track all five adapters.

    from plant_disease.models import load_models
    registry = load_models(config, class_mapping)
    registry.adapters      # {name: loaded adapter}
    registry.failures      # {name: error message} for models that did not load

Each model loads independently. One failure never stops the others, and a
failed model is always reported, never replaced with a fake prediction.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from plant_disease.config import (
    ClassMapping,
    enabled_models,
    get_logger,
    load_class_mapping,
    select_device,
)
from plant_disease.models.base import (
    ModelAdapter,
    ModelInferenceError,
    ModelLoadError,
)

log = get_logger("models.registry")

# spec["type"] in config.yaml -> (module, class). Imported lazily so that a
# problem in one model's dependencies does not break the others.
ADAPTER_REGISTRY: Dict[str, tuple] = {
    "convnext": ("plant_disease.models.cnn", "ConvNeXtAdapter"),
    "efficientnet": ("plant_disease.models.cnn", "EfficientNetAdapter"),
    "dinov2_linear": ("plant_disease.models.dinov2", "DinoV2LinearAdapter"),
    "dinov2_finetuned": ("plant_disease.models.dinov2", "DinoV2FinetunedAdapter"),
    "clip_mlp": ("plant_disease.models.clip_mlp", "ClipMlpAdapter"),
}

__all__ = [
    "ModelAdapter",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelRegistry",
    "load_models",
    "resolve_device",
]


def resolve_device(config: Dict[str, Any], override: Optional[str] = None) -> str:
    """'cpu' or 'cuda' from the override or config. CPU never imports PyTorch."""
    preference = (override or config["app"].get("device", "auto")).lower()
    return "cpu" if preference == "cpu" else select_device(preference)


@dataclass
class ModelRegistry:
    """Loaded models plus a record of every model that failed to load."""

    device: str
    class_mapping: ClassMapping
    adapters: Dict[str, ModelAdapter] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)

    @property
    def loaded_names(self) -> List[str]:
        return list(self.adapters)

    @property
    def failed_names(self) -> List[str]:
        return list(self.failures)

    def get(self, name: str) -> ModelAdapter:
        return self.adapters[name]

    def diagnostics(self) -> List[Dict[str, Any]]:
        """One row per enabled model, for the UI and reports."""
        rows: List[Dict[str, Any]] = []
        for name, display in self.display_names.items():
            if name in self.adapters:
                row = self.adapters[name].describe()
                row["error"] = None
            else:
                row = {
                    "name": name,
                    "display_name": display,
                    "device": self.device,
                    "loaded": False,
                    "load_seconds": None,
                    "preprocessing_verified": None,
                    "notes": [],
                    "error": self.failures.get(name, "not loaded"),
                }
            rows.append(row)
        return rows


def _build_adapter(
    name: str,
    spec: Dict[str, Any],
    class_mapping: ClassMapping,
    device: str,
    cache_dir: Any,
) -> ModelAdapter:
    kind = spec.get("type")
    if kind not in ADAPTER_REGISTRY:
        raise ModelLoadError(f"Unknown model type '{kind}' (known: {sorted(ADAPTER_REGISTRY)}).")
    module_name, class_name = ADAPTER_REGISTRY[kind]
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:  # noqa: BLE001 - missing dependency etc.
        raise ModelLoadError(f"Could not import {module_name}: {type(exc).__name__}: {exc}") from exc
    return cls(name, spec, class_mapping, device=device, cache_dir=cache_dir)


def load_models(
    config: Dict[str, Any],
    class_mapping: Optional[ClassMapping] = None,
    device: Optional[str] = None,
    only: Optional[List[str]] = None,
) -> ModelRegistry:
    """Load every enabled model (or only the names in `only`).

    Never raises for an individual model: failures are collected in
    registry.failures with the reason.
    """
    mapping = class_mapping or load_class_mapping(config)
    device = resolve_device(config, device)
    cache_dir = config["paths"].get("hf_cache")
    registry = ModelRegistry(device=device, class_mapping=mapping)

    specs = enabled_models(config)
    if only:
        unknown = [n for n in only if n not in specs]
        if unknown:
            raise ValueError(f"Unknown or disabled model(s): {unknown}. Available: {list(specs)}")
        specs = {n: specs[n] for n in only}

    for name, spec in specs.items():
        registry.display_names[name] = spec.get("display_name", name)
        start = time.perf_counter()
        try:
            adapter = _build_adapter(name, spec, mapping, device, cache_dir)
            adapter.load()
            registry.adapters[name] = adapter
        except Exception as exc:  # noqa: BLE001 - isolate each model
            message = str(exc) or type(exc).__name__
            registry.failures[name] = message
            log.error("Model %s failed to load after %.1fs: %s",
                      name, time.perf_counter() - start, message.splitlines()[0])

    log.info("Models loaded: %s | failed: %s",
             registry.loaded_names or "none", registry.failed_names or "none")
    return registry
