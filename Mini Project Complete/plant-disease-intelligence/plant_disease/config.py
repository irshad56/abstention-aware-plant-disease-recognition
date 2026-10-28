"""Configuration, paths, logging and class mapping.

Everything else in the project gets its settings from here:

    from plant_disease.config import load_config, get_logger, ClassMapping
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Environment variable -> config["paths"] key
_ENV_PATH_OVERRIDES = {
    "PD_ARTIFACT_ROOT": "artifact_root",
    "PD_DATASET_ROOT": "dataset_root",
    "PD_MANIFEST_PATH": "manifest",
    "PD_HF_CACHE": "hf_cache",
}


class ConfigError(Exception):
    """Raised when configuration or class-mapping files are missing or invalid."""


# --------------------------------------------------------------------------
# Paths and configuration
# --------------------------------------------------------------------------
def resolve_path(value: Optional[str | Path]) -> Optional[Path]:
    """Resolve a config path. Relative paths are relative to the project root."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is optional at import time
        pass


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load config.yaml, apply environment overrides and resolve all paths.

    Returns a plain dictionary. After this call every value under
    ``config["paths"]`` and every model checkpoint/directory entry is an
    absolute ``Path`` (or ``None`` when unset).
    """
    _load_dotenv()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"Config file is empty or malformed: {config_path}")

    for section in ("paths", "app", "models", "ensemble", "abstention", "retrieval"):
        if section not in config:
            raise ConfigError(f"Missing '{section}' section in {config_path}")

    # Environment overrides for paths
    for env_name, key in _ENV_PATH_OVERRIDES.items():
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            config["paths"][key] = env_value

    # Resolve paths
    config["paths"] = {k: resolve_path(v) for k, v in config["paths"].items()}
    for spec in config["models"].values():
        for key in ("checkpoint", "backbone_dir", "mlp_dir"):
            if key in spec:
                spec[key] = resolve_path(spec[key])

    config["_config_path"] = config_path
    return config


def enabled_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return only the model specs with ``enabled: true``."""
    return {n: s for n, s in config["models"].items() if s.get("enabled", True)}


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def select_device(preference: str = "auto") -> str:
    """Return 'cpu' or 'cuda'. 'auto' uses CUDA only when it is available."""
    import torch

    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("device: cuda requested but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _RedactSecrets(logging.Filter):
    """Mask API keys if they ever end up in a log message."""

    def filter(self, record: logging.LogRecord) -> bool:
        secret = os.environ.get("GROQ_API_KEY", "")
        if secret and len(secret) > 8:
            record.msg = str(record.getMessage()).replace(secret, "***REDACTED***")
            record.args = ()
        return True


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure console (and optional file) logging once."""
    root = logging.getLogger("plant_disease")
    if getattr(root, "_pd_configured", False):
        return
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.addFilter(_RedactSecrets())
        root.addHandler(handler)
    root._pd_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the 'plant_disease' namespace."""
    if not name.startswith("plant_disease"):
        name = f"plant_disease.{name}"
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Class mapping
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassMapping:
    """Authoritative class-index mapping loaded from class_mapping.json.

    Index meaning comes only from the saved file. Class names are never
    re-sorted, so classifier outputs keep their original meaning.
    """

    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def names(self) -> List[str]:
        """Class names ordered by index."""
        return [self.idx_to_class[i] for i in range(self.num_classes)]

    def name(self, idx: int) -> str:
        return self.idx_to_class[int(idx)]

    def index(self, name: str) -> int:
        return self.class_to_idx[name]

    def validate_output_dim(self, dim: int, model_name: str = "model") -> None:
        if dim != self.num_classes:
            raise ConfigError(
                f"{model_name} outputs {dim} classes but the class mapping has "
                f"{self.num_classes}."
            )

    @classmethod
    def load(
        cls, path: Path | str, expected_classes: Optional[int] = None
    ) -> "ClassMapping":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(
                f"Class mapping not found: {path}\n"
                "Place class_mapping.json in artifacts/metadata/."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "class_to_idx" not in data or "idx_to_class" not in data:
            raise ConfigError(
                "class_mapping.json must contain 'class_to_idx' and 'idx_to_class'."
            )

        class_to_idx = {str(k): int(v) for k, v in data["class_to_idx"].items()}
        idx_to_class = {int(k): str(v) for k, v in data["idx_to_class"].items()}

        n = len(class_to_idx)
        if sorted(class_to_idx.values()) != list(range(n)):
            raise ConfigError("class_to_idx indices are not contiguous 0..N-1.")
        if idx_to_class != {v: k for k, v in class_to_idx.items()}:
            raise ConfigError("class_to_idx and idx_to_class disagree.")
        if expected_classes is not None and n != expected_classes:
            raise ConfigError(
                f"Class mapping has {n} classes; config expects {expected_classes}."
            )
        return cls(class_to_idx=class_to_idx, idx_to_class=idx_to_class)


def load_class_mapping(config: Dict[str, Any]) -> ClassMapping:
    """Load and validate the class mapping named in the config."""
    return ClassMapping.load(
        config["paths"]["class_mapping"],
        expected_classes=config["app"].get("num_classes"),
    )
