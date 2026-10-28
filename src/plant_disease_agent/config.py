"""Configuration management and class mapping verification."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class AppConfig:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        registry_path: str = "config/model_registry.yaml",
        class_mapping_path: Optional[str] = None,
    ):
        self.project_root = Path(os.getcwd())

        # Auto-load .env file if present
        env_file = self.project_root / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as ef:
                    for line in ef:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")

        self.settings = self._load_yaml(settings_path)
        self.registry = self._load_yaml(registry_path)

        # Environment variable overrides
        self.model_root = os.getenv("PLANT_MODEL_ROOT", str(self.project_root))
        self.device = os.getenv("PLANT_DEVICE", self.settings.get("app", {}).get("device", "auto"))
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv(
            "GROQ_MODEL", self.settings.get("llm", {}).get("model", "openai/gpt-oss-120b")
        )

        mapping_file = class_mapping_path or self.settings.get("paths", {}).get(
            "class_mapping", "artifacts/metadata/class_mapping.json"
        )
        self.class_names, self.class_to_idx, self.idx_to_class = self._load_and_validate_class_mapping(mapping_file)

    @property
    def abstention(self) -> Dict[str, Any]:
        """Provides direct access to abstention policy configuration dictionary."""
        return self.settings.get("abstention", {})

    @property
    def app_settings(self) -> Dict[str, Any]:
        return self.settings.get("app", {})

    @property
    def retrieval_settings(self) -> Dict[str, Any]:
        return self.settings.get("retrieval", {})

    @property
    def llm_settings(self) -> Dict[str, Any]:
        return self.settings.get("llm", {})

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        if not p.exists():
            logger.warning(f"Configuration file not found: {p}. Returning empty dict.")
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_and_validate_class_mapping(
        self, path: str
    ) -> tuple[List[str], Dict[str, int], Dict[int, str]]:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        if not p.exists():
            raise FileNotFoundError(f"Class mapping file is missing at: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "class_names" in data:
            class_names = data["class_names"]
        elif "classes" in data:
            class_names = data["classes"]
        elif "class_to_idx" in data:
            sorted_pairs = sorted(data["class_to_idx"].items(), key=lambda x: x[1])
            class_names = [k for k, _ in sorted_pairs]
        else:
            raise ValueError(f"Invalid class mapping schema in {p}: missing class list or map.")

        if len(class_names) != 29:
            raise ValueError(f"Class mapping must contain exactly 29 classes, got {len(class_names)}")

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        idx_to_class = {idx: name for idx, name in enumerate(class_names)}

        # Verify bi-directional consistency
        if "class_to_idx" in data:
            for c_name, c_idx in data["class_to_idx"].items():
                if class_to_idx.get(c_name) != c_idx:
                    logger.warning(f"Index mismatch for class '{c_name}': {class_to_idx.get(c_name)} vs {c_idx}")

        logger.info(f"Loaded verified 29-class mapping from {p}")
        return class_names, class_to_idx, idx_to_class

    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        return self.registry.get("models", {}).get(model_id, {})
