"""Tests for class mapping bijectivity and consistency."""

import json
from pathlib import Path
import pytest
from plant_disease_agent.config import AppConfig


def test_class_mapping_bijectivity():
    config = AppConfig()
    assert len(config.class_names) == 29
    assert len(config.class_to_idx) == 29
    assert len(config.idx_to_class) == 29

    for idx, name in enumerate(config.class_names):
        assert config.class_to_idx[name] == idx
        assert config.idx_to_class[idx] == name


def test_metadata_file_integrity():
    p = Path("artifacts/metadata/class_mapping.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "class_names" in data or "class_to_idx" in data
    classes = data.get("class_names") or list(data.get("class_to_idx", {}).keys())
    assert len(classes) == 29
