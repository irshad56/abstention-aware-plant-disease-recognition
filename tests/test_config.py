"""Tests for configuration loading and validation."""

import os
import pytest
from plant_disease_agent.config import AppConfig


def test_app_config_loading():
    config = AppConfig()
    assert config.settings is not None
    assert config.registry is not None
    assert len(config.class_names) == 29
    assert config.class_names[0] == "Apple__Apple Scab"
    assert config.class_names[-1] == "Tomato__Yellow Leaf Curl Virus"


def test_model_registry_configuration():
    config = AppConfig()
    expected_models = ["convnext_tiny", "efficientnetv2_b0", "dinov2_linear", "dinov2_finetuned", "clip_vit_b32"]
    for m in expected_models:
        m_cfg = config.get_model_config(m)
        assert m_cfg is not None
        assert "candidate_paths" in m_cfg
        assert m_cfg.get("num_classes") == 29
