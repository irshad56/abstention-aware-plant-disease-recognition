"""Tests for image validation heuristics, EXIF handling, and corrupt input detection."""

import io
import numpy as np
import pytest
from PIL import Image

from plant_disease_agent.image_validation import ImageValidator


def test_valid_image_accepted(valid_image_bytes):
    validator = ImageValidator()
    img, assessment = validator.load_and_validate(valid_image_bytes)
    assert assessment.is_suitable is True
    assert len(assessment.reasons) == 0
    assert assessment.width == 224
    assert assessment.height == 224
    assert img is not None


def test_corrupt_image_rejected():
    validator = ImageValidator()
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_12345"
    img, assessment = validator.load_and_validate(corrupt_bytes)
    assert assessment.is_suitable is False
    assert img is None
    assert any("corrupt" in r.lower() or "invalid" in r.lower() for r in assessment.reasons)


def test_low_resolution_rejected():
    validator = ImageValidator(min_width=64, min_height=64)
    tiny = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    tiny.save(buf, format="JPEG")

    img, assessment = validator.load_and_validate(buf.getvalue())
    assert assessment.is_suitable is False
    assert any("resolution" in r.lower() for r in assessment.reasons)


def test_extreme_aspect_ratio_rejected():
    validator = ImageValidator(max_aspect_ratio=4.0)
    skinny = Image.new("RGB", (500, 50), color="green")
    buf = io.BytesIO()
    skinny.save(buf, format="JPEG")

    img, assessment = validator.load_and_validate(buf.getvalue())
    assert assessment.is_suitable is False
    assert any("aspect ratio" in r.lower() for r in assessment.reasons)


def test_overexposed_image_rejected():
    validator = ImageValidator(max_mean_intensity=0.95)
    washed_out = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    washed_out.save(buf, format="JPEG")

    img, assessment = validator.load_and_validate(buf.getvalue())
    assert assessment.is_suitable is False
    assert any("overexposed" in r.lower() for r in assessment.reasons)
