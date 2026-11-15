from pathlib import Path
import json
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

_CONFIG = Path(__file__).with_name("iqa_thresholds.json")

with open(_CONFIG, "r") as f:
    _T = json.load(f)


def assess_image_quality(image):
    """
    Assess image quality before plant disease classification.

    Returns:
        status:
            QUALITY_ACCEPTABLE
            LOW_IMAGE_QUALITY
            IMAGE_NOT_ELIGIBLE_FOR_CLASSIFICATION

        eligible_for_classification: bool
        reasons: list
        warnings: list
        features: dict
    """

    if image is None:
        return {
            "status": "IMAGE_NOT_ELIGIBLE_FOR_CLASSIFICATION",
            "eligible_for_classification": False,
            "reasons": ["INVALID_IMAGE"],
            "warnings": [],
            "features": {}
        }

    try:
        image = image.convert("RGB")
        width, height = image.size

        if width < 64 or height < 64:
            return {
                "status": "IMAGE_NOT_ELIGIBLE_FOR_CLASSIFICATION",
                "eligible_for_classification": False,
                "reasons": ["IMAGE_TOO_SMALL_TO_ASSESS"],
                "warnings": [],
                "features": {
                    "width": width,
                    "height": height
                }
            }

        gray = np.asarray(
            image.convert("L"),
            dtype=np.uint8
        )

        target_width = min(512, width)
        target_height = max(
            1,
            round(height * target_width / width)
        )

        if cv2 is not None:
            gray = cv2.resize(
                gray,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )
        else:
            gray = np.asarray(
                Image.fromarray(gray).resize(
                    (target_width, target_height),
                    Image.Resampling.BILINEAR
                ),
                dtype=np.uint8
            )

        gray_float = gray.astype(np.float32)
        contrast = float(gray_float.std())

        # FIXED: CV_32F is compatible with float32 input.
        if cv2 is not None:
            lap = cv2.Laplacian(
                gray_float,
                cv2.CV_32F
            )
        else:
            gy, gx = np.gradient(gray_float)
            gyy, _ = np.gradient(gy)
            _, gxx = np.gradient(gx)
            lap = gxx + gyy

        raw_laplacian = float(np.var(lap))

        normalized_sharpness = float(
            raw_laplacian / (contrast ** 2 + 1e-8)
        )

        features = {
            "normalized_sharpness": normalized_sharpness,
            "raw_laplacian_variance": raw_laplacian,
            "brightness": float(gray_float.mean()),
            "contrast": contrast,
            "width": width,
            "height": height,
            "min_side": min(width, height)
        }

        if features["min_side"] < _T["min_side_hard"]:
            return {
                "status": "IMAGE_NOT_ELIGIBLE_FOR_CLASSIFICATION",
                "eligible_for_classification": False,
                "reasons": ["INSUFFICIENT_NATIVE_RESOLUTION"],
                "warnings": [],
                "features": features
            }

        reasons = []
        warnings = []

        if (
            normalized_sharpness
            < _T["normalized_sharpness_hard_min"]
        ):
            reasons.append("SEVERE_BLUR_OR_DETAIL_LOSS")

        elif (
            normalized_sharpness
            < _T["normalized_sharpness_review_min"]
        ):
            warnings.append("BORDERLINE_SHARPNESS")

        if (
            features["brightness"]
            < _T["brightness_warning_min"]
        ):
            warnings.append("LOW_BRIGHTNESS_REVIEW")

        return {
            "status": (
                "LOW_IMAGE_QUALITY"
                if reasons
                else "QUALITY_ACCEPTABLE"
            ),
            "eligible_for_classification": not bool(reasons),
            "reasons": reasons,
            "warnings": warnings,
            "features": features
        }

    except Exception as exc:
        return {
            "status": "IMAGE_NOT_ELIGIBLE_FOR_CLASSIFICATION",
            "eligible_for_classification": False,
            "reasons": ["IMAGE_READ_OR_ASSESSMENT_ERROR"],
            "warnings": [],
            "error": str(exc)[:300],
            "features": {}
        }
