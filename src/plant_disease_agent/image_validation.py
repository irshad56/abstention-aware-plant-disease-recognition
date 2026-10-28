"""Input image validation, quality heuristics, and EXIF normalization."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

from plant_disease_agent.schemas import SuitabilityAssessment
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class ImageValidator:
    def __init__(
        self,
        min_width: int = 64,
        min_height: int = 64,
        max_aspect_ratio: float = 4.0,
        min_sharpness_variance: float = 15.0,
        min_mean_intensity: float = 0.05,
        max_mean_intensity: float = 0.95,
        max_file_size_mb: float = 15.0,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.min_sharpness_variance = min_sharpness_variance
        self.min_mean_intensity = min_mean_intensity
        self.max_mean_intensity = max_mean_intensity
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def load_and_validate(
        self,
        image_input: Union[bytes, io.BytesIO, Path, str]
    ) -> Tuple[Optional[Image.Image], SuitabilityAssessment]:
        """Loads an image, normalizes EXIF orientation, converts to RGB, and validates quality."""
        reasons = []

        # Check raw size if bytes
        if isinstance(image_input, bytes):
            if len(image_input) > self.max_file_size_bytes:
                reasons.append(f"File size exceeds {self.max_file_size_bytes // (1024 * 1024)} MB limit.")
            raw_stream = io.BytesIO(image_input)
        elif isinstance(image_input, io.BytesIO):
            val = image_input.getvalue()
            if len(val) > self.max_file_size_bytes:
                reasons.append(f"File size exceeds {self.max_file_size_bytes // (1024 * 1024)} MB limit.")
            raw_stream = image_input
        else:
            p = Path(image_input)
            if not p.exists():
                return None, SuitabilityAssessment(
                    is_suitable=False,
                    reasons=[f"File not found: {p}"]
                )
            if p.stat().st_size > self.max_file_size_bytes:
                reasons.append(f"File size exceeds {self.max_file_size_bytes // (1024 * 1024)} MB limit.")
            raw_stream = open(p, "rb")

        try:
            pil_img = Image.open(raw_stream)
            pil_img.verify()
        except Exception as e:
            return None, SuitabilityAssessment(
                is_suitable=False,
                reasons=[f"Image file is corrupt or invalid format: {str(e)}"]
            )
        finally:
            if not isinstance(image_input, (bytes, io.BytesIO)):
                raw_stream.close()

        # Re-open after verify() invalidated stream
        if isinstance(image_input, (bytes, io.BytesIO)):
            raw_stream.seek(0)
            pil_img = Image.open(raw_stream)
        else:
            pil_img = Image.open(image_input)

        # 1. EXIF orientation correction
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception as e:
            logger.debug(f"EXIF transpose skipped or failed: {e}")

        # 2. RGB conversion
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        w, h = pil_img.size

        # 3. Resolution check
        if w < self.min_width or h < self.min_height:
            reasons.append(f"Resolution too low ({w}x{h}). Minimum required is {self.min_width}x{self.min_height}.")

        # 4. Aspect ratio check
        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > self.max_aspect_ratio:
            reasons.append(f"Extreme aspect ratio ({aspect_ratio:.2f} > {self.max_aspect_ratio}).")

        # 5. Quality metrics (Grayscale conversion for heuristics)
        gray = pil_img.convert("L")
        img_np = np.asarray(gray, dtype=np.float32)

        # Mean brightness (normalized to 0-1)
        mean_intensity = float(np.mean(img_np) / 255.0)
        if mean_intensity < self.min_mean_intensity:
            reasons.append("Image is severely underexposed (too dark).")
        elif mean_intensity > self.max_mean_intensity:
            reasons.append("Image is severely overexposed (too bright / washed out).")

        # Sharpness / Blur heuristic via discrete Laplacian approximation
        # Filter: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        if img_np.shape[0] >= 3 and img_np.shape[1] >= 3:
            laplacian = (
                img_np[:-2, 1:-1]
                + img_np[2:, 1:-1]
                + img_np[1:-1, :-2]
                + img_np[1:-1, 2:]
                - 4 * img_np[1:-1, 1:-1]
            )
            sharpness_variance = float(np.var(laplacian))
        else:
            sharpness_variance = 0.0

        if sharpness_variance < self.min_sharpness_variance:
            reasons.append(
                f"Image appears excessively blurry (sharpness score: {sharpness_variance:.1f} < {self.min_sharpness_variance})."
            )

        is_suitable = len(reasons) == 0

        assessment = SuitabilityAssessment(
            is_suitable=is_suitable,
            reasons=reasons,
            width=w,
            height=h,
            sharpness_variance=round(sharpness_variance, 2),
            mean_intensity=round(mean_intensity, 3),
            aspect_ratio=round(aspect_ratio, 2),
        )

        return pil_img, assessment
