"""Image validation, quality checks and model-specific preprocessing.

Validation (no PyTorch needed):

    check = validate_image(uploaded_file, config)
    if check.ok:
        pil_image = check.image          # RGB, EXIF-corrected

Preprocessing (CNN and DINOv2 models):

    tensor = get_preprocessor(model_spec["preprocessing"])(pil_image)

CLIP uses its own Hugging Face processor, handled inside the CLIP adapter.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from plant_disease.config import get_logger

log = get_logger("imaging")

ImageInput = Union[str, Path, bytes, bytearray, Image.Image, Any]

# Heuristic quality limits (grayscale 0-255). Not validated; reported as warnings.
DARK_MEAN = 25.0
BRIGHT_MEAN = 235.0
LOW_CONTRAST_STD = 15.0
QUALITY_ANALYSIS_MAX_SIDE = 512


# ==========================================================================
# Validation
# ==========================================================================
@dataclass
class ImageCheck:
    """Outcome of validating one image."""

    ok: bool
    image: Optional[Image.Image] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_poor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly summary (the image itself is not included)."""
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
            "quality_poor": self.quality_poor,
        }


def _read_bytes(source: ImageInput) -> Optional[bytes]:
    """Return raw bytes for path/bytes/file-like inputs, None for PIL images."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):          # Streamlit UploadedFile, BytesIO
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    return None


def sharpness_score(image: Image.Image) -> float:
    """Variance of a 4-neighbour Laplacian on the grayscale image (higher = sharper)."""
    gray_img = image.convert("L")
    if max(gray_img.size) > QUALITY_ANALYSIS_MAX_SIDE:
        gray_img.thumbnail((QUALITY_ANALYSIS_MAX_SIDE, QUALITY_ANALYSIS_MAX_SIDE))
    g = np.asarray(gray_img, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def validate_image(source: ImageInput, config: Optional[Dict[str, Any]] = None) -> ImageCheck:
    """Decode and check an image. Never raises for bad input; returns errors instead.

    Hard failures (ok=False): unreadable file, file too large, image too small.
    Soft problems (warnings, quality_poor=True): blurry, too dark/bright, low contrast.
    """
    cfg_app = (config or {}).get("app", {})
    cfg_abs = (config or {}).get("abstention", {})
    max_mb = float(cfg_app.get("max_upload_mb", 15))
    min_side = int(cfg_abs.get("min_image_side", 64))
    min_sharp = float(cfg_abs.get("min_sharpness", 20.0))

    check = ImageCheck(ok=False)

    try:
        raw = _read_bytes(source)
    except OSError as exc:
        check.errors.append(f"Could not read the file: {exc}")
        return check

    try:
        if raw is not None:
            if len(raw) == 0:
                check.errors.append("The file is empty.")
                return check
            size_mb = len(raw) / (1024 * 1024)
            check.metrics["file_size_mb"] = round(size_mb, 3)
            if size_mb > max_mb:
                check.errors.append(
                    f"File is {size_mb:.1f} MB; the limit is {max_mb:.0f} MB."
                )
                return check
            with Image.open(io.BytesIO(raw)) as probe:
                probe.verify()                      # detect truncated/corrupt data
            image = Image.open(io.BytesIO(raw))
            image.load()
        elif isinstance(source, Image.Image):
            image = source.copy()
        else:
            check.errors.append(f"Unsupported input type: {type(source).__name__}")
            return check
    except Image.DecompressionBombError:
        check.errors.append("Image has too many pixels and was rejected for safety.")
        return check
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        check.errors.append(f"Not a valid image file ({type(exc).__name__}).")
        return check

    # EXIF orientation, then RGB (matches the documented preprocessing order)
    image = ImageOps.exif_transpose(image)
    had_alpha = image.mode in ("RGBA", "LA", "P") and "transparency" in image.info \
        or image.mode in ("RGBA", "LA")
    if had_alpha:
        check.warnings.append("Image has transparency; it was flattened onto a white background.")
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    width, height = image.size
    check.metrics.update(width=width, height=height)
    if min(width, height) < min_side:
        check.errors.append(
            f"Image is {width}x{height}px; the shortest side must be at least {min_side}px."
        )
        return check

    # Quality heuristics
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness, contrast = float(gray.mean()), float(gray.std())
    sharpness = sharpness_score(image)
    check.metrics.update(
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2),
    )
    if sharpness < min_sharp:
        check.warnings.append(
            f"Image looks blurry (sharpness {sharpness:.1f} < {min_sharp:.1f}; heuristic)."
        )
        check.quality_poor = True
    if brightness < DARK_MEAN:
        check.warnings.append("Image is very dark.")
        check.quality_poor = True
    elif brightness > BRIGHT_MEAN:
        check.warnings.append("Image is very bright / overexposed.")
        check.quality_poor = True
    if contrast < LOW_CONTRAST_STD:
        check.warnings.append("Image has very low contrast.")
        check.quality_poor = True

    check.ok = True
    check.image = image
    return check


# ==========================================================================
# Preprocessing
# ==========================================================================
def _interpolation(name: str):
    from torchvision.transforms import InterpolationMode

    table = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
        "lanczos": InterpolationMode.LANCZOS,
    }
    try:
        return table[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown interpolation '{name}'. Use one of {sorted(table)}.") from exc


def build_transform(spec: Dict[str, Any]):
    """Build a torchvision transform (PIL RGB -> normalized float tensor [3,H,W]).

    spec keys: resize (int), interpolation, mean, std, and optional
    resize_mode: 'square' (default, resize to size x size) or
                 'shorter_center_crop' (resize shorter side, center-crop).
    """
    from torchvision import transforms as T

    if spec.get("use_clip_processor"):
        raise ValueError("CLIP preprocessing is handled by the CLIP adapter, not build_transform.")

    size = int(spec["resize"])
    interp = _interpolation(spec.get("interpolation", "bilinear"))
    mode = spec.get("resize_mode", "square")
    if mode == "square":
        steps = [T.Resize((size, size), interpolation=interp)]
    elif mode == "shorter_center_crop":
        steps = [T.Resize(size, interpolation=interp), T.CenterCrop(size)]
    else:
        raise ValueError(f"Unknown resize_mode '{mode}'.")
    steps += [T.ToTensor(), T.Normalize(mean=list(spec["mean"]), std=list(spec["std"]))]
    return T.Compose(steps)


def get_preprocessor(spec: Dict[str, Any]) -> Callable[[Image.Image], Any]:
    """Return a callable PIL image -> tensor [3,H,W] for the given preprocessing spec."""
    transform = build_transform(spec)

    def preprocess(image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return transform(image)

    return preprocess


def preprocess_batch(images: List[Image.Image], spec: Dict[str, Any]):
    """Stack several PIL images into a tensor [N,3,H,W]."""
    import torch

    fn = get_preprocessor(spec)
    return torch.stack([fn(img) for img in images])
