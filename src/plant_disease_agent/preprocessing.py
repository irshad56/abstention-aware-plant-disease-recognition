"""Image preprocessing pipelines for ConvNeXt, EfficientNetV2, DINOv2, and CLIP."""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def preprocess_numpy(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    crop_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Pure NumPy/PIL image preprocessing fallback."""
    if crop_size:
        # Resize shorter edge to target_size[0], then center crop
        img = img.resize(target_size, Image.Resampling.BICUBIC)
        w, h = img.size
        cw, ch = crop_size
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
    else:
        img = img.resize(target_size, Image.Resampling.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def get_torch_transform(
    resize_size: int = 224,
    crop_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    interpolation: str = "bicubic",
):
    """Builds a deterministic PyTorch evaluation transform."""
    if not TORCH_AVAILABLE:
        return None

    interp = (
        InterpolationMode.BICUBIC
        if interpolation.lower() == "bicubic"
        else InterpolationMode.BILINEAR
    )

    transforms_list = [
        T.Resize(resize_size, interpolation=interp, antialias=True),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(transforms_list)
