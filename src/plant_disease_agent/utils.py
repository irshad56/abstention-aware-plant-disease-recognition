"""Utility functions for device resolution, file validation, and format checking."""

import os
from pathlib import Path
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def resolve_device(requested_device: Optional[str] = "auto") -> str:
    """Resolves 'auto' to 'cuda' if GPU is available, else 'cpu'."""
    if not TORCH_AVAILABLE:
        return "cpu"

    req = (requested_device or "auto").lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def get_system_info() -> dict:
    """Returns runtime system specifications for diagnostics."""
    info = {
        "python_version": os.sys.version.split()[0],
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "gpu_name": None,
    }
    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
