"""Model adapters package."""

from plant_disease_agent.model_adapters.base import BaseModelAdapter
from plant_disease_agent.model_adapters.convnext import ConvNeXtAdapter
from plant_disease_agent.model_adapters.efficientnetv2 import EfficientNetV2Adapter
from plant_disease_agent.model_adapters.dinov2_linear import DINOv2LinearAdapter
from plant_disease_agent.model_adapters.dinov2_finetuned import DINOv2FineTunedAdapter
from plant_disease_agent.model_adapters.clip_mlp import CLIPMLPAdapter

__all__ = [
    "BaseModelAdapter",
    "ConvNeXtAdapter",
    "EfficientNetV2Adapter",
    "DINOv2LinearAdapter",
    "DINOv2FineTunedAdapter",
    "CLIPMLPAdapter",
]
