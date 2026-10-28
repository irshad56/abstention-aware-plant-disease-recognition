"""ConvNeXt-Tiny (torchvision) and EfficientNetV2-B0 (timm) adapters."""

from __future__ import annotations

from typing import Any, List

from PIL import Image

from plant_disease.models.base import (
    ModelAdapter,
    ModelLoadError,
    extract_state_dict,
    load_state_dict_strict,
    read_checkpoint,
)


class ConvNeXtAdapter(ModelAdapter):
    """torchvision.models.convnext_tiny(weights=None, num_classes=N) + saved weights."""

    def _load(self) -> None:
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None, num_classes=self.num_classes)
        checkpoint = read_checkpoint(self.spec["checkpoint"])
        state_dict = extract_state_dict(checkpoint, self.display_name)
        load_state_dict_strict(model, state_dict, self.display_name)
        self.model = self._finalize(model)
        if not self.preprocessing_verified:
            self.notes.append("preprocessing is provisional (not confirmed from training metadata)")

    def _forward(self, images: List[Image.Image]) -> Any:
        return self.model(self._prepare_batch(images))


class EfficientNetAdapter(ModelAdapter):
    """timm EfficientNetV2-B0 (name from config) + saved weights."""

    def _load(self) -> None:
        try:
            import timm
        except ImportError as exc:
            raise ModelLoadError("timm is not installed (pip install -r requirements.txt).") from exc

        timm_name = self.spec.get("timm_name", "tf_efficientnetv2_b0")
        try:
            model = timm.create_model(timm_name, pretrained=False, num_classes=self.num_classes)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"timm could not build '{timm_name}' (timm {timm.__version__}): {exc}"
            ) from exc
        checkpoint = read_checkpoint(self.spec["checkpoint"])
        state_dict = extract_state_dict(checkpoint, self.display_name)
        load_state_dict_strict(model, state_dict, self.display_name)
        self.model = self._finalize(model)
        self.notes.append(f"timm {timm.__version__}, architecture {timm_name}")
        if not self.preprocessing_verified:
            self.notes.append("preprocessing is provisional (not confirmed from training metadata)")

    def _forward(self, images: List[Image.Image]) -> Any:
        return self.model(self._prepare_batch(images))
