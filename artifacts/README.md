# Model Artifacts Directory

This directory hosts the 5 deep learning models and their checkpoints.
Checkpoints can be placed directly here or discovered in the project workspace by `scripts/inspect_artifacts.py`.

## Directory Structure
- `01_ConvNeXt_Tiny/checkpoint.pt` (or `convnext_tiny/checkpoint.pt`)
- `02_EfficientNetV2_B0/checkpoint.pt` (or `efficientnetv2_b0/checkpoint.pt`)
- `03_DINOv2_Small_Linear/checkpoint.pth` (or `dinov2_linear/checkpoint.pth`)
- `04_DINOv2_Small_FineTuned/checkpoint.pth` (or `dinov2_finetuned/checkpoint.pth`)
- `05_CLIP_ViT_B32_Frozen_MLP/clip_final_artifacts.zip` (or extracted into `clip_vit_b32/`)
- `metadata/class_mapping.json` (authoritative 29-class mapping)
