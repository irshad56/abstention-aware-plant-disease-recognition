# Model Artifacts Specification and Verification Protocol

This document outlines the architecture, checkpoint format, input tensor preprocessing, and strict verification protocol for all five deep learning models integrated into the ensemble.

## Summary Table

| Model ID | Display Name | Backbone / Family | Feature Dim | Input Size | Interpolation | Checkpoint Target |
|---|---|---|---|---|---|---|
| `convnext_tiny` | ConvNeXt-Tiny | `torchvision.models.convnext_tiny` | 768 | $224\times 224$ | Bicubic (Resize 236, Crop 224) | `convnext_tiny_final.pth` |
| `efficientnetv2_b0` | EfficientNetV2-B0 | `timm:tf_efficientnetv2_b0` | 1280 | $224\times 224$ | Bicubic ($224\times 224$) | `PlantVillage_EfficientNetV2-B0_best.pth` |
| `dinov2_linear` | DINOv2-Small (Linear) | `facebook/dinov2-small` | 384 | $518\times 518$ | Bicubic ($518\times 518$) | `dinov2_linear_best.pth` |
| `dinov2_finetuned` | DINOv2-Small (Fine-Tuned) | `facebook/dinov2-small` | 384 | $518\times 518$ | Bicubic ($518\times 518$) | `dinov2_finetuned_best.pth` |
| `clip_vit_b32` | CLIP ViT-B/32 (MLP) | `openai/clip-vit-base-patch32` | 512 | $224\times 224$ | Bicubic ($224\times 224$) | `clip_final_artifacts.zip` |

---

## 1. ConvNeXt-Tiny
- **Torchvision Identifier:** `torchvision.models.convnext_tiny(weights=None)`
- **Classifier Configuration:** `Linear(768, 29)`
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Checkpoint Keys:** Standard `features.*` blocks and `classifier.2.*` linear head.

## 2. EfficientNetV2-B0
- **Timm Identifier:** `timm.create_model("tf_efficientnetv2_b0", pretrained=False, num_classes=29)`
- **Input Pipeline:** Bicubic resize to $224\times 224$, CenterCrop(224), ImageNet normalization.

## 3. DINOv2-Small (Linear Classifier)
- **Backbone:** Frozen `facebook/dinov2-small` Vision Transformer.
- **Input Pipeline:** Bicubic resize to $518\times 518$, ImageNet normalization.
- **Representation:** CLS token extracted at index 0 of the last hidden state.
- **Feature Standardization:**
  $$z = \frac{f_{\text{CLS}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
  where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are the saved training feature statistics from the checkpoint.
- **Head:** Trained Linear head mapping normalized $384 \to 29$ logits.

## 4. DINOv2-Small (Fine-Tuned)
- **Backbone:** Fine-tuned `facebook/dinov2-small`.
- **Key Remapping Protocol:**
  Checkpoint uses attention aliases between PyTorch/timm conventions and HuggingFace transformers (`q_proj`, `k_proj`, `v_proj`, `o_proj` $\leftrightarrow$ `attention.attention.query/key/value` and `attention.output.dense`). The adapter executes automatic dynamic key alignment.

## 5. CLIP ViT-B/32 (Frozen Visual Encoder + Trained MLP)
- **Archive Format:** Contained in nested ZIP `clip_final_artifacts.zip`.
- **Extraction Protocol:** Automated safe extraction via `scripts/extract_model_archives.py` into `artifacts/models/clip_vit_b32/`.
- **Representation Pipeline:**
  $$\text{Pooled Vision Output} \longrightarrow \text{Visual Projection} \longrightarrow \text{L2 Normalization} \longrightarrow \text{MLP}$$
- **MLP Architecture:**
  - `Linear(512, 256)`
  - `LayerNorm(256)`
  - `GELU()`
  - `Dropout(0.20)`
  - `Linear(256, 29)`
