# Technical Limitations, Scope Boundaries, and Agricultural Disclaimer

This document clarifies verified experimental boundaries, operational constraints, and honest technical reporting for the system.

## 1. Verified vs. Unverified Performance Claims

- **Grouped Accuracy Parity:**
  Historical records from the Google Colab experimental notebooks note top-1 accuracies above 99% for individual models (e.g. ConvNeXt-Tiny and EfficientNetV2-B0 on PlantVillage validation splits). However, these historical metrics must not be represented as benchmark results or generalization proof on arbitrary field foliage.
- **In-the-Wild Foliage Generalization:**
  The training dataset (PlantVillage) features clean, uniform laboratory backdrops. Real farm or garden environments introduce varying background soil, complex multi-leaf canopies, insect vectors, variable sun angles, and pesticide residues. The deep ensemble cannot guarantee equal accuracy on complex in-the-wild field photography.
- **Out-of-Distribution (OOD) Detection:**
  The current abstention policy utilizes heuristic image validation (Laplacian variance, intensity bounds) alongside ensemble predictive entropy, margin, and disagreement. While these reject out-of-focus or unconfident inputs, they do not constitute a formal conformal prediction or dedicated Out-Of-Distribution detector.

## 2. Image Preprocessing Parity

- **CLIP Visual Preprocessing:**
  The CLIP ViT-B/32 frozen visual encoder utilizes standard OpenAI CLIP bicubic $224\times 224$ normalization. Preprocessing parity with the original notebook training pipeline is documented as an approximation; empirical validation on a held-out test split is recommended before production use.
- **DINOv2 Key Alignment:**
  The fine-tuned DINOv2 checkpoint requires 96 attention key projections to be dynamically aligned with the installed Transformers module. If weight key names diverge across future Transformers library versions, the adapter logs warnings and falls back to non-strict matching.

## 3. Agricultural and Diagnostic Disclaimer

> [!IMPORTANT]
> **Not a Substitute for Agronomical Inspection:**
> Computer vision systems operating purely on 2D digital photographs cannot observe microscopic fungal structures, bacterial oozes, viral systemic distributions, or soil-borne nutrient deficiencies. This application is an assistive decision-support research tool for academic evaluation. It must never supersede direct examination by an accredited agricultural extension specialist or plant pathology diagnostic laboratory.

## 4. Evidence Retrieval Boundaries

- **Corpus vs. Live Web:**
  The local evidence corpus (`data/evidence/corpus.json`) contains verified, curated facts from university extension services (Penn State, Cornell, UC Davis, Purdue, UF IFAS, etc.) for each of the 29 supported classes.
- Live internet searches, when enabled, are restricted to trusted academic and governmental domains (`.edu`, `.gov`, `apsnet.org`, `cabi.org`, `fao.org`). If an internet connection or search credentials are unavailable, the system transparently utilizes the local verified corpus and explicitly reports that live search was bypassed.
