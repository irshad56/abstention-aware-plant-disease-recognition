# Abstention-Aware Plant Disease Recognition Using Diverse Deep Ensembles and Agentic Evidence Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green.svg)](https://github.com/langchain-ai/langgraph)
[![Tests Passing](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()

Production-oriented, research-grade B.Tech mini-project integrating **five diverse deep learning architectures**, **calibrated uncertainty estimation**, **statistical abstention policies**, and **LangGraph agentic evidence retrieval** from verified agricultural institutions.

---

## 1. System Highlights

1. **Heterogeneous Deep Ensemble:**
   - **ConvNeXt-Tiny:** Pure convolutional modern vision backbone ($7\times 7$ depthwise kernels).
   - **EfficientNetV2-B0:** Compound-scaled mobile inverted bottleneck architecture.
   - **DINOv2-Small (Linear Head):** Self-supervised Vision Transformer with standardized feature normalization.
   - **DINOv2-Small (Fine-Tuned):** ViT-Small adapted to plant pathology with attention-key alignment.
   - **CLIP ViT-B/32 (MLP):** Multimodal visual encoder with L2-normalized embeddings and trained multi-layer perceptron.

2. **Calibrated Abstention Engine:**
   - Evaluates Predictive Shannon Entropy, Top-2 Probability Margin, and Inter-Model Disagreement.
   - Rejects unconfident predictions, out-of-distribution inputs, and poor-quality photographs instead of hallucinating wrong diseases.
   - Supports five explicit decision states: `ACCEPTED`, `ABSTAINED`, `INVALID_IMAGE`, `MODEL_UNAVAILABLE`, `EVIDENCE_INSUFFICIENT`.

3. **LangGraph Agentic Orchestration:**
   - Multi-node state graph with dynamic conditional routing and cycle safety.
   - Validates every retrieved agricultural citation against domain whitelists (`.edu`, `.gov`, `apsnet.org`, `cabi.org`, `fao.org`).
   - Hallucination-free explanation generator powered by optional Groq LLM with a 100% deterministic fallback.

4. **Streamlit Interactive Application:**
   - 6 tabs: *Prediction & Diagnosis*, *Model Diagnostics*, *Evidence & Citations*, *Agent Trace*, *Evaluation*, *About & Methodology*.

---

## 2. Quickstart & Installation

### A. Environment Setup
```powershell
# 1. Create Python virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt
```

### B. Artifact Placement and Discovery
The application searches for model checkpoints either in the workspace root or under `artifacts/models/`:

```text
artifacts/models/
├── 01_ConvNeXt_Tiny/checkpoint.pt       (or convnext_tiny_final.pth)
├── 02_EfficientNetV2_B0/checkpoint.pt   (or PlantVillage_EfficientNetV2-B0_best.pth)
├── 03_DINOv2_Small_Linear/checkpoint.pth(or dinov2_linear_best.pth)
├── 04_DINOv2_Small_FineTuned/checkpoint.pth (or dinov2_finetuned_best.pth)
└── 05_CLIP_ViT_B32_Frozen_MLP/          (extracted from clip_final_artifacts.zip)
```

Run the automated inspection tool to verify checkpoints without modifying them:
```powershell
python scripts/inspect_artifacts.py
```

If using `clip_final_artifacts.zip`, safely extract its contents:
```powershell
python scripts/extract_model_archives.py clip_final_artifacts.zip --dest artifacts/models/clip_vit_b32
```

---

## 3. Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 4. Verification and Automated Testing

Run the automated test suite (25 test cases covering unit logic, graph routing, uncertainty calculation, and failure recovery):
```powershell
pytest -v
```

Verify all 5 deep learning model adapters and execute forward passes on a test image:
```powershell
python scripts/verify_models.py
```

Validate the 29-class agricultural evidence corpus:
```powershell
python scripts/build_evidence_index.py
```

---

## 5. Offline Evaluation & Reproducibility

Evaluate the ensemble against a user-provided labeled manifest CSV:
```powershell
python scripts/evaluate_models.py --manifest path/to/manifest.csv --split test
```
The script computes:
- Multi-class Accuracy and Macro F1
- Expected Calibration Error (ECE)
- Brier Score and Negative Log-Likelihood (NLL)
- Selective Accuracy, Selective Risk, and Coverage after abstention
- Machine-readable JSON reports saved to `outputs/evaluations/`

---

## 6. Project Structure

```text
├── app.py                     # Streamlit multi-tab web application
├── requirements.txt           # Pin-compatible dependency list
├── pyproject.toml             # Packaging and pytest configuration
├── .env.example               # Environment variable templates
├── config/
│   ├── settings.yaml          # Quality thresholds and application settings
│   └── model_registry.yaml    # 5-model candidate paths and hyperparameters
├── artifacts/
│   └── metadata/class_mapping.json  # Authoritative 29-class mapping
├── scripts/
│   ├── inspect_artifacts.py   # Checkpoint inspection without mutation
│   ├── extract_model_archives.py # Secure zip extractor preventing zip-slip
│   ├── verify_models.py       # Live forward verification pass CLI
│   ├── evaluate_models.py     # Labeled dataset evaluation CLI
│   └── build_evidence_index.py# Evidence corpus coverage validator
├── src/plant_disease_agent/
│   ├── schemas.py             # Typed Pydantic data models
│   ├── config.py              # Configuration and class mapping manager
│   ├── image_validation.py    # Blur, exposure, and aspect ratio checks
│   ├── uncertainty.py         # Shannon entropy, margin, disagreement math
│   ├── ensemble.py            # Calibrated deep ensemble aggregator
│   ├── abstention.py          # Multi-criterion selective abstention engine
│   ├── calibration.py         # ECE, Brier score, selective metrics
│   ├── model_registry.py      # Adapter lifecycle and checkpoint discovery
│   ├── inference.py           # Multi-model inference runner
│   ├── reporting.py           # Presentation formatting and comparison tables
│   ├── utils.py               # Hardware resolution (CPU / CUDA)
│   ├── model_adapters/        # 5 Independent Model Adapters
│   │   ├── base.py
│   │   ├── convnext.py
│   │   ├── efficientnetv2.py
│   │   ├── dinov2_linear.py
│   │   ├── dinov2_finetuned.py
│   │   └── clip_mlp.py
│   ├── evidence/              # Grounded Evidence Retrieval
│   │   ├── sources.py         # Domain whitelist (.edu, .gov, apsnet.org)
│   │   ├── retrieval.py       # Local corpus & optional live search
│   │   ├── validation.py      # Citation verification & URL checking
│   │   └── citations.py       # Markdown citation formatting
│   └── agents/                # LangGraph Agentic Workflow
│       ├── state.py           # Typed shared workflow state
│       ├── nodes.py           # Execution node functions
│       ├── tools.py           # LLM synthesizer & deterministic fallback
│       ├── graph.py           # Compiled StateGraph with conditional edges
│       └── orchestrator.py    # End-to-end pipeline entry point
├── data/evidence/corpus.json  # Verified university extension facts for 29 classes
├── tests/                     # 25 automated pytest tests
└── docs/                      # In-depth technical documentation
    ├── architecture.md
    ├── model_artifacts.md
    ├── methodology.md
    └── limitations.md
```

---

## 7. Configuration & Environment Variables

Create `.env` from `.env.example`:
```ini
# Optional model checkpoint root override
PLANT_MODEL_ROOT=c:/path/to/models

# Optional Groq LLM API Key (deterministic fallback is used if omitted)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Device Selection (auto, cpu, cuda)
PLANT_DEVICE=auto
```

---

## 8. Agricultural Diagnostic Disclaimer

Image-based plant disease recognition is an assistive research and decision-support tool. It cannot observe microscopic fungal spores, vascular bacterial streaming, viral systemic distribution, or soil chemistry. It must not supersede on-site inspection by an accredited agricultural extension specialist or plant pathology diagnostic laboratory.
