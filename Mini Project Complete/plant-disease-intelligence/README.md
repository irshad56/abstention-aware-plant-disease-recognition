# Abstention-Aware Plant Disease Recognition

**Using Diverse Deep Ensembles and Agentic Evidence Retrieval**

B.Tech mini-project, VRSEC, Vijayawada
Team: Shaik Irshad, P. Santhosh Kumar

## What this project does

1. Takes a leaf image and checks that it is usable.
2. Runs five existing, already-trained models independently:
   ConvNeXt-Tiny, EfficientNetV2-B0, DINOv2-Small (linear head),
   DINOv2-Small (fine-tuned) and CLIP ViT-B/32 with an MLP head.
3. Averages their probabilities (equal weights by default) and measures
   uncertainty: confidence, entropy, top-two margin and model disagreement.
4. **Abstains** ("I cannot confidently identify the disease from this image.")
   when the evidence is weak, instead of guessing.
5. When it accepts a prediction, retrieves agricultural evidence from a locally
   curated corpus of real sources and shows a cited explanation.

The models are **not retrained or replaced**. Classification and language
generation are separate: the language model can never change the abstention
decision or invent a confidence score.

## Honest status

- No accuracy figure is claimed anywhere in this repository. Numbers appear
  only after you run the evaluation yourself.
- Abstention thresholds start as **provisional and uncalibrated**. They become
  validated only after `--fit-calibration` is run on the validation split.
- Preprocessing for ConvNeXt, EfficientNetV2, the fine-tuned DINOv2 and CLIP is
  marked `verified: false` in `configs/config.yaml` until confirmed from the
  checkpoints (`setup_artifacts.py inspect`).
- Softmax outputs are model scores, not calibrated probabilities, unless a
  validation-fitted temperature is applied.
- The evidence corpus contains only sources you list and fetch yourself.

## Project layout

```text
plant-disease-intelligence/
  configs/config.yaml        all settings (paths, models, abstention, retrieval)
  plant_disease/             the package
    config.py                config, paths, logging, class mapping
    imaging.py               image validation and preprocessing
    models/                  one adapter per model + registry
    ensemble.py              averaging, uncertainty, abstention, calibration
    evaluation.py            dataset, metrics, reports
    retrieval.py             evidence corpus, search, citation checks
    agent.py                 LangGraph workflow, explanation
    pipeline.py              single entry point used by the UI
  app/streamlit_app.py       the user interface (Streamlit only)
  scripts/                   setup, verification, smoke test, evaluation, evidence
  data/evidence/sources.yaml the real sources you want retrieved
  tests/                     pytest suite
  artifacts/                 your model files (not committed)
  outputs/                   generated results (not committed)
```

## Where the model files go

| Your file | Place at |
|---|---|
| `convnext_tiny_best.pt` | `artifacts\models\convnext_tiny\checkpoint.pt` |
| `effnetv2_b0_best.pt` | `artifacts\models\efficientnetv2_b0\checkpoint.pt` |
| `dinov2_linear_clean_best.pth` | `artifacts\models\dinov2_linear\checkpoint.pth` |
| `dinov2_finetuned_best.pth` | `artifacts\models\dinov2_finetuned\checkpoint.pth` |
| `clip_final_artifacts.zip` | `artifacts\input\clip_final_artifacts.zip` |
| `class_mapping.json` | `artifacts\metadata\class_mapping.json` |
| test manifest CSV (optional) | `data\manifests\source_grouped_split_manifest_v2.csv` |

## Quick start (Windows)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts\setup_artifacts.py extract      :: unpack the CLIP archive
python scripts\setup_artifacts.py verify       :: check every file is present
python scripts\setup_artifacts.py inspect      :: dump checkpoint keys and shapes
python scripts\download_backbones.py           :: DINOv2 / CLIP backbones (internet, once)
python scripts\smoke_test.py --image your_leaf.jpg

copy .env.example .env                         :: optional: add GROQ_API_KEY
python scripts\build_evidence.py               :: fetch sources listed in data\evidence\sources.yaml
streamlit run app\streamlit_app.py
pytest
```

Linux/macOS: `python3.11 -m venv .venv && source .venv/bin/activate`, and use `/`
in paths.

## Evaluation (needs the dataset)

Set the dataset folder in `configs\config.yaml` (`paths.dataset_root`) or with
`PD_DATASET_ROOT`, then run in this order:

```bat
python scripts\evaluate.py --check-manifest    :: find every image, report missing ones
python scripts\evaluate.py --fit-calibration   :: validation split ONLY
python scripts\evaluate.py --split test        :: held-out evaluation, nothing is fitted
```

Results go to `outputs\evaluation\<split>_<timestamp>\` as CSV, JSON and plots.
The test split is never used for calibration, ensemble weights or thresholds.

## Troubleshooting

- **A checkpoint cannot be read in safe mode:** if you created the file yourself,
  set `PD_TRUST_CHECKPOINTS=1` and retry.
- **A model fails to load:** the error lists the missing/unexpected keys. Run
  `python scripts\setup_artifacts.py inspect` and compare.
- **DINOv2/CLIP backbone not found:** run `python scripts\download_backbones.py`.
- **No Groq key:** the app still works; explanations fall back to a plain
  summary of the retrieved evidence.
