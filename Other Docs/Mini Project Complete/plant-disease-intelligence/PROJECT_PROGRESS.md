# PROJECT PROGRESS

- Architecture version: 2.0 (lean)
- Delivery method: directory by directory. A directory is complete only when
  every file listed for it is present. The next directory starts after that.
- Python 3.11; single requirements.txt; package at project root (plant_disease/)

## Stages

| Stage | Directory | Files | Status |
|---|---|---|---|
| 1 | project root | .gitignore, requirements.txt, pytest.ini, .env.example, README.md, PROJECT_PROGRESS.md | delivered |
| 2 | configs/ | config.yaml | pending |
| 3 | plant_disease/models/ | __init__.py, base.py, cnn.py, dinov2.py, clip_mlp.py | pending |
| 4 | data/evidence/ | sources.yaml | pending |
| 5 | plant_disease/ | __init__.py, config.py, imaging.py, ensemble.py, evaluation.py, retrieval.py, agent.py, pipeline.py | pending |
| 6 | scripts/ | setup_artifacts.py, download_backbones.py, smoke_test.py, evaluate.py, build_evidence.py | pending |
| 7 | app/ | streamlit_app.py | pending |
| 8 | tests/ | conftest.py, test_models.py, test_ensemble.py, test_retrieval.py, test_pipeline.py | pending |

After stage 8: Phase C (setup, model placement, verification, smoke test,
pytest, Streamlit, evaluation).

## Key decisions

- Checkpoints are renamed to checkpoint.pt / checkpoint.pth under artifacts/models/<name>/
- Strict checkpoint loading everywhere; no silent fallbacks
- DINOv2 key alignment is driven by the installed model's own keys
- CLIP path: vision_model -> pooler_output -> visual_projection -> L2 norm -> MLP
- Decision calibration: one temperature fitted on the averaged ensemble, validation split only
- Abstention thresholds are provisional until fitted on validation data
- Test split is never used to fit anything
- LLM (Groq) is optional and cannot override abstention

## Known unresolved (needs your files)

- Contents of the CLIP archive and the exact MLP keys
- timm constructor/version for EfficientNetV2-B0
- Preprocessing for ConvNeXt, EfficientNetV2, fine-tuned DINOv2, CLIP
- Real manifest column names (auto-detected; check with --check-manifest)
- Evidence corpus must be curated by the team from real sources

## Tested vs not tested (honest record)

- Tested in a sandbox without PyTorch: config, image validation, key alignment,
  registry logic, ensemble maths, calibration, evaluation workflow (stand-in models).
- NOT yet run: anything that loads real checkpoints or PyTorch/Transformers models.

## Resume prompt

"Resume the Plant Disease project from this PROJECT_PROGRESS.md. Architecture v2.0
is approved; do not redesign it. Deliver the next pending stage: all files of that
one directory, complete, with their destination folder. Do not move to a later
directory until I confirm the current one. Do not start Phase C until all
stages are delivered."
