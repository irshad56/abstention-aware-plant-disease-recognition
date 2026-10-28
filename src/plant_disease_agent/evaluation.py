"""Evaluation and verification module for labeled datasets and checkpoints."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

from plant_disease_agent.calibration import (
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_selective_metrics,
)
from plant_disease_agent.config import AppConfig
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.model_registry import ModelRegistry
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class DatasetEvaluator:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.registry = ModelRegistry(self.config)
        self.registry.discover_and_load_all()
        self.aggregator = DeepEnsembleAggregator(class_names=self.config.class_names)

    def evaluate_manifest(
        self,
        manifest_path: str,
        image_root: Optional[str] = None,
        split: Optional[str] = "test",
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates all loaded models and ensemble against a labeled manifest CSV."""
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir or f"outputs/evaluations/run_{run_id}")
        out_path.mkdir(parents=True, exist_ok=True)

        m_file = Path(manifest_path)
        if not m_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {m_file}")

        df = pd.read_csv(m_file)

        # Detect columns
        path_col = next((c for c in ["filepath", "path", "image_path", "filename"] if c in df.columns), None)
        label_col = next((c for c in ["label", "class", "target", "class_name"] if c in df.columns), None)
        split_col = next((c for c in ["split", "set", "dataset_split"] if c in df.columns), None)

        if not path_col or not label_col:
            raise ValueError(f"Could not automatically detect image path and label columns in {m_file}. Columns: {df.columns.tolist()}")

        if split and split_col and split in df[split_col].values:
            df = df[df[split_col] == split]
            logger.info(f"Filtered manifest to split '{split}': {len(df)} samples.")

        if max_samples:
            df = df.head(max_samples)

        img_base = Path(image_root) if image_root else m_file.parent
        loaded_adapters = self.registry.get_loaded_adapters()
        if not loaded_adapters:
            logger.warning("No model adapters are currently loaded. Evaluation cannot proceed.")
            return {"status": "FAILED", "reason": "No models loaded."}

        labels = []
        model_preds: Dict[str, List[int]] = {m_id: [] for m_id in loaded_adapters.keys()}
        model_probs: Dict[str, List[List[float]]] = {m_id: [] for m_id in loaded_adapters.keys()}
        ensemble_preds = []
        ensemble_probs = []
        skipped_count = 0

        for _, row in df.iterrows():
            rel_p = str(row[path_col])
            raw_label = row[label_col]

            # Resolve label to index
            if isinstance(raw_label, int) or str(raw_label).isdigit():
                label_idx = int(raw_label)
            else:
                label_idx = self.config.class_to_idx.get(str(raw_label))
                if label_idx is None:
                    # Try normalizing spaces
                    label_idx = self.config.class_to_idx.get(str(raw_label).replace("__", "_"))

            if label_idx is None or label_idx >= len(self.config.class_names):
                skipped_count += 1
                continue

            # Load image
            img_path = img_base / rel_p
            if not img_path.exists():
                skipped_count += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped_count += 1
                continue

            labels.append(label_idx)

            # Run per-model inference
            step_predictions = {}
            for m_id, adapter in loaded_adapters.items():
                p = adapter.predict(img)
                step_predictions[m_id] = p
                model_preds[m_id].append(p.predicted_idx)
                model_probs[m_id].append(p.probabilities)

            # Run ensemble
            ens_res = self.aggregator.aggregate(step_predictions)
            ensemble_preds.append(ens_res.predicted_idx)
            ensemble_probs.append(ens_res.ensemble_probabilities)

        if not labels:
            return {"status": "FAILED", "reason": "No valid samples evaluated (check image paths).", "skipped": skipped_count}

        y_true = np.array(labels)
        results = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "manifest": str(m_file),
            "split": split,
            "total_evaluated": len(labels),
            "skipped": skipped_count,
            "models": {},
        }

        # Calculate per-model metrics
        for m_id in loaded_adapters.keys():
            y_pred = np.array(model_preds[m_id])
            probs_arr = np.array(model_probs[m_id])
            confs = np.max(probs_arr, axis=1)

            acc = float(np.mean(y_pred == y_true))
            ece = compute_ece(confs, y_pred, y_true)
            brier = compute_brier_score(probs_arr, y_true, len(self.config.class_names))
            nll = compute_nll(probs_arr, y_true)

            results["models"][m_id] = {
                "display_name": loaded_adapters[m_id].display_name,
                "accuracy": round(acc, 4),
                "ece": round(ece, 4),
                "brier_score": round(brier, 4),
                "nll": round(nll, 4),
            }

        # Calculate ensemble metrics
        ens_pred_arr = np.array(ensemble_preds)
        ens_prob_arr = np.array(ensemble_probs)
        ens_confs = np.max(ens_prob_arr, axis=1)

        ens_acc = float(np.mean(ens_pred_arr == y_true))
        ens_ece = compute_ece(ens_confs, ens_pred_arr, y_true)
        ens_brier = compute_brier_score(ens_prob_arr, y_true, len(self.config.class_names))
        ens_nll = compute_nll(ens_prob_arr, y_true)

        # Selective metrics at default threshold
        abstain_mask = ens_confs < 0.70
        selective = compute_selective_metrics(ens_pred_arr, y_true, abstain_mask)

        results["ensemble"] = {
            "accuracy": round(ens_acc, 4),
            "ece": round(ens_ece, 4),
            "brier_score": round(ens_brier, 4),
            "nll": round(ens_nll, 4),
            "selective_prediction": selective,
        }

        # Save JSON output
        json_out = out_path / "evaluation_report.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save readable summary
        summary_out = out_path / "summary.md"
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write(f"# Evaluation Summary Report — Run {run_id}\n\n")
            f.write(f"- **Evaluated Samples:** {len(labels)}\n")
            f.write(f"- **Skipped Samples:** {skipped_count}\n")
            f.write(f"- **Split:** `{split}`\n\n")
            f.write("## Ensemble Performance\n")
            f.write(f"- **Accuracy:** {ens_acc:.2%}\n")
            f.write(f"- **Expected Calibration Error (ECE):** {ens_ece:.4f}\n")
            f.write(f"- **Brier Score:** {ens_brier:.4f}\n")
            f.write(f"- **Coverage (Confidence >= 0.70):** {selective['coverage']:.2%}\n")
            f.write(f"- **Selective Accuracy:** {selective['selective_accuracy']:.2%}\n\n")
            f.write("## Individual Models\n")
            for m_id, m_metrics in results["models"].items():
                f.write(f"### {m_metrics['display_name']}\n")
                f.write(f"- Accuracy: {m_metrics['accuracy']:.2%}\n")
                f.write(f"- ECE: {m_metrics['ece']:.4f} | Brier: {m_metrics['brier_score']:.4f}\n\n")

        logger.info(f"Evaluation report written to {out_path}")
        return results
