"""Reporting and formatting helpers for presentation tables and markdown views."""

from typing import Dict, List
import pandas as pd

from plant_disease_agent.schemas import EnsembleResult, ModelPrediction


def create_model_comparison_table(
    predictions: Dict[str, ModelPrediction],
    ensemble: EnsembleResult,
) -> pd.DataFrame:
    """Builds a comparison DataFrame comparing the 5 models against the ensemble decision."""
    rows = []
    ens_top = ensemble.predicted_class

    for m_id, p in predictions.items():
        is_agree = (p.predicted_class == ens_top) if p.status == "success" else False
        rows.append({
            "Model": p.display_name,
            "Status": "Loaded" if p.status == "success" else p.status.capitalize(),
            "Predicted Class": p.predicted_class if p.status == "success" else "—",
            "Confidence": f"{p.confidence:.1%}" if p.status == "success" else "—",
            "Agreement with Ensemble": "Agreement" if is_agree else "Disagreement",
            "Latency (ms)": f"{p.latency_ms:.1f}" if p.status == "success" else "—",
        })

    # Add ensemble row
    rows.append({
        "Model": "★ Deep Ensemble",
        "Status": "Consensus",
        "Predicted Class": ensemble.predicted_class,
        "Confidence": f"{ensemble.confidence:.1%}",
        "Agreement with Ensemble": "Majority Target",
        "Latency (ms)": "—",
    })

    return pd.DataFrame(rows)


def create_top_k_distribution_table(
    probabilities: List[float],
    class_names: List[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Creates a top-K probability distribution table."""
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:top_k]
    rows = []
    for rank, (idx, prob) in enumerate(indexed, 1):
        rows.append({
            "Rank": rank,
            "Class Name": class_names[idx],
            "Probability": f"{prob:.2%}",
            "Score": round(prob, 4),
        })
    return pd.DataFrame(rows)
