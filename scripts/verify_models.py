"""Model Verification Script: Loads all discoverable checkpoints and runs a forward verification pass."""

import argparse
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plant_disease_agent.config import AppConfig
from plant_disease_agent.model_registry import ModelRegistry
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


def verify_models():
    print("\n========================================================")
    print(" 5-DEEP-ENSEMBLE MODEL ADAPTER VERIFICATION")
    print("========================================================\n")

    config = AppConfig()
    registry = ModelRegistry(config)
    status_report = registry.discover_and_load_all()

    # Create dummy RGB image for test pass
    dummy_img = Image.new("RGB", (256, 256), color=(100, 150, 75))

    success_count = 0
    total = len(registry.adapters)

    for m_id, adapter in registry.adapters.items():
        info = status_report.get(m_id, {})
        status = info.get("status")
        path = info.get("path")
        d_name = adapter.display_name

        print(f"[{status}] {d_name} ({m_id})")
        if status == "LOADED":
            # Run forward pass
            pred = adapter.predict(dummy_img)
            if pred.status == "success":
                print(f"      -> Forward pass: SUCCESS | Latency: {pred.latency_ms:.1f}ms | Top class: {pred.predicted_class} ({pred.confidence:.1%})")
                print(f"      -> Output shape: {len(pred.probabilities)} classes | Sum: {sum(pred.probabilities):.4f}")
                success_count += 1
            else:
                print(f"      -> Forward pass FAILED: {pred.error_message}")
        elif status == "FAILED":
            print(f"      -> Load Error: {info.get('error')}")
        else:
            print(f"      -> Missing checkpoint. Searched candidate paths.")

    print(f"\nVerification Complete: {success_count}/{total} models verified for inference.\n")


if __name__ == "__main__":
    verify_models()
