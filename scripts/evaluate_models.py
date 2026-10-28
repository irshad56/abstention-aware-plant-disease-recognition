"""Dataset Evaluation CLI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plant_disease_agent.evaluation import DatasetEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate deep ensemble models against a labeled manifest.")
    parser.add_argument("--manifest", "-m", type=str, required=True, help="Path to manifest CSV file")
    parser.add_argument("--image-root", "-i", type=str, default=None, help="Root folder containing images")
    parser.add_argument("--split", "-s", type=str, default="test", help="Dataset split to evaluate (e.g. test, val)")
    parser.add_argument("--max-samples", "-n", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Directory to save evaluation reports")
    args = parser.parse_args()

    evaluator = DatasetEvaluator()
    print(f"Starting evaluation on manifest: {args.manifest} (split: {args.split})...")
    results = evaluator.evaluate_manifest(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
    print("Evaluation completed. Check summary report in output directory.")


if __name__ == "__main__":
    main()
