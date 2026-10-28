"""Five-model smoke test on one image.

Loads every enabled model, runs one image through each and checks that the
output has the right shape, is finite and sums to 1. This shows the
pipelines EXECUTE. It says nothing about accuracy.

    python scripts\\smoke_test.py                       # synthetic test image
    python scripts\\smoke_test.py --image leaf.jpg      # your own image
    python scripts\\smoke_test.py --models convnext_tiny dinov2_linear
    python scripts\\smoke_test.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from plant_disease.config import (  # noqa: E402
    ConfigError,
    load_class_mapping,
    load_config,
    setup_logging,
)
from plant_disease.imaging import validate_image  # noqa: E402
from plant_disease.models import load_models  # noqa: E402


def make_synthetic_image(size: int = 256) -> Image.Image:
    """A textured green image. Predictions on it are meaningless by design."""
    rng = np.random.default_rng(0)
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 40 + 30 * rng.random((size, size))
    base[..., 1] = 110 + 60 * rng.random((size, size))
    base[..., 2] = 30 + 30 * rng.random((size, size))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def package_versions() -> Dict[str, str]:
    versions = {}
    for pkg in ("torch", "torchvision", "timm", "transformers", "safetensors", "Pillow", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Five-model smoke test.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", default=None, help="Image path (default: synthetic image)")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        setup_logging(config["app"].get("log_level", "INFO"))
        mapping = load_class_mapping(config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ---- image ----------------------------------------------------------
    if args.image:
        check = validate_image(args.image, config)
        if not check.ok:
            print("ERROR: image rejected: " + "; ".join(check.errors), file=sys.stderr)
            return 1
        image, image_label = check.image, args.image
        for warning in check.warnings:
            print(f"[image warning] {warning}")
    else:
        image, image_label = make_synthetic_image(), "synthetic image (predictions are meaningless)"
    print(f"Image: {image_label}\n")

    # ---- load models ------------------------------------------------------
    try:
        registry = load_models(config, mapping, device=args.device, only=args.models)
    except (ValueError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Device: {registry.device}\n")

    # ---- inference ----------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for row in registry.diagnostics():
        entry: Dict[str, Any] = {
            "name": row["name"],
            "display_name": row["display_name"],
            "load_ok": row["loaded"],
            "load_seconds": row["load_seconds"],
            "error": row["error"],
            "notes": row["notes"],
            "inference_ok": False,
        }
        if row["loaded"]:
            adapter = registry.adapters[row["name"]]
            try:
                adapter.predict_logits([image])              # warm-up
                start = time.perf_counter()
                proba = adapter.predict_proba([image])[0]
                elapsed = time.perf_counter() - start
                finite = bool(np.isfinite(proba).all())
                sums_to_one = bool(abs(float(proba.sum()) - 1.0) < 1e-6)
                top = int(np.argmax(proba))
                entry.update(
                    inference_ok=finite and sums_to_one and proba.shape == (mapping.num_classes,),
                    finite=finite,
                    sums_to_one=sums_to_one,
                    top1_class=mapping.name(top),
                    top1_score=round(float(proba[top]), 4),
                    latency_seconds=round(elapsed, 4),
                )
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"inference failed: {exc}"
        results.append(entry)

    # ---- report ---------------------------------------------------------------
    print(f"{'model':<28}{'load':<8}{'infer':<8}{'sec':<8}{'top-1 class (score)'}")
    print("-" * 82)
    for r in results:
        load = "OK" if r["load_ok"] else "FAIL"
        infer = "OK" if r["inference_ok"] else ("FAIL" if r["load_ok"] else "-")
        sec = f"{r['latency_seconds']:.2f}" if "latency_seconds" in r else "-"
        top = f"{r['top1_class']} ({r['top1_score']:.2f})" if "top1_class" in r else "-"
        print(f"{r['display_name']:<28}{load:<8}{infer:<8}{sec:<8}{top}")

    problems = [r for r in results if r["error"] or not r["inference_ok"]]
    if problems:
        print("\nProblems:")
        for r in problems:
            print(f"\n* {r['display_name']}:\n  " + str(r["error"] or "inference output invalid")
                  .replace("\n", "\n  "))

    notes = [(r["display_name"], n) for r in results for n in r["notes"]]
    if notes:
        print("\nNotes:")
        for name, note in notes:
            print(f"  [{name}] {note}")

    ok_count = sum(1 for r in results if r["inference_ok"])
    print(f"\n{ok_count}/{len(results)} model pipelines executed successfully.")
    print("This is a smoke test only: it does not measure accuracy.")

    out_dir: Path = config["paths"]["outputs"] / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_label,
        "device": registry.device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "results": results,
    }
    (out_dir / "smoke_test_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Result saved to {out_dir / 'smoke_test_result.json'}")
    return 0 if not problems and results else 1


if __name__ == "__main__":
    sys.exit(main())
