"""Artifact Discovery and Inspection Utility.

Scans configured and default directories for model checkpoints, class mappings,
and nested archives. Generates both human-readable terminal and JSON reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def inspect_pt_checkpoint(path: Path) -> Dict[str, Any]:
    """Inspects a PyTorch checkpoint keys and shapes safely without modifying it."""
    if not TORCH_AVAILABLE:
        return {"status": "READABLE", "size_mb": round(path.stat().st_size / (1024 * 1024), 2), "notes": "PyTorch not available to unpack keys"}

    try:
        # Load on CPU
        data = torch.load(path, map_location="cpu")
        report = {
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "type": type(data).__name__,
        }
        if isinstance(data, dict):
            keys = list(data.keys())
            report["top_level_keys"] = keys[:10]
            report["total_keys"] = len(keys)

            # Check for standard sub-dictionaries
            target_dict = (
                data.get("model_state_dict")
                or data.get("state_dict")
                or data.get("model")
                or data
            )
            if isinstance(target_dict, dict):
                tensor_keys = list(target_dict.keys())
                report["num_tensors"] = len(tensor_keys)
                sample_shapes = {}
                for k in tensor_keys[:6]:
                    v = target_dict[k]
                    if hasattr(v, "shape"):
                        sample_shapes[k] = list(v.shape)
                report["sample_shapes"] = sample_shapes
        return report
    except Exception as e:
        return {"status": "CORRUPT_OR_UNREADABLE", "error": str(e)}


def inspect_artifacts(
    model_root: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes full discovery across default search roots."""
    root = Path(model_root or os.getenv("PLANT_MODEL_ROOT", os.getcwd()))

    report: Dict[str, Any] = {
        "search_root": str(root.resolve()),
        "torch_available": TORCH_AVAILABLE,
        "models": {},
        "metadata": {},
        "issues": [],
    }

    print(f"\n========================================================")
    print(f" ARTIFACT DISCOVERY AND HEALTH INSPECTION REPORT")
    print(f" Search Root: {root.resolve()}")
    print(f"========================================================\n")

    # Define targets to search
    targets = {
        "convnext_tiny": [
            "convnext_tiny_final.pth",
            "artifacts/models/01_ConvNeXt_Tiny/checkpoint.pt",
            "artifacts/models/convnext_tiny/checkpoint.pt",
            "convnext_final_complete.zip",
        ],
        "efficientnetv2_b0": [
            "PlantVillage_EfficientNetV2-B0_best.pth",
            "PlantVillage_EfficientNetV2-B0_best_final.pth",
            "artifacts/models/02_EfficientNetV2_B0/checkpoint.pt",
            "artifacts/models/efficientnetv2_b0/checkpoint.pt",
        ],
        "dinov2_linear": [
            "dinov2_linear_best.pth",
            "artifacts/models/03_DINOv2_Small_Linear/checkpoint.pth",
            "artifacts/models/dinov2_linear/checkpoint.pth",
            "dinov2_final_complete.zip",
        ],
        "dinov2_finetuned": [
            "dinov2_finetuned_best.pth",
            "artifacts/models/04_DINOv2_Small_FineTuned/checkpoint.pth",
            "artifacts/models/dinov2_finetuned/checkpoint.pth",
        ],
        "clip_vit_b32": [
            "clip_final_artifacts.zip",
            "artifacts/models/05_CLIP_ViT_B32_Frozen_MLP/clip_final_artifacts.zip",
            "artifacts/models/clip_vit_b32/clip_frozen_mlp/clip_mlp_weights.pth",
        ],
    }

    for model_id, candidates in targets.items():
        found = None
        for c in candidates:
            # Check root / c
            p1 = root / c
            if p1.exists():
                found = p1
                break
            # Check cwd / c
            p2 = Path(c)
            if p2.exists():
                found = p2
                break

        if found:
            print(f"[FOUND] {model_id.upper()}: {found}")
            if found.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(found, "r") as z:
                        nl = z.namelist()
                        report["models"][model_id] = {
                            "status": "ZIP_ARCHIVE",
                            "path": str(found),
                            "entries_count": len(nl),
                            "entries_sample": nl[:6],
                        }
                        print(f"       -> Valid Zip Archive containing {len(nl)} files.")
                except Exception as ze:
                    report["models"][model_id] = {"status": "INVALID_ZIP", "path": str(found), "error": str(ze)}
                    report["issues"].append(f"{model_id} zip is invalid: {ze}")
            else:
                ckpt_info = inspect_pt_checkpoint(found)
                report["models"][model_id] = {
                    "status": "CHECKPOINT_FILE",
                    "path": str(found),
                    "details": ckpt_info,
                }
                print(f"       -> Size: {found.stat().st_size / (1024*1024):.1f} MB | Info: {ckpt_info.get('num_tensors', 'N/A')} tensors")
        else:
            print(f"[MISSING] {model_id.upper()}: not found in candidate paths.")
            report["models"][model_id] = {"status": "MISSING"}
            report["issues"].append(f"Model checkpoint for '{model_id}' was not located.")

    # Check class mapping
    mapping_candidates = [
        root / "artifacts/metadata/class_mapping.json",
        Path("artifacts/metadata/class_mapping.json"),
    ]
    mapping_found = None
    for mc in mapping_candidates:
        if mc.exists():
            mapping_found = mc
            break

    if mapping_found:
        try:
            with open(mapping_found, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            if "class_names" in cdata:
                c_names = cdata["class_names"]
            elif "classes" in cdata:
                c_names = cdata["classes"]
            elif "class_to_idx" in cdata:
                c_names = [k for k, _ in sorted(cdata["class_to_idx"].items(), key=lambda x: int(x[1]))]
            else:
                c_names = []
            print(f"\n[FOUND] CLASS MAPPING: {mapping_found} ({len(c_names)} classes)")
            report["metadata"]["class_mapping"] = {
                "path": str(mapping_found),
                "num_classes": len(c_names),
                "verified": len(c_names) == 29,
            }
            if len(c_names) != 29:
                report["issues"].append(f"Class mapping has {len(c_names)} classes instead of 29.")
        except Exception as me:
            report["issues"].append(f"Failed parsing class mapping {mapping_found}: {me}")
    else:
        print(f"\n[MISSING] CLASS MAPPING: metadata/class_mapping.json not found.")
        report["issues"].append("Authoritative class mapping JSON not found.")

    print(f"\nSummary: {len(report['issues'])} potential issues identified.")
    print(f"========================================================\n")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Machine-readable JSON report written to: {out_p}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect local model checkpoints and artifacts.")
    parser.add_argument("--model-root", "-r", type=str, default=None, help="Root folder to search")
    parser.add_argument("--output-json", "-o", type=str, default="outputs/artifact_inspection_report.json", help="Path to write JSON report")
    args = parser.parse_args()

    inspect_artifacts(model_root=args.model_root, output_json=args.output_json)
