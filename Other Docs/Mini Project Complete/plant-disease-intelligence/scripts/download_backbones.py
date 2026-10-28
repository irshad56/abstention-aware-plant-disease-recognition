"""Download the pretrained backbones the checkpoints depend on.

DINOv2 (both models) needs  facebook/dinov2-small
CLIP (model 5)       needs  openai/clip-vit-base-patch32  (unless the CLIP
                            archive already bundled it)

This is the only script that uses the internet, and only when you run it.
Files go to the local cache folder set in configs/config.yaml (paths.hf_cache).

    python scripts\\download_backbones.py                 # download what is missing
    python scripts\\download_backbones.py --check-only    # offline check, no download
    python scripts\\download_backbones.py --only dinov2   # or: --only clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease.config import ConfigError, enabled_models, load_config  # noqa: E402

ALLOW_SAFETENSORS = ["*.json", "*.txt", "*.safetensors"]
ALLOW_BIN = ["*.json", "*.txt", "pytorch_model.bin"]


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def has_weights(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (
        any(path.rglob("*.safetensors")) or any(path.rglob("pytorch_model.bin"))
    )


def required_backbones(config: Dict[str, Any], only: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Map repo_id -> {'used_by': [...], 'local_dir': Path|None} for enabled models."""
    needed: Dict[str, Dict[str, Any]] = {}
    for name, spec in enabled_models(config).items():
        kind = spec["type"]
        if kind.startswith("dinov2"):
            family = "dinov2"
        elif kind == "clip_mlp":
            family = "clip"
        else:
            continue
        if only and family != only:
            continue
        entry = needed.setdefault(
            spec["backbone"],
            {"used_by": [], "local_dir": spec.get("backbone_dir"), "family": family},
        )
        entry["used_by"].append(name)
    return needed


def cached_path(repo_id: str, cache_dir: Path) -> Optional[Path]:
    """Return the cached snapshot folder if the weights are present locally."""
    from huggingface_hub import snapshot_download

    try:
        folder = Path(
            snapshot_download(
                repo_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
                allow_patterns=ALLOW_SAFETENSORS + ["pytorch_model.bin"],
            )
        )
    except Exception:  # noqa: BLE001 - any failure means "not cached"
        return None
    return folder if has_weights(folder) else None


def download(repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(repo_id, cache_dir=str(cache_dir), allow_patterns=ALLOW_SAFETENSORS)
    )
    if not has_weights(folder):
        print("  no .safetensors weights in repo; trying pytorch_model.bin ...")
        folder = Path(
            snapshot_download(repo_id, cache_dir=str(cache_dir), allow_patterns=ALLOW_BIN)
        )
    if not has_weights(folder):
        raise RuntimeError(f"No model weights found after downloading {repo_id}.")
    return folder


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download pretrained backbones.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--check-only", action="store_true", help="Do not download anything")
    parser.add_argument("--only", choices=["dinov2", "clip"], default=None)
    parser.add_argument("--force", action="store_true",
                        help="Download CLIP even if a bundled backbone exists")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    cache_dir: Path = config["paths"]["hf_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = required_backbones(config, args.only)
    if not needed:
        print("No backbones required for the enabled models.")
        return 0

    print(f"Cache folder: {cache_dir}\n")
    problems = 0
    for repo_id, entry in needed.items():
        used = ", ".join(entry["used_by"])
        print(f"{repo_id}  (used by: {used})")

        local_dir = entry["local_dir"]
        if entry["family"] == "clip" and local_dir is not None and has_weights(local_dir) \
                and not args.force:
            print(f"  [OK] bundled backbone found in {local_dir}\n")
            continue

        existing = cached_path(repo_id, cache_dir)
        if existing is not None:
            print(f"  [OK] already cached ({human_size(folder_size(existing))})\n")
            continue

        if args.check_only:
            print("  [MISSING] not cached. Run this script without --check-only.\n")
            problems += 1
            continue

        print("  downloading from huggingface.co (internet required) ...")
        try:
            folder = download(repo_id, cache_dir)
            print(f"  [OK] downloaded ({human_size(folder_size(folder))}) -> {folder}\n")
        except Exception as exc:  # noqa: BLE001
            problems += 1
            print(f"  [FAILED] {type(exc).__name__}: {str(exc)[:300]}")
            print("  Check your internet connection, or copy the model folder into the "
                  "cache manually.\n")

    if problems:
        print(f"{problems} backbone(s) unavailable.")
        return 1
    print("All required backbones are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
