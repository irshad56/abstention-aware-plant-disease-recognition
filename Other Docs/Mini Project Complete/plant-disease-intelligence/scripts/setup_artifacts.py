"""Artifact setup tool: extract the CLIP archive, verify files, inspect checkpoints.

Run from the project root:

    python scripts\\setup_artifacts.py extract            # unpack CLIP archive
    python scripts\\setup_artifacts.py verify             # check all files exist
    python scripts\\setup_artifacts.py verify --hash      # also compute SHA-256
    python scripts\\setup_artifacts.py inspect            # dump checkpoint keys/shapes
    python scripts\\setup_artifacts.py inspect --model dinov2_finetuned

Checkpoints are loaded with torch.load(weights_only=True) by default. If a
file cannot be read that way, add --trust (only for files you created).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease.config import (  # noqa: E402
    ClassMapping,
    ConfigError,
    enabled_models,
    load_class_mapping,
    load_config,
)

WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin")
METADATA_SUFFIXES = (".json", ".txt", ".csv", ".md", ".yaml", ".yml")
MAX_MLP_BYTES = 50 * 1024 * 1024          # an MLP head is far smaller than this
MAX_ARCHIVE_BYTES = 4 * 1024 ** 3         # refuse archives that expand beyond 4 GB
MAX_NESTING = 2


class SetupError(Exception):
    """A problem the user needs to fix; printed without a traceback."""


# ==========================================================================
# Generic helpers
# ==========================================================================
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise SetupError(
            "PyTorch is not installed in this environment. Install the "
            "requirements first (pip install -r requirements.txt)."
        ) from exc


def torch_load(path: Path, trust: bool = False) -> Any:
    """Load a torch checkpoint on CPU, safely by default."""
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:  # noqa: BLE001
        if not trust:
            raise SetupError(
                f"Could not load {path.name} with weights_only=True "
                f"({type(safe_error).__name__}: {str(safe_error)[:200]}).\n"
                "If you created this file yourself, re-run with --trust."
            ) from safe_error
        print(f"  [trust] loading {path.name} with weights_only=False")
        return torch.load(path, map_location="cpu", weights_only=False)


def safetensors_shapes(path: Path) -> Dict[str, Tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SetupError("safetensors is not installed.") from exc
    shapes: Dict[str, Tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


# ==========================================================================
# Checkpoint structure analysis
# ==========================================================================
def _is_tensor(value: Any) -> bool:
    torch = _torch()
    return isinstance(value, torch.Tensor)


def walk_checkpoint(obj: Any, path: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Split a loaded checkpoint into state dicts, loose tensors and metadata."""
    result: Dict[str, Any] = {"state_dicts": {}, "tensors": {}, "metadata": {}}

    def visit(value: Any, p: Tuple[str, ...]) -> None:
        name = ".".join(p) if p else "<root>"
        if isinstance(value, dict):
            if value and all(_is_tensor(v) for v in value.values()):
                result["state_dicts"][name] = value
                return
            for k, v in value.items():
                visit(v, p + (str(k),))
        elif _is_tensor(value):
            result["tensors"][name] = value
        elif hasattr(value, "state_dict") and callable(value.state_dict):
            result["state_dicts"][name + " (pickled module state_dict)"] = value.state_dict()
        else:
            result["metadata"][name] = value

    visit(obj, path)
    return result


def tensor_shapes_from_file(path: Path, trust: bool) -> Dict[str, Tuple[int, ...]]:
    """Flat {key: shape} for every tensor in a weight file."""
    if path.suffix == ".safetensors":
        return safetensors_shapes(path)
    parts = walk_checkpoint(torch_load(path, trust))
    shapes: Dict[str, Tuple[int, ...]] = {}
    for sd in parts["state_dicts"].values():
        for k, v in sd.items():
            shapes[k] = tuple(v.shape)
    for k, v in parts["tensors"].items():
        shapes[k] = tuple(v.shape)
    return shapes


def looks_like_clip_mlp(shapes: Dict[str, Tuple[int, ...]], num_classes: int) -> bool:
    """True if the file holds an MLP mapping 512 -> hidden -> num_classes."""
    two_d = [s for s in shapes.values() if len(s) == 2]
    for first in two_d:
        if first[1] == 512 and first[0] not in (num_classes, 512):
            hidden = first[0]
            if any(s == (num_classes, hidden) for s in two_d):
                return True
    return False


# ==========================================================================
# extract
# ==========================================================================
def safe_extract(zip_path: Path, dest: Path) -> List[Path]:
    """Extract a zip, rejecting paths that escape `dest` (zip-slip)."""
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    written: List[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SetupError(f"Corrupt member in {zip_path.name}: {bad}")
        total = sum(i.file_size for i in zf.infolist())
        if total > MAX_ARCHIVE_BYTES:
            raise SetupError(
                f"{zip_path.name} expands to {human_size(total)}, above the "
                f"{human_size(MAX_ARCHIVE_BYTES)} safety limit."
            )
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest_resolved != target and dest_resolved not in target.parents:
                raise SetupError(f"Unsafe path in archive rejected: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(target)
    return written


def expand_nested(staging: Path) -> None:
    """Unpack zip files found inside the staging folder (up to MAX_NESTING)."""
    for level in range(MAX_NESTING):
        inner = [p for p in staging.rglob("*.zip") if p.is_file()]
        if not inner:
            return
        for inner_zip in inner:
            print(f"  nested archive (level {level + 1}): {inner_zip.name}")
            target = inner_zip.with_suffix("")
            safe_extract(inner_zip, target)
            inner_zip.unlink()
    if any(p.is_file() for p in staging.rglob("*.zip")):
        raise SetupError(f"Archive nesting deeper than {MAX_NESTING} levels.")


def find_backbone_dirs(staging: Path) -> List[Path]:
    """Directories that look like a Hugging Face CLIP model folder."""
    found = []
    for cfg in staging.rglob("config.json"):
        folder = cfg.parent
        has_weights = any(
            (folder / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_weights:
            continue
        try:
            model_type = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "")
        except (OSError, json.JSONDecodeError):
            continue
        if "clip" in str(model_type).lower():
            found.append(folder)
    return found


def _is_inside(path: Path, folders: List[Path]) -> bool:
    return any(f == path or f in path.parents for f in folders)


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def cmd_extract(config: Dict[str, Any], args: argparse.Namespace) -> int:
    archive: Path = config["paths"]["clip_archive"]
    spec = config["models"]["clip_vit_b32"]
    num_classes = config["app"]["num_classes"]
    clip_dir: Path = spec["mlp_dir"].parent
    mlp_dir: Path = spec["mlp_dir"]
    backbone_dir: Path = spec["backbone_dir"]
    meta_dir: Path = clip_dir / "metadata"
    staging = clip_dir / "_staging"

    if not archive.is_file():
        raise SetupError(
            f"CLIP archive not found: {archive}\n"
            "Place clip_final_artifacts.zip in artifacts\\input\\."
        )
    if not zipfile.is_zipfile(archive):
        raise SetupError(f"{archive.name} is not a valid zip file.")
    if (_dir_has_files(mlp_dir) or _dir_has_files(backbone_dir)) and not args.force:
        raise SetupError(
            "CLIP folders already contain files. Re-run with --force to replace them."
        )

    print(f"Extracting {archive.name} ({human_size(archive.stat().st_size)}) ...")
    if staging.exists():
        shutil.rmtree(staging)
    safe_extract(archive, staging)
    expand_nested(staging)

    files = sorted(p for p in staging.rglob("*") if p.is_file())
    if not files:
        raise SetupError("The archive contained no files.")

    # Manifest of everything inside
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "path": p.relative_to(staging).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    ]
    (meta_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  {len(files)} file(s) found; manifest written.")

    # Backbone detection
    backbones = find_backbone_dirs(staging)
    if len(backbones) > 1:
        listing = "\n  ".join(str(b.relative_to(staging)) for b in backbones)
        raise SetupError(f"Several CLIP backbone folders found, cannot choose:\n  {listing}")
    backbone_src: Optional[Path] = backbones[0] if backbones else None

    # MLP detection
    candidates = [
        p
        for p in files
        if p.suffix in WEIGHT_SUFFIXES
        and p.stat().st_size <= MAX_MLP_BYTES
        and not _is_inside(p, backbones)
    ]
    matches: List[Path] = []
    report: List[str] = []
    for cand in candidates:
        try:
            shapes = tensor_shapes_from_file(cand, args.trust)
        except SetupError as exc:
            report.append(f"{cand.relative_to(staging)}: unreadable ({exc})")
            continue
        if looks_like_clip_mlp(shapes, num_classes):
            matches.append(cand)
        else:
            report.append(f"{cand.relative_to(staging)}: {len(shapes)} tensors, not an MLP head")

    if len(matches) != 1:
        detail = "\n  ".join(report) if report else "(no candidate weight files)"
        kind = "No" if not matches else "Several"
        raise SetupError(
            f"{kind} CLIP MLP checkpoint(s) identified (expected exactly one).\n"
            f"Looked for a 512 -> hidden -> {num_classes} MLP. Candidates:\n  {detail}\n"
            f"Staging kept at {staging} for inspection."
        )
    mlp_src = matches[0]

    # Place files
    for folder in (mlp_dir, backbone_dir):
        if folder.exists() and args.force:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(mlp_src), str(mlp_dir / mlp_src.name))
    if backbone_src is not None:
        for item in backbone_src.iterdir():
            shutil.move(str(item), str(backbone_dir / item.name))
    moved_metadata = []
    for p in files:
        if p.exists() and p.suffix in METADATA_SUFFIXES and not _is_inside(p, backbones):
            shutil.move(str(p), str(meta_dir / p.name))
            moved_metadata.append(p.name)

    summary = {
        "mlp_file": mlp_src.name,
        "backbone_bundled": backbone_src is not None,
        "backbone_files": sorted(p.name for p in backbone_dir.iterdir()) if backbone_src else [],
        "metadata_files": sorted(moved_metadata),
    }
    (meta_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)

    print("\nExtraction complete.")
    print(f"  MLP checkpoint : {mlp_dir / mlp_src.name}")
    if backbone_src is not None:
        print(f"  CLIP backbone  : bundled, placed in {backbone_dir}")
    else:
        print("  CLIP backbone  : NOT bundled in the archive.")
        print("                   Run: python scripts\\download_backbones.py")
    print(f"  Metadata files : {', '.join(moved_metadata) or 'none'}")
    return 0


# ==========================================================================
# verify
# ==========================================================================
def _hf_cache_has(config: Dict[str, Any], repo_id: str) -> bool:
    folder = "models--" + repo_id.replace("/", "--")
    roots = [config["paths"].get("hf_cache"), Path.home() / ".cache" / "huggingface" / "hub"]
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        for candidate in (root / folder, root / "hub" / folder):
            if not candidate.is_dir():
                continue
            if any(candidate.rglob("*.safetensors")) or any(candidate.rglob("pytorch_model.bin")):
                return True
    return False


def cmd_verify(config: Dict[str, Any], args: argparse.Namespace) -> int:
    rows: List[Dict[str, Any]] = []

    def add(name: str, path: Optional[Path], required: bool, note: str = "") -> None:
        exists = path is not None and Path(path).exists()
        size = ""
        if exists and Path(path).is_file():
            size = human_size(Path(path).stat().st_size)
        rows.append(
            {
                "item": name,
                "path": path,
                "required": required,
                "status": "OK" if exists else ("MISSING" if required else "WARN"),
                "size": size,
                "note": note,
            }
        )

    add("class mapping", config["paths"]["class_mapping"], True)
    for name, spec in enabled_models(config).items():
        if spec["type"] == "clip_mlp":
            continue
        add(name, spec["checkpoint"], True)

    clip = config["models"].get("clip_vit_b32", {})
    if clip.get("enabled", True):
        mlp_dir = clip["mlp_dir"]
        extracted = _dir_has_files(mlp_dir)
        rows.append(
            {
                "item": "clip_vit_b32 MLP",
                "path": mlp_dir,
                "required": True,
                "status": "OK" if extracted else "MISSING",
                "size": "",
                "note": "" if extracted else "run: setup_artifacts.py extract",
            }
        )
        add("clip archive (input)", config["paths"]["clip_archive"], False,
            "only needed until extracted")
        rows.append(
            {
                "item": "clip backbone",
                "path": clip["backbone_dir"],
                "required": False,
                "status": "OK" if _dir_has_files(clip["backbone_dir"])
                or _hf_cache_has(config, clip["backbone"]) else "WARN",
                "size": "",
                "note": "run: download_backbones.py if WARN",
            }
        )

    for name, spec in enabled_models(config).items():
        if spec["type"].startswith("dinov2"):
            ok = _hf_cache_has(config, spec["backbone"])
            rows.append(
                {
                    "item": f"{name} backbone",
                    "path": spec["backbone"],
                    "required": False,
                    "status": "OK" if ok else "WARN",
                    "size": "",
                    "note": "run: download_backbones.py if WARN",
                }
            )

    add("test manifest", config["paths"].get("manifest"), False, "optional (evaluation)")
    add("dataset root", config["paths"].get("dataset_root"), False, "optional (evaluation)")

    # Class mapping content check
    mapping_note = ""
    if rows[0]["status"] == "OK":
        try:
            mapping = load_class_mapping(config)
            mapping_note = f"{mapping.num_classes} classes, valid"
        except ConfigError as exc:
            rows[0]["status"] = "INVALID"
            mapping_note = str(exc)
        rows[0]["note"] = mapping_note

    # Hashes
    if args.hash:
        hash_file = PROJECT_ROOT / "artifacts" / "metadata" / "checksums.json"
        saved = json.loads(hash_file.read_text(encoding="utf-8")) if hash_file.is_file() else {}
        current: Dict[str, str] = {}
        for row in rows:
            p = row["path"]
            if isinstance(p, Path) and p.is_file():
                print(f"hashing {p.name} ...")
                current[row["item"]] = sha256_file(p)
                if row["item"] in saved and saved[row["item"]] != current[row["item"]]:
                    row["status"] = "CHANGED"
                    row["note"] = "hash differs from checksums.json"
        if not saved or args.update_hashes:
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            hash_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
            print(f"Hashes saved to {hash_file}")

    # Print table
    print()
    width = max(len(r["item"]) for r in rows)
    for r in rows:
        extra = f"  {r['size']}" if r["size"] else ""
        note = f"  ({r['note']})" if r["note"] else ""
        print(f"[{r['status']:<7}] {r['item']:<{width}}{extra}{note}")
        if r["status"] in ("MISSING", "INVALID", "CHANGED"):
            print(f"{'':<10}{r['path']}")

    failures = [r for r in rows if r["status"] in ("MISSING", "INVALID", "CHANGED")]
    warnings = [r for r in rows if r["status"] == "WARN"]
    print()
    print(f"Summary: {len(failures)} problem(s), {len(warnings)} warning(s).")
    return 1 if failures else 0


# ==========================================================================
# inspect
# ==========================================================================
ATTENTION_TOKENS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "attention.attention.query", "attention.attention.key", "attention.attention.value",
    "attention.output.dense", "attn.qkv", "attn.proj",
]


def _short(value: Any, limit: int = 300) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


def compare_class_info(value: Any, mapping: ClassMapping) -> Optional[str]:
    """Compare a class list/dict found in a checkpoint with the shared mapping."""
    try:
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            diffs = sum(1 for i, n in enumerate(value) if i >= mapping.num_classes
                        or mapping.names[i] != n)
            if len(value) != mapping.num_classes:
                return f"DIFFERS (length {len(value)} vs {mapping.num_classes})"
            return "MATCHES class_mapping.json order" if diffs == 0 else f"DIFFERS ({diffs} positions)"
        if isinstance(value, dict) and value:
            keys, vals = list(value.keys()), list(value.values())
            if all(isinstance(v, int) for v in vals):
                got = {str(k): int(v) for k, v in value.items()}
                diffs = sum(1 for k, v in got.items() if mapping.class_to_idx.get(k) != v)
                return "MATCHES class_to_idx" if diffs == 0 and len(got) == mapping.num_classes \
                    else f"DIFFERS ({diffs} entries, size {len(got)})"
            if all(isinstance(v, str) for v in vals):
                got_i = {int(k): v for k, v in value.items()}
                diffs = sum(1 for k, v in got_i.items() if mapping.idx_to_class.get(k) != v)
                return "MATCHES idx_to_class" if diffs == 0 and len(got_i) == mapping.num_classes \
                    else f"DIFFERS ({diffs} entries, size {len(got_i)})"
    except (ValueError, TypeError):
        return None
    return None


def describe_checkpoint(path: Path, trust: bool, num_classes: int,
                        mapping: Optional[ClassMapping]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "file": str(path),
        "size": human_size(path.stat().st_size),
        "state_dicts": {},
        "tensors": {},
        "metadata": {},
        "hints": [],
    }

    if path.suffix == ".safetensors":
        shapes = safetensors_shapes(path)
        info["state_dicts"]["<root>"] = {
            "num_tensors": len(shapes),
            "num_params": sum(_prod(s) for s in shapes.values()),
            "prefix_groups": _prefix_groups(list(shapes)),
            "keys": [[k, list(s), "?"] for k, s in shapes.items()],
        }
        _add_hints(info, {"<root>": shapes}, num_classes)
        return info

    obj = torch_load(path, trust)
    info["container_type"] = type(obj).__name__
    parts = walk_checkpoint(obj)

    all_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for name, sd in parts["state_dicts"].items():
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        all_shapes[name] = shapes
        info["state_dicts"][name] = {
            "num_tensors": len(sd),
            "num_params": sum(v.numel() for v in sd.values()),
            "prefix_groups": _prefix_groups(list(sd)),
            "keys": [[k, list(v.shape), str(v.dtype).replace("torch.", "")] for k, v in sd.items()],
        }

    for name, t in parts["tensors"].items():
        entry: Dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
        if t.numel() > 0 and t.is_floating_point():
            entry.update(
                min=round(float(t.min()), 6), max=round(float(t.max()), 6),
                mean=round(float(t.mean()), 6), finite=bool(t.isfinite().all()),
            )
        info["tensors"][name] = entry

    for name, value in parts["metadata"].items():
        info["metadata"][name] = _short(value)
        if mapping is not None and any(w in name.lower() for w in ("class", "label")):
            verdict = compare_class_info(value, mapping)
            if verdict:
                info["hints"].append(f"class info '{name}': {verdict}")

    _add_hints(info, all_shapes, num_classes)
    return info


def _prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _prefix_groups(keys: List[str]) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for k in keys:
        head = k.split(".")[0]
        groups[head] = groups.get(head, 0) + 1
    return groups


def _add_hints(info: Dict[str, Any], all_shapes: Dict[str, Dict[str, Tuple[int, ...]]],
               num_classes: int) -> None:
    for sd_name, shapes in all_shapes.items():
        heads = [(k, s) for k, s in shapes.items() if len(s) == 2 and s[0] == num_classes]
        for k, s in heads:
            info["hints"].append(f"[{sd_name}] {num_classes}-way weight: {k} {list(s)}")
        counts = {t: sum(1 for k in shapes if t in k) for t in ATTENTION_TOKENS}
        counts = {t: c for t, c in counts.items() if c}
        if counts:
            info["hints"].append(f"[{sd_name}] attention key naming counts: {counts}")
        for prefix in ("module.", "model.", "backbone.", "encoder.", "net."):
            n = sum(1 for k in shapes if k.startswith(prefix))
            if n:
                info["hints"].append(f"[{sd_name}] {n} keys start with '{prefix}'")


def render_report(name: str, info: Dict[str, Any], max_keys: Optional[int]) -> List[str]:
    lines = [f"=== {name} ===", f"file: {info['file']}", f"size: {info['size']}"]
    if "container_type" in info:
        lines.append(f"container: {info['container_type']}")
    for sd_name, sd in info["state_dicts"].items():
        lines.append(
            f"state dict '{sd_name}': {sd['num_tensors']} tensors, "
            f"{sd['num_params']:,} parameters"
        )
        lines.append(f"  key groups: {sd['prefix_groups']}")
        keys = sd["keys"] if max_keys is None else sd["keys"][:max_keys]
        for k, shape, dtype in keys:
            lines.append(f"  {k}  {shape}  {dtype}")
        if max_keys is not None and len(sd["keys"]) > max_keys:
            lines.append(f"  ... {len(sd['keys']) - max_keys} more (full list in report file)")
    for t_name, t in info["tensors"].items():
        lines.append(f"tensor '{t_name}': {t}")
    for m_name, m in info["metadata"].items():
        lines.append(f"metadata '{m_name}': {m}")
    for hint in info["hints"]:
        lines.append(f"hint: {hint}")
    lines.append("")
    return lines


def cmd_inspect(config: Dict[str, Any], args: argparse.Namespace) -> int:
    mapping: Optional[ClassMapping] = None
    try:
        mapping = load_class_mapping(config)
    except ConfigError as exc:
        print(f"[note] class mapping not available for comparison: {exc}\n")

    num_classes = config["app"]["num_classes"]
    targets: Dict[str, Path] = {}
    for name, spec in enabled_models(config).items():
        if args.model and name != args.model:
            continue
        if spec["type"] == "clip_mlp":
            mlp_dir: Path = spec["mlp_dir"]
            files = sorted(p for p in mlp_dir.glob("*") if p.suffix in WEIGHT_SUFFIXES) \
                if mlp_dir.is_dir() else []
            if files:
                targets[name] = files[0]
            else:
                print(f"[skip] {name}: no MLP file yet (run: setup_artifacts.py extract)\n")
        else:
            if spec["checkpoint"].is_file():
                targets[name] = spec["checkpoint"]
            else:
                print(f"[skip] {name}: missing {spec['checkpoint']}\n")

    if args.model and not targets:
        raise SetupError(f"Nothing to inspect for model '{args.model}'.")

    console: List[str] = []
    full: List[str] = []
    json_out: Dict[str, Any] = {}
    for name, path in targets.items():
        print(f"Reading {name} ({human_size(path.stat().st_size)}) ...")
        try:
            info = describe_checkpoint(path, args.trust, num_classes, mapping)
        except SetupError as exc:
            print(f"[error] {name}: {exc}\n")
            json_out[name] = {"error": str(exc)}
            continue
        json_out[name] = info
        console += render_report(name, info, args.max_keys)
        full += render_report(name, info, None)

    print()
    print("\n".join(console))

    out_dir: Path = config["paths"]["outputs"] / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_report.json").write_text(
        json.dumps(json_out, indent=2), encoding="utf-8"
    )
    (out_dir / "checkpoint_report.txt").write_text("\n".join(full), encoding="utf-8")
    print(f"Full report saved to {out_dir}")
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract, verify and inspect model artifacts.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="Unpack the nested CLIP archive")
    p_ext.add_argument("--force", action="store_true", help="Replace existing CLIP folders")
    p_ext.add_argument("--keep-staging", action="store_true", help="Keep the temporary folder")
    p_ext.add_argument("--trust", action="store_true", help="Allow weights_only=False loading")

    p_ver = sub.add_parser("verify", help="Check that all required files are present")
    p_ver.add_argument("--hash", action="store_true", help="Compute SHA-256 checksums")
    p_ver.add_argument("--update-hashes", action="store_true", help="Overwrite saved checksums")

    p_ins = sub.add_parser("inspect", help="Dump checkpoint keys, shapes and metadata")
    p_ins.add_argument("--model", default=None, help="Inspect one model only")
    p_ins.add_argument("--max-keys", type=int, default=12, help="Keys shown per state dict")
    p_ins.add_argument("--trust", action="store_true", help="Allow weights_only=False loading")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        handlers = {"extract": cmd_extract, "verify": cmd_verify, "inspect": cmd_inspect}
        return handlers[args.command](config, args)
    except (SetupError, ConfigError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
