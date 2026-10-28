"""Dataset/manifest handling, metrics, evaluation runs and reports.

Split discipline (enforced in code):
  * calibration and abstention thresholds are fitted ONLY on the validation
    split (run_fit_calibration refuses any other split);
  * evaluating the test split only READS calibration.json, it never fits.

Nothing here reports a number that was not computed from the images actually
evaluated. Missing files, unreadable images and failed models are counted and
written to the report, never silently skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from plant_disease import ensemble as ens
from plant_disease.config import ClassMapping, enabled_models, get_logger

log = get_logger("evaluation")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Historical counts from the project notes; NOT proof the data is present.
EXPECTED_SPLIT_COUNTS = {"train": 42848, "val": 10810, "test": 13453}

COLUMN_CANDIDATES = {
    "split": ["split", "subset", "set", "partition"],
    "label": ["label", "class", "class_name", "disease", "target", "category"],
    "path": ["resolved_path", "image_path", "filepath", "file_path", "path",
             "relative_path", "filename"],
}


class EvaluationError(Exception):
    """A problem with the data or setup that the user needs to fix."""


# ==========================================================================
# Manifest and path resolution
# ==========================================================================
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def normalize_split(value: Any) -> str:
    v = str(value).strip().lower()
    if v in ("train", "training"):
        return "train"
    if v in ("val", "valid", "validation", "dev"):
        return "val"
    if v in ("test", "testing"):
        return "test"
    return v


def detect_columns(
    df: pd.DataFrame,
    split_col: Optional[str] = None,
    label_col: Optional[str] = None,
    path_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    lower = {c.lower(): c for c in df.columns}

    def pick(key: str, override: Optional[str]) -> Optional[str]:
        if override:
            if override not in df.columns:
                raise EvaluationError(f"Column '{override}' not in manifest. Columns: {list(df.columns)}")
            return override
        for cand in COLUMN_CANDIDATES[key]:
            if cand in lower:
                return lower[cand]
        return None

    split = pick("split", split_col)
    if split is None:
        raise EvaluationError(f"No split column found. Columns: {list(df.columns)}. Use --split-column.")
    if path_cols:
        for c in path_cols:
            if c not in df.columns:
                raise EvaluationError(f"Path column '{c}' not in manifest. Columns: {list(df.columns)}")
        paths = list(path_cols)
    else:
        paths = [lower[c] for c in COLUMN_CANDIDATES["path"] if c in lower]
    if not paths:
        raise EvaluationError(f"No path column found. Columns: {list(df.columns)}. Use --path-column.")
    return {"split": split, "label": pick("label", label_col), "paths": paths}


class PathResolver:
    """Finds image files without relying on the original Kaggle/Colab paths."""

    def __init__(self, dataset_root: Optional[Path]) -> None:
        self.root = Path(dataset_root) if dataset_root else None
        self._index: Optional[Dict[str, List[Path]]] = None

    @staticmethod
    def _parts(raw: str) -> List[str]:
        parts = PurePosixPath(raw.replace("\\", "/")).parts
        return [p for p in parts if p not in ("/", "\\") and not re.fullmatch(r"[A-Za-z]:", p)]

    def _lookup(self, name: str) -> List[Path]:
        if self._index is None:
            log.info("Indexing dataset folder %s (one time) ...", self.root)
            self._index = {}
            for dirpath, _, files in os.walk(self.root):
                for f in files:
                    if Path(f).suffix.lower() in IMAGE_SUFFIXES:
                        self._index.setdefault(f, []).append(Path(dirpath) / f)
        return self._index.get(name, [])

    def resolve(self, raw_values: Sequence[Any]) -> Tuple[Optional[Path], str]:
        raws = [str(r).strip() for r in raw_values if str(r).strip() and str(r).lower() != "nan"]
        for raw in raws:
            try:
                if Path(raw).is_file():
                    return Path(raw), "as_is"
            except OSError:
                pass
            if self.root is not None:
                parts = self._parts(raw)
                for i in range(len(parts)):
                    cand = self.root.joinpath(*parts[i:])
                    if cand.is_file():
                        return cand, "root_relative" if i == 0 else "root_suffix"
        ambiguous = False
        if self.root is not None and self.root.is_dir():
            for raw in raws:
                parts = self._parts(raw)
                if not parts:
                    continue
                hits = self._lookup(parts[-1])
                if len(hits) == 1:
                    return hits[0], "filename_index"
                if len(hits) > 1:
                    parent = parts[-2] if len(parts) > 1 else ""
                    narrowed = [h for h in hits if h.parent.name == parent]
                    if len(narrowed) == 1:
                        return narrowed[0], "filename_index"
                    ambiguous = True
        return None, "ambiguous" if ambiguous else "missing"


class LabelMapper:
    """Maps manifest labels (names or indices) to class indices via class_mapping.json."""

    def __init__(self, mapping: ClassMapping) -> None:
        self.mapping = mapping
        self.norm = {_norm(n): i for n, i in mapping.class_to_idx.items()}

    def map(self, value: Any) -> Optional[int]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        s = str(value).strip()
        if s in self.mapping.class_to_idx:
            return self.mapping.class_to_idx[s]
        n = _norm(s)
        if n in self.norm:
            return self.norm[n]
        if re.fullmatch(r"\d+", s) and int(s) < self.mapping.num_classes:
            return int(s)
        return None


def prepare_samples(
    manifest_path: Path,
    mapping: ClassMapping,
    dataset_root: Optional[Path],
    splits: Sequence[str],
    limit: Optional[int] = None,
    seed: int = 0,
    split_col: Optional[str] = None,
    label_col: Optional[str] = None,
    path_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve files and labels for the requested splits.

    Returns (rows, report). `rows` has one row per manifest row processed, with
    status ok | missing | ambiguous | bad_label. Nothing is dropped silently:
    the report counts every status.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise EvaluationError(
            f"Manifest not found: {manifest_path}\n"
            "Place it in data\\manifests\\ or pass --manifest."
        )
    df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    cols = detect_columns(df, split_col, label_col, path_cols)
    df["_split"] = df[cols["split"]].map(normalize_split)
    wanted = [normalize_split(s) for s in splits]

    counts = df["_split"].value_counts().to_dict()
    sub = df[df["_split"].isin(wanted)].copy()
    if sub.empty:
        raise EvaluationError(f"No manifest rows for split(s) {wanted}. Splits present: {sorted(counts)}")
    if limit is not None and limit < len(sub):
        sub = sub.sample(n=limit, random_state=seed).sort_index()

    resolver, labels = PathResolver(dataset_root), LabelMapper(mapping)
    records = []
    for idx, row in sub.iterrows():
        path, method = resolver.resolve([row[c] for c in cols["paths"]])
        if cols["label"] is not None:
            label_idx = labels.map(row[cols["label"]])
            label_raw = row[cols["label"]]
        else:
            label_idx = labels.map(path.parent.name) if path is not None else None
            label_raw = path.parent.name if path is not None else ""
        if path is None:
            status = method                                   # missing | ambiguous
        elif label_idx is None:
            status = "bad_label"
        else:
            status = "ok"
        records.append({
            "row_id": int(idx), "split": row["_split"],
            "raw_path": row[cols["paths"][0]], "path": str(path) if path else "",
            "resolution": method, "label_raw": label_raw,
            "label_idx": -1 if label_idx is None else int(label_idx),
            "label_name": mapping.name(label_idx) if label_idx is not None else "",
            "status": status,
        })
    rows = pd.DataFrame.from_records(records)

    per_split = {}
    for s in wanted:
        part = rows[rows["split"] == s]
        per_split[s] = {
            "manifest_rows": int(counts.get(s, 0)),
            "expected_historical": EXPECTED_SPLIT_COUNTS.get(s),
            "processed": int(len(part)),
            **{k: int((part["status"] == k).sum()) for k in ("ok", "missing", "ambiguous", "bad_label")},
        }
    report = {
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "columns_used": cols,
        "manifest_rows_per_split": {k: int(v) for k, v in counts.items()},
        "limit": limit,
        "seed": seed,
        "splits": per_split,
        "resolution_methods": rows["resolution"].value_counts().to_dict(),
        "missing_examples": rows.loc[rows["status"] != "ok", "raw_path"].head(10).tolist(),
        "note": "Expected counts are historical project notes, not proof the data is present.",
    }
    return rows, report


def load_rgb(path: Path | str) -> Image.Image:
    """Open an image with EXIF orientation applied, as RGB."""
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


# ==========================================================================
# Metrics
# ==========================================================================
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_table(cm: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({
        "class": list(class_names), "support": support, "predicted": predicted,
        "precision": precision, "recall": recall, "f1": f1,
    })


def topk_accuracy(proba: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top = np.argpartition(-proba, k - 1, axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        rows.append({
            "bin_low": edges[b], "bin_high": edges[b + 1], "count": n,
            "mean_confidence": float(confidence[m].mean()) if n else np.nan,
            "accuracy": float(correct[m].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(rel: pd.DataFrame) -> float:
    total = rel["count"].sum()
    if total == 0:
        return float("nan")
    filled = rel[rel["count"] > 0]
    return float((filled["count"] / total * (filled["accuracy"] - filled["mean_confidence"]).abs()).sum())


def brier_score(proba: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(proba: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Accept the most confident k samples for k = 1..N; risk = error rate among accepted."""
    order = np.argsort(-confidence, kind="stable")
    c = correct[order].astype(float)
    k = np.arange(1, len(c) + 1)
    return pd.DataFrame({
        "coverage": k / len(c),
        "risk": 1.0 - np.cumsum(c) / k,
        "confidence_threshold": confidence[order],
    })


def selective_summary(correct: np.ndarray, abstain: np.ndarray) -> Dict[str, Any]:
    accepted = ~abstain
    n, n_acc = len(correct), int(accepted.sum())
    errors = ~correct
    return {
        "n": n,
        "accepted": n_acc,
        "coverage": n_acc / n if n else float("nan"),
        "abstention_rate": 1 - n_acc / n if n else float("nan"),
        "selective_accuracy": float(correct[accepted].mean()) if n_acc else None,
        "selective_risk": float(1 - correct[accepted].mean()) if n_acc else None,
        "correct_but_abstained": int((correct & abstain).sum()),
        "errors_caught_fraction": float((errors & abstain).sum() / errors.sum()) if errors.any() else None,
    }


def full_metrics(proba: np.ndarray, labels: np.ndarray, class_names: Sequence[str]) -> Dict[str, Any]:
    """All classification + calibration metrics for one probability matrix [N, C]."""
    C = len(class_names)
    pred = proba.argmax(axis=1)
    correct = pred == labels
    cm = confusion_matrix(labels, pred, C)
    table = per_class_table(cm, class_names)
    present = table["support"] > 0                      # macro over classes that occur in this split
    conf = proba.max(axis=1)
    rel = reliability_bins(conf, correct)
    curve = risk_coverage_curve(conf, correct)
    return {
        "scalars": {
            "n_samples": int(len(labels)),
            "accuracy": float(correct.mean()),
            "macro_precision": float(table.loc[present, "precision"].mean()),
            "macro_recall": float(table.loc[present, "recall"].mean()),
            "macro_f1": float(table.loc[present, "f1"].mean()),
            "balanced_accuracy": float(table.loc[present, "recall"].mean()),
            "top3_accuracy": topk_accuracy(proba, labels, 3),
            "top5_accuracy": topk_accuracy(proba, labels, 5),
            "ece_15bin": expected_calibration_error(rel),
            "brier": brier_score(proba, labels),
            "nll": negative_log_likelihood(proba, labels),
            "mean_confidence": float(conf.mean()),
            "aurc": float(curve["risk"].mean()),
            "classes_with_support": int(present.sum()),
        },
        "confusion": cm, "per_class": table, "reliability": rel, "risk_coverage": curve,
        "correct": correct, "pred": pred,
    }


# ==========================================================================
# Running the models over a split (with caching)
# ==========================================================================
@dataclass
class LogitsCollection:
    samples: pd.DataFrame                       # samples that every model processed
    logits: Dict[str, np.ndarray]               # model -> [N, C], aligned to `samples`
    model_info: Dict[str, Dict[str, Any]]
    failed_images: List[Dict[str, Any]]
    model_errors: Dict[str, str]
    device: str


def _file_stats(path: Any) -> List[Any]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    files = [p] if p.is_file() else sorted(q for q in p.glob("*") if q.is_file())
    return [[q.name, q.stat().st_size, q.stat().st_mtime_ns] for q in files]


def model_signature(spec: Dict[str, Any]) -> str:
    """Fingerprint of a model's files and preprocessing; changes invalidate cached logits."""
    payload = {
        "type": spec.get("type"), "pre": spec.get("preprocessing"),
        "timm": spec.get("timm_name"), "backbone": spec.get("backbone"),
        "files": {k: _file_stats(spec.get(k)) for k in ("checkpoint", "mlp_dir", "backbone_dir")
                  if spec.get(k) is not None},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _read_cache(path: Path, signature: str, expected_ids: set) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["signature"]) != signature:
                return None
            ids = z["row_ids"]
            failed = json.loads(str(z["failed_json"]))
            id_set = set(ids.tolist())
            failed_ids = {int(f["row_id"]) for f in failed}
            if len(id_set) != len(ids) or id_set & failed_ids or (id_set | failed_ids) != expected_ids:
                return None
            return {
                "row_ids": ids, "logits": z["logits"], "failed": failed,
                "load_seconds": float(z["load_seconds"]), "seconds": float(z["seconds"]),
                "n_images": int(z["n_images"]),
            }
    except Exception:  # noqa: BLE001 - any problem means "recompute"
        return None


def _write_cache(path: Path, signature: str, entry: Dict[str, Any]) -> None:
    np.savez_compressed(
        path, logits=entry["logits"].astype(np.float32), row_ids=entry["row_ids"],
        signature=np.array(signature), failed_json=np.array(json.dumps(entry["failed"])),
        load_seconds=np.array(entry["load_seconds"]), seconds=np.array(entry["seconds"]),
        n_images=np.array(entry["n_images"]),
    )


def collect_logits(
    config: Dict[str, Any],
    mapping: ClassMapping,
    samples: pd.DataFrame,
    split: str,
    batch_size: int = 16,
    device: Optional[str] = None,
    use_cache: bool = True,
    only_models: Optional[List[str]] = None,
) -> LogitsCollection:
    """Logits of every enabled model for `samples`. Each image is loaded once per batch.

    Per-model results are cached in outputs/evaluation/cache/, so re-running an
    evaluation (or fitting calibration after evaluating) does not repeat inference
    unless the model files, preprocessing or sample list changed.
    """
    specs = enabled_models(config)
    if only_models:
        unknown = [n for n in only_models if n not in specs]
        if unknown:
            raise EvaluationError(f"Unknown or disabled model(s): {unknown}. Available: {list(specs)}")
        specs = {n: specs[n] for n in only_models}

    samples = samples.reset_index(drop=True)
    expected_ids = set(samples["row_id"].astype(int).tolist())
    cache_dir = Path(config["paths"]["outputs"]) / "evaluation" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: Dict[str, Dict[str, Any]] = {}
    signatures: Dict[str, str] = {}
    need: List[str] = []
    for name, spec in specs.items():
        signatures[name] = model_signature(spec)
        hit = _read_cache(cache_dir / f"{split}__{name}.npz", signatures[name], expected_ids) if use_cache else None
        if hit is not None:
            cached[name] = hit
            log.info("Using cached logits for %s (%s split)", name, split)
        else:
            need.append(name)

    computed: Dict[str, Dict[str, Any]] = {}
    model_errors: Dict[str, str] = {}
    display = {n: s.get("display_name", n) for n, s in specs.items()}
    device_used = device or config["app"].get("device", "auto")
    load_seconds: Dict[str, float] = {}

    if need:
        from plant_disease.models import load_models      # lazy: needs PyTorch

        registry = load_models(config, mapping, device=device, only=need)
        device_used = registry.device
        model_errors.update(registry.failures)
        adapters = dict(registry.adapters)
        load_seconds = {n: (a.load_seconds or 0.0) for n, a in adapters.items()}
        state = {n: {"chunks": [], "ids": [], "seconds": 0.0, "n": 0} for n in adapters}
        warmed: set = set()
        failed_images: List[Dict[str, Any]] = []
        n_batches = (len(samples) + batch_size - 1) // batch_size
        log.info("Running %d model(s) on %d images (%d batches) ...", len(adapters), len(samples), n_batches)

        for b, start in enumerate(range(0, len(samples), batch_size), 1):
            images, ids = [], []
            for _, row in samples.iloc[start:start + batch_size].iterrows():
                try:
                    images.append(load_rgb(row["path"]))
                    ids.append(int(row["row_id"]))
                except Exception as exc:  # noqa: BLE001
                    failed_images.append({"row_id": int(row["row_id"]), "path": row["path"],
                                          "error": f"{type(exc).__name__}: {exc}"})
            if not images:
                continue
            for name, adapter in adapters.items():
                if name in model_errors:
                    continue
                try:
                    if name not in warmed:
                        adapter.predict_logits(images[:1])          # warm-up, not timed
                        warmed.add(name)
                    t0 = time.perf_counter()
                    out = adapter.predict_logits(images, batch_size=len(images))
                    state[name]["seconds"] += time.perf_counter() - t0
                    state[name]["chunks"].append(out)
                    state[name]["ids"].extend(ids)
                    state[name]["n"] += len(images)
                except Exception as exc:  # noqa: BLE001
                    model_errors[name] = f"inference failed: {exc}"
                    log.error("Model %s failed during evaluation: %s", name, exc)
            if b % 20 == 0 or b == n_batches:
                log.info("  batch %d/%d", b, n_batches)

        for name in adapters:
            if name in model_errors:
                continue
            st = state[name]
            if st["n"] == 0:
                model_errors[name] = "no images could be processed"
                continue
            entry = {
                "row_ids": np.array(st["ids"], dtype=np.int64),
                "logits": np.concatenate(st["chunks"], axis=0),
                "failed": failed_images, "load_seconds": load_seconds[name],
                "seconds": st["seconds"], "n_images": st["n"],
            }
            computed[name] = entry
            _write_cache(cache_dir / f"{split}__{name}.npz", signatures[name], entry)

    results = {**cached, **computed}
    if not results:
        raise EvaluationError(f"No model produced logits. Errors: {model_errors}")

    # Keep only the samples that every contributing model processed
    common = set.intersection(*(set(r["row_ids"].tolist()) for r in results.values()))
    kept_ids = [int(r) for r in samples["row_id"] if int(r) in common]
    aligned: Dict[str, np.ndarray] = {}
    info: Dict[str, Dict[str, Any]] = {}
    for name, r in results.items():
        pos = {int(rid): i for i, rid in enumerate(r["row_ids"])}
        aligned[name] = r["logits"][[pos[i] for i in kept_ids]]
        secs, n_img = r["seconds"], r["n_images"]
        info[name] = {
            "display_name": display[name],
            "source": "cache" if name in cached else "computed",
            "load_seconds": r["load_seconds"],
            "latency_ms_per_image": 1000.0 * secs / n_img if n_img else None,
            "throughput_img_per_s": n_img / secs if secs > 0 else None,
            "timing_note": "includes preprocessing and forward pass; excludes image decoding",
        }
    failed_union = {int(f["row_id"]): f for r in results.values() for f in r["failed"]}
    kept = samples[samples["row_id"].astype(int).isin(kept_ids)].reset_index(drop=True)
    return LogitsCollection(kept, aligned, info, list(failed_union.values()), model_errors, device_used)


# ==========================================================================
# Helpers for reports
# ==========================================================================
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def environment_metadata(config: Dict[str, Any], device: str) -> Dict[str, Any]:
    packages = {}
    for pkg in ("torch", "torchvision", "timm", "transformers", "numpy", "pandas", "Pillow", "scikit-learn"):
        try:
            packages[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            packages[pkg] = "not installed"
    cfg_path = config.get("_config_path")
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "device": device, "packages": packages,
        "config_file": str(cfg_path) if cfg_path else None,
        "config_text": Path(cfg_path).read_text(encoding="utf-8") if cfg_path and Path(cfg_path).is_file() else None,
    }


def _save_plots(run_dir: Path, class_names: Sequence[str], decision_name: str,
                results: Dict[str, Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; plots skipped.")
        return []
    saved = []
    dec = results[decision_name]

    cm = dec["confusion"].astype(float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized): {decision_name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    saved.append("confusion_matrix.png")

    rel = dec["reliability"]
    filled = rel[rel["count"] > 0]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.bar(filled["bin_low"], filled["accuracy"], width=(1 / len(rel)), align="edge",
           alpha=0.7, edgecolor="k", label="accuracy per bin")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram: {decision_name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(run_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)
    saved.append("reliability_diagram.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, res in results.items():
        curve = res["risk_coverage"]
        ax.plot(curve["coverage"], curve["risk"], label=name, linewidth=2 if name == decision_name else 1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (error rate among accepted)")
    ax.set_title("Risk-coverage curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)
    saved.append("risk_coverage.png")
    return saved


# ==========================================================================
# Evaluation run
# ==========================================================================
def run_evaluation(
    config: Dict[str, Any],
    mapping: ClassMapping,
    split: str,
    manifest_path: Path,
    dataset_root: Optional[Path],
    batch_size: int = 16,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    only_models: Optional[List[str]] = None,
    column_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate every model and the ensemble on one split and write all outputs.

    Reads calibration.json if present (fitted earlier on validation) but never fits.
    """
    split = normalize_split(split)
    co = column_overrides or {}
    rows, manifest_report = prepare_samples(manifest_path, mapping, dataset_root, [split], limit=limit, **co)
    ok = rows[rows["status"] == "ok"]
    if ok.empty:
        raise EvaluationError("No usable samples. Manifest report:\n" + json.dumps(_jsonable(manifest_report), indent=2))

    coll = collect_logits(config, mapping, ok, split, batch_size, device, use_cache, only_models)
    names = mapping.names
    labels = coll.samples["label_idx"].to_numpy(dtype=np.int64)
    n_eval = len(labels)

    calibration = ens.load_calibration(ens.calibration_path(config))
    policy = ens.AbstentionPolicy.from_config(config, calibration)
    weights = config["ensemble"].get("weights")
    raw = ens.apply_temperatures(coll.logits, None)          # plain softmax per model
    ens_temp = ens.ensemble_temperature_from(calibration)    # validation-fitted, or None
    models = list(coll.logits)

    results: Dict[str, Dict[str, Any]] = {m: full_metrics(raw[m], labels, names) for m in models}
    ens_raw, _ = ens.combine_probabilities(raw, weights)
    results["ensemble_uncalibrated"] = full_metrics(ens_raw, labels, names)
    decision_name, decision_ens = "ensemble_uncalibrated", ens_raw
    if ens_temp is not None:
        ens_cal = ens.calibrate_ensemble(ens_raw, ens_temp)
        results["ensemble_calibrated"] = full_metrics(ens_cal, labels, names)
        decision_name, decision_ens = "ensemble_calibrated", ens_cal

    um = ens.uncertainty_metrics(decision_ens)
    dis = ens.disagreement_metrics(raw, decision_ens)
    arrays = {"confidence": um["confidence"], "entropy": um["entropy"],
              "margin": um["margin"], "disagreement": dis["disagreement"]}
    abstain, triggers = ens.abstain_mask(arrays, policy, len(models))
    correct = results[decision_name]["correct"]
    selective = selective_summary(correct, abstain)
    selective["trigger_counts"] = {k: int(v.sum()) for k, v in triggers.items()}
    selective["note"] = "image-quality abstention is not applied in offline evaluation"

    # ---- tables -------------------------------------------------------------
    table_rows = []
    for name, res in results.items():
        is_model = name in coll.model_info
        info = coll.model_info.get(name, {})
        row = {"name": name, "kind": "model" if is_model else "ensemble",
               "display_name": info.get("display_name", name), **res["scalars"],
               "load_seconds": info.get("load_seconds"),
               "latency_ms_per_image": info.get("latency_ms_per_image"),
               "throughput_img_per_s": info.get("throughput_img_per_s"),
               "decision_ensemble": name == decision_name}
        if name == decision_name:
            row.update({k: selective[k] for k in ("coverage", "abstention_rate", "selective_accuracy", "selective_risk")})
        table_rows.append(row)
    summary_df = pd.DataFrame(table_rows)

    pred_df = coll.samples[["row_id", "path", "label_name"]].copy()
    for m in models:
        pred_df[f"pred_{m}"] = [names[i] for i in results[m]["pred"]]
    pred_df["pred_ensemble"] = [names[i] for i in um["top1"]]
    pred_df["confidence"] = um["confidence"]
    pred_df["margin"] = um["margin"]
    pred_df["entropy"] = um["entropy"]
    pred_df["disagreement"] = dis["disagreement"]
    pred_df["abstain"] = abstain
    pred_df["abstain_reasons"] = [
        ";".join(k for k, v in triggers.items() if v[i]) for i in range(n_eval)
    ]
    pred_df["ensemble_correct"] = correct

    problems = rows[rows["status"] != "ok"][["row_id", "split", "raw_path", "status", "resolution"]].copy()
    problems["error"] = ""
    img_fail = pd.DataFrame(coll.failed_images)
    if not img_fail.empty:
        problems = pd.concat([problems, pd.DataFrame({
            "row_id": img_fail["row_id"], "split": split, "raw_path": img_fail["path"],
            "status": "unreadable_image", "resolution": "", "error": img_fail["error"]})], ignore_index=True)

    # ---- write ----------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"]["outputs"]) / "evaluation" / f"{split}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(run_dir / "metrics_summary.csv", index=False)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    problems.to_csv(run_dir / "failed_or_missing_samples.csv", index=False)
    for name, res in results.items():
        res["per_class"].to_csv(run_dir / f"per_class_{name}.csv", index=False)
        pd.DataFrame(res["confusion"], index=names, columns=names).to_csv(run_dir / f"confusion_{name}.csv")
        res["reliability"].to_csv(run_dir / f"reliability_{name}.csv", index=False)
        res["risk_coverage"].to_csv(run_dir / f"risk_coverage_{name}.csv", index=False)
    plots = _save_plots(run_dir, names, decision_name, results)

    counts = {
        "manifest_rows_in_split": manifest_report["splits"][split]["manifest_rows"],
        "processed_rows": int(len(rows)),
        "resolved_ok": int(len(ok)),
        "evaluated": n_eval,
        "missing_or_unusable": int(len(rows) - len(ok)),
        "unreadable_images": int(len(coll.failed_images)),
        "limit": limit,
    }
    summary = {
        "split": split,
        "sample_counts": counts,
        "models_evaluated": models,
        "models_failed": coll.model_errors,
        "decision_ensemble": decision_name,
        "calibration_applied": ens_temp is not None,
        "ensemble_temperature": ens_temp,
        "calibration_in_sample": ens_temp is not None and split == "val",
        "abstention_policy": {"status": policy.status, "thresholds": {
            "min_confidence": policy.min_confidence, "max_entropy": policy.max_entropy,
            "min_margin": policy.min_margin, "max_disagreement": policy.max_disagreement,
            "min_models": policy.min_models}},
        "selective": selective,
        "metrics": {r["name"]: {k: r[k] for k in r if k not in ("name",)} for r in table_rows},
        "model_info": coll.model_info,
        "plots": plots,
        "output_dir": str(run_dir),
        "manifest_report": manifest_report,
        "warnings": [],
    }
    if split != "test":
        summary["warnings"].append(f"Evaluated on '{split}', not the held-out test split.")
    if len(models) < len(enabled_models(config)):
        summary["warnings"].append("Some enabled models were not evaluated; see models_failed.")
    if calibration and ens_temp is not None and set(calibration.get("models", models)) != set(models):
        summary["warnings"].append(
            f"Calibration was fitted with models {calibration.get('models')} but this run used "
            f"{models}; re-run --fit-calibration for this model set.")
    if counts["missing_or_unusable"] or counts["unreadable_images"]:
        summary["warnings"].append(
            f"{counts['missing_or_unusable']} manifest rows unusable and "
            f"{counts['unreadable_images']} images unreadable; results cover {n_eval} samples only.")
    if limit:
        summary["warnings"].append(f"Run limited to {limit} randomly chosen rows; not the full split.")
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    (run_dir / "manifest_report.json").write_text(json.dumps(_jsonable(manifest_report), indent=2), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(_jsonable({**environment_metadata(config, coll.device),
                              "settings": {"split": split, "batch_size": batch_size, "limit": limit,
                                           "use_cache": use_cache, "only_models": only_models}}), indent=2),
        encoding="utf-8")
    summary["table"] = summary_df
    return summary


# ==========================================================================
# Calibration fitting (validation split only)
# ==========================================================================
def run_fit_calibration(
    config: Dict[str, Any],
    mapping: ClassMapping,
    manifest_path: Path,
    dataset_root: Optional[Path],
    batch_size: int = 16,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    only_models: Optional[List[str]] = None,
    target_accuracy: float = 0.99,
    min_coverage: float = 0.05,
    column_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit per-model temperatures and abstention thresholds on the VALIDATION split only.

    The split is fixed to validation inside this function; there is no parameter
    that can point it at the test set.
    """
    split = "val"
    rows, manifest_report = prepare_samples(manifest_path, mapping, dataset_root, [split],
                                            limit=limit, **(column_overrides or {}))
    ok = rows[rows["status"] == "ok"]
    if ok.empty:
        raise EvaluationError("No usable validation samples. " + json.dumps(_jsonable(manifest_report["splits"])))
    coll = collect_logits(config, mapping, ok, split, batch_size, device, use_cache, only_models)
    labels = coll.samples["label_idx"].to_numpy(dtype=np.int64)
    names = mapping.names
    models = list(coll.logits)
    weights = config["ensemble"].get("weights")

    temps = {m: ens.fit_temperature(coll.logits[m], labels) for m in models}
    raw = ens.apply_temperatures(coll.logits, None)
    cal = ens.apply_temperatures(coll.logits, temps)
    before_after = {}
    for m in models:
        b, a = full_metrics(raw[m], labels, names)["scalars"], full_metrics(cal[m], labels, names)["scalars"]
        before_after[m] = {"temperature": temps[m],
                           "nll_before": b["nll"], "nll_after": a["nll"],
                           "ece_before": b["ece_15bin"], "ece_after": a["ece_15bin"]}

    # Decision calibration: one temperature fitted on the averaged ensemble itself
    ens_raw, _ = ens.combine_probabilities(raw, weights)
    ens_temp = ens.fit_ensemble_temperature(ens_raw, labels)
    ens_cal = ens.calibrate_ensemble(ens_raw, ens_temp)
    eb, ea = full_metrics(ens_raw, labels, names)["scalars"], full_metrics(ens_cal, labels, names)["scalars"]
    ensemble_before_after = {"temperature": ens_temp,
                             "nll_before": eb["nll"], "nll_after": ea["nll"],
                             "ece_before": eb["ece_15bin"], "ece_after": ea["ece_15bin"]}
    um = ens.uncertainty_metrics(ens_cal)
    dis = ens.disagreement_metrics(raw, ens_cal)
    arrays = {"confidence": um["confidence"], "entropy": um["entropy"],
              "margin": um["margin"], "disagreement": dis["disagreement"]}
    correct = um["top1"] == labels
    fit = ens.fit_abstention_thresholds(arrays, correct, target_accuracy, min_coverage)

    path = ens.calibration_path(config)
    data = ens.save_calibration(path, temps, fit, len(labels), ensemble_temperature=ens_temp)
    data.update({"models": models, "ensemble_weights": weights, "limit": limit,
                 "split": "val", "manifest": str(manifest_path)})
    path.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")

    policy = ens.AbstentionPolicy.from_config(config, data)
    mask, triggers = ens.abstain_mask(arrays, policy, len(models))
    combined = selective_summary(correct, mask)
    report = {
        "calibration_file": str(path),
        "n_validation_samples": int(len(labels)),
        "models": models, "models_failed": coll.model_errors,
        "per_model": before_after,
        "ensemble": ensemble_before_after,
        "ensemble_accuracy_validation": float(correct.mean()),
        "thresholds_fitted": fit["thresholds"],
        "threshold_report": fit["report"],
        "policy_status": policy.status,
        "combined_policy_on_validation": combined,
        "trigger_counts": {k: int(v.sum()) for k, v in triggers.items()},
        "warnings": [
            "These numbers are in-sample on the validation split. Evaluate on the test split to check them.",
        ],
    }
    if len(models) < len(enabled_models(config)):
        report["warnings"].append("Not all enabled models were available; thresholds fit the subset only.")
    (path.parent / "fit_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def check_manifest(
    config: Dict[str, Any],
    mapping: ClassMapping,
    manifest_path: Path,
    dataset_root: Optional[Path],
    limit: Optional[int] = None,
    column_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve every file for all three splits without running any model."""
    _, report = prepare_samples(manifest_path, mapping, dataset_root, ["train", "val", "test"],
                                limit=limit, **(column_overrides or {}))
    out = Path(config["paths"]["outputs"]) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_check.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    report["saved_to"] = str(out / "manifest_check.json")
    return report
