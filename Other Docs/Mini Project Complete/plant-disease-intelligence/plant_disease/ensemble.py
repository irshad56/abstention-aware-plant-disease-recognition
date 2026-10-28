"""Ensemble prediction, uncertainty, disagreement, abstention and calibration.

Pure NumPy: no model or PyTorch code lives here.

Typical single-image use (the pipeline does this):

    result = run_ensemble(per_model_proba, class_names, policy,
                          weights=config["ensemble"]["weights"],
                          models_failed=registry.failures)
    result.decision.abstain      # True -> do not report a disease
    result.top1_name, result.confidence, result.entropy, ...

Everything that selects numbers (temperatures, thresholds, weights) must be
fitted on VALIDATION data only. Softmax outputs are model scores, not
calibrated probabilities, unless temperatures fitted on validation data are
applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ABSTAIN_MESSAGE = "I cannot confidently identify the disease from this image."
SCORE_THRESHOLD_FIELDS = ("min_confidence", "max_entropy", "min_margin", "max_disagreement")
_EPS = 1e-12


class EnsembleError(Exception):
    """Invalid probabilities, weights or shapes."""


# ==========================================================================
# Basic numerics
# ==========================================================================
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(proba: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in nats (0 for a one-hot vector, ln(C) for uniform)."""
    p = np.asarray(proba, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log(safe)).sum(axis=axis)


def top_k(proba: np.ndarray, names: Sequence[str], k: int = 3) -> List[Tuple[str, float]]:
    """Top-k (name, probability) for a single probability vector."""
    order = np.argsort(proba)[::-1][:k]
    return [(names[i], float(proba[i])) for i in order]


def uncertainty_metrics(proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-row confidence, top-two margin, entropy and top-1 index for [N, C]."""
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    order = np.argsort(-p, axis=1)
    rows = np.arange(p.shape[0])
    top1 = order[:, 0]
    confidence = p[rows, top1]
    second = p[rows, order[:, 1]] if p.shape[1] > 1 else np.zeros(p.shape[0])
    return {
        "top1": top1,
        "confidence": confidence,
        "margin": confidence - second,
        "entropy": entropy(p),
    }


# ==========================================================================
# Combining models
# ==========================================================================
def validate_probabilities(proba: np.ndarray, num_classes: int, label: str) -> Optional[str]:
    """Return an error string if the array is not a valid probability array, else None."""
    p = np.asarray(proba)
    if p.ndim not in (1, 2) or p.shape[-1] != num_classes:
        return f"{label}: shape {p.shape}, expected (..., {num_classes})"
    if not np.isfinite(p).all():
        return f"{label}: contains non-finite values"
    if (p < -1e-9).any() or (p > 1 + 1e-6).any():
        return f"{label}: values outside [0, 1]"
    if not np.allclose(p.sum(axis=-1), 1.0, atol=1e-4):
        return f"{label}: rows do not sum to 1"
    return None


def effective_weights(names: Sequence[str], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Normalized weights for exactly the models that contributed.

    weights=None -> equal weights. Otherwise every contributing model needs a
    non-negative weight and the weights of contributing models must not all be 0.
    """
    if not names:
        raise EnsembleError("No models contributed.")
    if weights is None:
        return {n: 1.0 / len(names) for n in names}
    missing = [n for n in names if n not in weights]
    if missing:
        raise EnsembleError(f"No ensemble weight given for model(s): {missing}")
    raw = {n: float(weights[n]) for n in names}
    if any(w < 0 or not np.isfinite(w) for w in raw.values()):
        raise EnsembleError("Ensemble weights must be finite and non-negative.")
    total = sum(raw.values())
    if total <= 0:
        raise EnsembleError("Ensemble weights of contributing models sum to zero.")
    return {n: w / total for n, w in raw.items()}


def combine_probabilities(
    per_model: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Weighted average of per-model probabilities ([C] or [N, C]).

    Returns (ensemble_probabilities, effective_weights).
    """
    names = list(per_model)
    w = effective_weights(names, weights)
    first = np.asarray(per_model[names[0]], dtype=np.float64)
    out = np.zeros_like(first)
    for name in names:
        arr = np.asarray(per_model[name], dtype=np.float64)
        if arr.shape != first.shape:
            raise EnsembleError(f"Shape mismatch: {name} {arr.shape} vs {first.shape}")
        out += w[name] * arr
    return out, w


def disagreement_metrics(per_model: Dict[str, np.ndarray], ensemble: np.ndarray) -> Dict[str, np.ndarray]:
    """How much the models disagree, per row ([N, C] inputs).

    disagreement      fraction of models whose top-1 differs from the ensemble top-1
    mutual_information H(mean) - mean(H): spread of the models' distributions (nats)
    all_agree         every model has the same top-1
    """
    stack = np.stack([np.atleast_2d(np.asarray(p, dtype=np.float64)) for p in per_model.values()])
    ens = np.atleast_2d(np.asarray(ensemble, dtype=np.float64))
    model_top1 = stack.argmax(axis=2)                       # [M, N]
    ens_top1 = ens.argmax(axis=1)                           # [N]
    disagreement = (model_top1 != ens_top1[None, :]).mean(axis=0)
    mi = entropy(stack.mean(axis=0)) - entropy(stack, axis=2).mean(axis=0)
    return {
        "disagreement": disagreement,
        "mutual_information": np.maximum(mi, 0.0),
        "all_agree": (model_top1 == model_top1[0:1]).all(axis=0),
        "model_top1": model_top1,
    }


# ==========================================================================
# Calibration (temperature scaling) - validation data only
# ==========================================================================
def _nll_at_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, t_min: float = 0.05, t_max: float = 20.0) -> float:
    """Temperature T minimizing validation NLL of softmax(logits / T).

    Call ONLY with validation-split logits and labels.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or y.shape != (z.shape[0],) or z.shape[0] == 0:
        raise EnsembleError("fit_temperature needs logits [N, C] and labels [N] with N > 0.")
    lo, hi = np.log(t_min), np.log(t_max)
    inv_phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = _nll_at_temperature(z, y, np.exp(c)), _nll_at_temperature(z, y, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _nll_at_temperature(z, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _nll_at_temperature(z, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def apply_temperatures(
    per_model_logits: Dict[str, np.ndarray], temperatures: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """Softmax of each model's logits, divided by its fitted temperature (1.0 if none)."""
    out = {}
    for name, logits in per_model_logits.items():
        t = float((temperatures or {}).get(name, 1.0))
        if not np.isfinite(t) or t <= 0:
            raise EnsembleError(f"Invalid temperature for {name}: {t}")
        out[name] = softmax(np.asarray(logits, dtype=np.float64) / t)
    return out


def calibrate_ensemble(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale ensemble probabilities: p^(1/T), renormalized.

    This is the calibration used for decisions. Averaging models that were each
    calibrated separately tends to under-confidence, so one temperature is fitted
    on the averaged ensemble itself (validation data only). T=1 changes nothing.
    """
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise EnsembleError(f"Invalid ensemble temperature: {t}")
    p = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1.0)
    return softmax(np.log(p) / t)


def fit_ensemble_temperature(ensemble_proba: np.ndarray, labels: np.ndarray) -> float:
    """Temperature for the averaged ensemble. Call ONLY with validation data."""
    p = np.clip(np.asarray(ensemble_proba, dtype=np.float64), _EPS, 1.0)
    return fit_temperature(np.log(p), labels)


def ensemble_temperature_from(calibration: Optional[Dict[str, Any]]) -> Optional[float]:
    """The saved ensemble temperature, only if it was fitted on the validation split."""
    if calibration and calibration.get("fitted_on") == "validation":
        value = calibration.get("ensemble_temperature")
        if value:
            return float(value)
    return None


# ==========================================================================
# Abstention policy
# ==========================================================================
@dataclass(frozen=True)
class AbstentionPolicy:
    """Rules for declining to give a diagnosis.

    The thresholds start as PROVISIONAL, UNCALIBRATED values from config.yaml.
    Fitted values (from validation data) override them via calibration.json;
    `status` reports which thresholds are fitted.
    """

    min_confidence: float
    max_entropy: float
    min_margin: float
    max_disagreement: float
    min_models: int = 3
    abstain_on_poor_quality: bool = True
    fitted_fields: Tuple[str, ...] = ()

    @property
    def status(self) -> str:
        fitted = [f for f in SCORE_THRESHOLD_FIELDS if f in self.fitted_fields]
        if len(fitted) == len(SCORE_THRESHOLD_FIELDS):
            return "validated (thresholds fitted on the validation split)"
        if fitted:
            return f"partially validated (fitted: {', '.join(fitted)}; others provisional)"
        return "provisional (uncalibrated thresholds)"

    @classmethod
    def from_config(cls, config: Dict[str, Any], calibration: Optional[Dict[str, Any]] = None) -> "AbstentionPolicy":
        a = config["abstention"]
        values = {
            "min_confidence": float(a["min_confidence"]),
            "max_entropy": float(a["max_entropy"]),
            "min_margin": float(a["min_margin"]),
            "max_disagreement": float(a["max_disagreement"]),
        }
        fitted: List[str] = []
        if calibration and calibration.get("fitted_on") == "validation":
            for key, val in (calibration.get("thresholds") or {}).items():
                if key in values and val is not None:
                    values[key] = float(val)
                    fitted.append(key)
        return cls(
            **values,
            min_models=int(config["ensemble"].get("min_models", 3)),
            abstain_on_poor_quality=bool(a.get("abstain_on_poor_quality", True)),
            fitted_fields=tuple(fitted),
        )


@dataclass
class AbstentionDecision:
    abstain: bool
    codes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    policy_status: str = ""

    @property
    def message(self) -> str:
        return ABSTAIN_MESSAGE if self.abstain else ""


def trigger_masks(
    confidence: np.ndarray,
    entropy_: np.ndarray,
    margin: np.ndarray,
    disagreement: np.ndarray,
    policy: AbstentionPolicy,
) -> Dict[str, np.ndarray]:
    """Boolean arrays: True where each threshold rule says 'abstain'."""
    return {
        "low_confidence": np.asarray(confidence) < policy.min_confidence,
        "high_entropy": np.asarray(entropy_) > policy.max_entropy,
        "small_margin": np.asarray(margin) < policy.min_margin,
        "high_disagreement": np.asarray(disagreement) > policy.max_disagreement,
    }


def abstain_mask(
    metrics: Dict[str, np.ndarray],
    policy: AbstentionPolicy,
    n_models_used: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Vectorized policy for evaluation. metrics needs confidence, entropy, margin, disagreement."""
    masks = trigger_masks(
        metrics["confidence"], metrics["entropy"], metrics["margin"], metrics["disagreement"], policy
    )
    n = len(metrics["confidence"])
    masks["too_few_models"] = np.full(n, n_models_used < policy.min_models)
    combined = np.zeros(n, dtype=bool)
    for m in masks.values():
        combined |= m
    return combined, masks


_REASON_TEXT = {
    "no_model_output": "No model produced a usable prediction.",
    "too_few_models": "Too few models produced a valid prediction ({used} of {expected}; at least {min} required).",
    "invalid_image": "The image could not be used.",
    "poor_image_quality": "The image quality looks poor (blurry, too dark/bright or low contrast).",
    "inconsistent_output": "The combined probabilities failed a consistency check.",
    "low_confidence": "Top-class probability {confidence:.2f} is below the minimum {min_confidence:.2f}.",
    "high_entropy": "Prediction entropy {entropy:.2f} nats is above the maximum {max_entropy:.2f}.",
    "small_margin": "Top-two probability margin {margin:.2f} is below the minimum {min_margin:.2f}.",
    "high_disagreement": "Models disagree: {disagreement:.0%} differ from the ensemble (maximum {max_disagreement:.0%}).",
}


# ==========================================================================
# Single-image ensemble
# ==========================================================================
@dataclass
class EnsembleResult:
    class_names: List[str]
    probabilities: Optional[np.ndarray]
    top1_index: Optional[int]
    top1_name: Optional[str]
    confidence: Optional[float]
    margin: Optional[float]
    entropy: Optional[float]
    top_k: List[Tuple[str, float]]
    models_used: List[str]
    excluded: Dict[str, str]
    effective_weights: Dict[str, float]
    per_model: List[Dict[str, Any]]
    disagreement: Dict[str, Any]
    calibrated: bool
    decision: AbstentionDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top1_name": self.top1_name,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "top_k": [{"class": n, "probability": p} for n, p in self.top_k],
            "models_used": self.models_used,
            "excluded": self.excluded,
            "effective_weights": self.effective_weights,
            "per_model": self.per_model,
            "disagreement": self.disagreement,
            "calibrated": self.calibrated,
            "abstain": self.decision.abstain,
            "abstain_codes": self.decision.codes,
            "abstain_reasons": self.decision.reasons,
            "policy_status": self.decision.policy_status,
        }


def run_ensemble(
    per_model_proba: Dict[str, np.ndarray],
    class_names: Sequence[str],
    policy: AbstentionPolicy,
    weights: Optional[Dict[str, float]] = None,
    models_failed: Optional[Dict[str, str]] = None,
    image_ok: bool = True,
    quality_poor: bool = False,
    calibrated: bool = False,
    ensemble_temperature: Optional[float] = None,
    k: int = 3,
) -> EnsembleResult:
    """Combine per-model probability vectors for ONE image and decide accept/abstain.

    per_model_proba holds each model's plain softmax scores. If ensemble_temperature
    (fitted on validation data) is given, it is applied to the averaged ensemble.

    Models whose output is invalid are excluded and reported in `excluded`,
    never silently dropped or replaced.
    """
    names = list(class_names)
    excluded: Dict[str, str] = dict(models_failed or {})
    usable: Dict[str, np.ndarray] = {}
    for name, proba in per_model_proba.items():
        problem = validate_probabilities(np.asarray(proba), len(names), name)
        if problem:
            excluded[name] = problem
        else:
            usable[name] = np.asarray(proba, dtype=np.float64).reshape(-1)

    expected = len(usable) + len(excluded)
    codes: List[str] = []
    ctx: Dict[str, Any] = {"used": len(usable), "expected": expected, "min": policy.min_models}

    if not image_ok:
        codes.append("invalid_image")
    if quality_poor and policy.abstain_on_poor_quality and image_ok:
        codes.append("poor_image_quality")

    if not usable:
        if image_ok:                       # an invalid image is the reason; no model was run
            codes.append("no_model_output")
        decision = AbstentionDecision(
            True, codes, [_REASON_TEXT[c].format(**ctx) for c in codes], policy.status
        )
        return EnsembleResult(
            names, None, None, None, None, None, None, [], [], excluded, {}, [], {}, calibrated, decision
        )

    ens, eff = combine_probabilities(usable, weights)
    ens = ens / ens.sum()                                  # remove tiny float drift
    if ensemble_temperature is not None:
        ens = calibrate_ensemble(ens, ensemble_temperature)
        calibrated = True
    metrics = uncertainty_metrics(ens)
    dis = disagreement_metrics(usable, ens)
    top1 = int(metrics["top1"][0])
    conf, margin, ent = (float(metrics[x][0]) for x in ("confidence", "margin", "entropy"))
    disagreement = float(dis["disagreement"][0])

    if validate_probabilities(ens, len(names), "ensemble"):
        codes.append("inconsistent_output")
    if len(usable) < policy.min_models:
        codes.append("too_few_models")
    masks = trigger_masks(conf, ent, margin, disagreement, policy)
    codes += [c for c, hit in masks.items() if bool(hit)]

    ctx.update(
        confidence=conf, entropy=ent, margin=margin, disagreement=disagreement,
        min_confidence=policy.min_confidence, max_entropy=policy.max_entropy,
        min_margin=policy.min_margin, max_disagreement=policy.max_disagreement,
    )
    decision = AbstentionDecision(
        abstain=bool(codes),
        codes=codes,
        reasons=[_REASON_TEXT[c].format(**ctx) for c in codes],
        policy_status=policy.status,
    )

    model_names = list(usable)
    per_model = []
    for i, name in enumerate(model_names):
        p = usable[name]
        m_top = int(np.argmax(p))
        per_model.append({
            "model": name,
            "top1_name": names[m_top],
            "top1_probability": float(p[m_top]),
            "probability_of_ensemble_top1": float(p[top1]),
            "agrees_with_ensemble": m_top == top1,
            "weight": eff[name],
        })

    return EnsembleResult(
        class_names=names,
        probabilities=ens,
        top1_index=top1,
        top1_name=names[top1],
        confidence=conf,
        margin=margin,
        entropy=ent,
        top_k=top_k(ens, names, k),
        models_used=model_names,
        excluded=excluded,
        effective_weights=eff,
        per_model=per_model,
        disagreement={
            "fraction_disagreeing": disagreement,
            "mutual_information": float(dis["mutual_information"][0]),
            "all_agree": bool(dis["all_agree"][0]),
        },
        calibrated=calibrated,
        decision=decision,
    )


# ==========================================================================
# Fitting thresholds (validation data only)
# ==========================================================================
def fit_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    target_accuracy: float,
    higher_is_better: bool = True,
    min_coverage: float = 0.05,
) -> Optional[Dict[str, float]]:
    """Loosest threshold whose accepted samples reach `target_accuracy`.

    Accept rule: score >= t (higher_is_better) or score <= t (otherwise).
    Returns None if no threshold reaches the target with at least
    `min_coverage` of the samples accepted.
    """
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    n = len(s)
    if n == 0 or c.shape != s.shape:
        raise EnsembleError("fit_threshold needs equally sized, non-empty scores and correct arrays.")
    if not higher_is_better:
        s = -s
    order = np.argsort(-s, kind="stable")
    s_sorted, c_sorted = s[order], c[order]
    accepted = np.arange(1, n + 1)
    accuracy = np.cumsum(c_sorted) / accepted
    group_end = np.r_[s_sorted[1:] != s_sorted[:-1], True]          # whole tie groups only
    ok = group_end & (accuracy >= target_accuracy) & (accepted / n >= min_coverage)
    if not ok.any():
        return None
    idx = int(np.nonzero(ok)[0].max())
    threshold = float(s_sorted[idx] if higher_is_better else -s_sorted[idx])
    return {
        "threshold": threshold,
        "coverage": float(accepted[idx] / n),
        "selective_accuracy": float(accuracy[idx]),
    }


def fit_abstention_thresholds(
    metrics: Dict[str, np.ndarray],
    correct: np.ndarray,
    target_accuracy: float = 0.99,
    min_coverage: float = 0.05,
) -> Dict[str, Any]:
    """Fit each score threshold on VALIDATION data to a target selective accuracy.

    Each threshold is fitted separately to the same target. The combined
    policy (all rules together) must then be evaluated, and it is stricter
    than any single rule. Thresholds that cannot reach the target are left
    out and stay provisional.
    """
    rules = {
        "min_confidence": ("confidence", True),
        "max_entropy": ("entropy", False),
        "min_margin": ("margin", True),
        "max_disagreement": ("disagreement", False),
    }
    thresholds: Dict[str, float] = {}
    report: Dict[str, Any] = {}
    for field_name, (metric, higher) in rules.items():
        fit = fit_threshold(metrics[metric], correct, target_accuracy, higher, min_coverage)
        if fit is None:
            report[field_name] = {"fitted": False, "reason": "target accuracy not reachable"}
        else:
            thresholds[field_name] = fit["threshold"]
            report[field_name] = {"fitted": True, **fit}
    return {
        "thresholds": thresholds,
        "report": report,
        "target_selective_accuracy": target_accuracy,
        "min_coverage": min_coverage,
    }


# ==========================================================================
# Calibration file (written by scripts/evaluate.py --fit-calibration)
# ==========================================================================
def calibration_path(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["outputs"]) / "calibration" / "calibration.json"


def load_calibration(path: Path) -> Optional[Dict[str, Any]]:
    """Return the saved calibration dict, or None if there is none / it is unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_calibration(
    path: Path,
    temperatures: Dict[str, float],
    fit: Dict[str, Any],
    n_validation_samples: int,
    ensemble_temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Write calibration.json. Always records that it was fitted on the validation split."""
    data = {
        "fitted_on": "validation",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_validation_samples": int(n_validation_samples),
        "temperatures": {k: float(v) for k, v in temperatures.items()},
        "ensemble_temperature": None if ensemble_temperature is None else float(ensemble_temperature),
        "thresholds": fit["thresholds"],
        "fit_report": fit["report"],
        "target_selective_accuracy": fit["target_selective_accuracy"],
        "min_coverage": fit["min_coverage"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
