"""Single entry point used by the Streamlit app and the scripts.

    pipeline = Pipeline.from_config()          # loads the five models once
    vision   = pipeline.predict(uploaded_file)  # classification only
    result   = pipeline.analyze(uploaded_file)  # classification + evidence + explanation

predict() never raises for a bad image or a failing model: it returns a VisionResult
that says what happened (and abstains). Evidence and language-model steps can fail
without affecting the classification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from plant_disease import ensemble as ens
from plant_disease import retrieval as R
from plant_disease.agent import AgentResult, EvidenceWorkflow, Explainer
from plant_disease.config import (
    ClassMapping,
    get_logger,
    load_class_mapping,
    load_config,
    setup_logging,
)
from plant_disease.imaging import validate_image

log = get_logger("pipeline")


@dataclass
class VisionResult:
    """Everything the classifier side produced for one image."""

    image_ok: bool
    image_check: Dict[str, Any]
    ensemble: Optional[ens.EnsembleResult]
    model_seconds: Dict[str, float]
    total_seconds: float
    calibrated: bool
    policy_status: str
    unverified_preprocessing: List[str] = field(default_factory=list)
    calibration_note: str = ""

    @property
    def accepted(self) -> bool:
        return self.ensemble is not None and not self.ensemble.decision.abstain

    @property
    def abstained(self) -> bool:
        return not self.accepted

    @property
    def predicted_class(self) -> Optional[str]:
        """The disease name, only when the system accepted its own prediction."""
        return self.ensemble.top1_name if self.accepted else None

    @property
    def best_guess(self) -> Optional[str]:
        """Top class even when abstaining. For technical details only, never as a diagnosis."""
        return self.ensemble.top1_name if self.ensemble is not None else None

    @property
    def models_used(self) -> List[str]:
        return list(self.ensemble.models_used) if self.ensemble is not None else []

    @property
    def excluded(self) -> Dict[str, str]:
        return dict(self.ensemble.excluded) if self.ensemble is not None else {}

    @property
    def image_warnings(self) -> List[str]:
        return list(self.image_check.get("warnings", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "predicted_class": self.predicted_class,
            "image": self.image_check,
            "ensemble": self.ensemble.to_dict() if self.ensemble is not None else None,
            "model_seconds": {k: round(v, 4) for k, v in self.model_seconds.items()},
            "total_seconds": round(self.total_seconds, 4),
            "calibrated": self.calibrated,
            "policy_status": self.policy_status,
            "unverified_preprocessing": self.unverified_preprocessing,
            "calibration_note": self.calibration_note,
        }


class Pipeline:
    """Holds the loaded models, policy, calibration, evidence index and workflow."""

    def __init__(
        self,
        config: Dict[str, Any],
        registry: Any,
        class_mapping: ClassMapping,
        evidence_index: Optional[R.EvidenceIndex] = None,
        explainer: Optional[Explainer] = None,
        use_langgraph: bool = True,
    ) -> None:
        self.config = config
        self.registry = registry
        self.class_mapping = class_mapping
        self.calibration = ens.load_calibration(ens.calibration_path(config))
        self.ensemble_temperature = ens.ensemble_temperature_from(self.calibration)
        self.policy = ens.AbstentionPolicy.from_config(config, self.calibration)
        self.weights = config["ensemble"].get("weights")
        self.index = evidence_index if evidence_index is not None else R.EvidenceIndex([])
        self.explainer = explainer or Explainer(config)
        self.workflow = EvidenceWorkflow(self.predict, self.index, self.explainer, config, use_langgraph)

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        only_models: Optional[List[str]] = None,
        use_langgraph: bool = True,
    ) -> "Pipeline":
        """Load config, class mapping, all models and the evidence index."""
        config = config or load_config()
        setup_logging(config["app"].get("log_level", "INFO"),
                      Path(config["paths"]["outputs"]) / "logs" / "app.log")
        mapping = load_class_mapping(config)
        from plant_disease.models import load_models          # lazy: needs PyTorch

        registry = load_models(config, mapping, device=device, only=only_models)
        try:
            index = R.EvidenceIndex.from_config(config)
        except R.RetrievalError as exc:
            log.error("Evidence index unavailable: %s", exc)
            index = R.EvidenceIndex([], notice=str(exc))
        return cls(config, registry, mapping, index, Explainer(config), use_langgraph)

    # ---- classification -----------------------------------------------------
    def predict(self, source: Any) -> VisionResult:
        """Validate the image, run every loaded model, ensemble and decide accept/abstain."""
        start = time.perf_counter()
        check = validate_image(source, self.config)
        failed: Dict[str, str] = dict(getattr(self.registry, "failures", {}) or {})
        per_model: Dict[str, np.ndarray] = {}
        seconds: Dict[str, float] = {}

        if check.ok:
            for name, adapter in self.registry.adapters.items():
                t0 = time.perf_counter()
                try:
                    per_model[name] = adapter.predict_proba([check.image])[0]
                    seconds[name] = time.perf_counter() - t0
                except Exception as exc:  # noqa: BLE001 - report, never substitute
                    failed[name] = f"inference failed: {exc}"
                    log.error("Model %s failed on this image: %s", name, exc)

        result = ens.run_ensemble(
            per_model, self.class_mapping.names, self.policy, weights=self.weights,
            models_failed=failed, image_ok=check.ok, quality_poor=check.quality_poor,
            ensemble_temperature=self.ensemble_temperature,
        )

        unverified = [self.registry.adapters[n].display_name for n in result.models_used
                      if not getattr(self.registry.adapters[n], "preprocessing_verified", True)]
        note = ""
        fitted_models = (self.calibration or {}).get("models")
        if self.ensemble_temperature is not None and fitted_models \
                and set(fitted_models) != set(result.models_used):
            note = ("Calibration was fitted with a different set of models than the ones that "
                    "contributed to this result.")
        return VisionResult(
            image_ok=check.ok, image_check=check.to_dict(), ensemble=result,
            model_seconds=seconds, total_seconds=time.perf_counter() - start,
            calibrated=result.calibrated, policy_status=self.policy.status,
            unverified_preprocessing=unverified, calibration_note=note,
        )

    # ---- full analysis ------------------------------------------------------
    def analyze(self, source: Any) -> AgentResult:
        """Classification, then (if accepted) evidence retrieval and a cited explanation."""
        return self.workflow.run(source)

    # ---- diagnostics ----------------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        rows = self.registry.diagnostics() if hasattr(self.registry, "diagnostics") else []
        cal = self.calibration or {}
        return {
            "device": getattr(self.registry, "device", "unknown"),
            "models": rows,
            "models_loaded": len(self.registry.adapters),
            "models_failed": dict(getattr(self.registry, "failures", {}) or {}),
            "abstention": {
                "status": self.policy.status,
                "min_confidence": self.policy.min_confidence,
                "max_entropy": self.policy.max_entropy,
                "min_margin": self.policy.min_margin,
                "max_disagreement": self.policy.max_disagreement,
                "min_models": self.policy.min_models,
            },
            "calibration": {
                "present": bool(cal), "fitted_on": cal.get("fitted_on"),
                "created_utc": cal.get("created_utc"),
                "ensemble_temperature": self.ensemble_temperature,
                "n_validation_samples": cal.get("n_validation_samples"),
            },
            "evidence": self.index.stats(),
            "language_model": self.explainer.status(),
            "workflow_engine": self.workflow.engine,
        }
