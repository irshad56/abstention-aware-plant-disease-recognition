"""Evaluate the five models and the ensemble; fit calibration on validation data.

    python scripts\\evaluate.py --check-manifest              # resolve files, no models
    python scripts\\evaluate.py --fit-calibration             # validation split ONLY
    python scripts\\evaluate.py --split test                  # held-out evaluation
    python scripts\\evaluate.py --split test --limit 200      # quick partial run

Typical order: --check-manifest, then --fit-calibration, then --split test.
The test split is never used to fit anything.

Dataset and manifest locations come from configs\\config.yaml or the
PD_DATASET_ROOT / PD_MANIFEST_PATH environment variables, or --dataset-root /
--manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease.config import (  # noqa: E402
    ConfigError,
    load_class_mapping,
    load_config,
    setup_logging,
)
from plant_disease.evaluation import (  # noqa: E402
    EvaluationError,
    check_manifest,
    run_evaluation,
    run_fit_calibration,
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        if value != value:                      # NaN
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_manifest_report(report: Dict[str, Any]) -> None:
    print("Manifest check")
    print(f"  manifest : {report['manifest_path']}")
    print(f"  dataset  : {report['dataset_root']}")
    print(f"  columns  : {report['columns_used']}")
    print(f"\n  {'split':<7}{'in manifest':>12}{'historical':>12}{'ok':>8}{'missing':>9}{'ambiguous':>11}{'bad label':>11}")
    for split, s in report["splits"].items():
        print(f"  {split:<7}{s['manifest_rows']:>12}{str(s['expected_historical']):>12}"
              f"{s['ok']:>8}{s['missing']:>9}{s['ambiguous']:>11}{s['bad_label']:>11}")
    print(f"\n  path resolution methods: {report['resolution_methods']}")
    if report["missing_examples"]:
        print("  examples of unresolved paths:")
        for p in report["missing_examples"][:5]:
            print(f"    {p}")
    print(f"  ({report['note']})")


def print_evaluation(summary: Dict[str, Any]) -> None:
    c = summary["sample_counts"]
    print(f"\nEvaluation on split '{summary['split']}'")
    print(f"  rows in manifest split : {c['manifest_rows_in_split']}")
    print(f"  processed / resolved   : {c['processed_rows']} / {c['resolved_ok']}")
    print(f"  evaluated              : {c['evaluated']}")
    print(f"  missing or unusable    : {c['missing_or_unusable']}   unreadable images: {c['unreadable_images']}")
    if summary["models_failed"]:
        print(f"  models that FAILED     : {list(summary['models_failed'])}")
    print(f"  decision ensemble      : {summary['decision_ensemble']} "
          f"(calibration applied: {summary['calibration_applied']})")
    print(f"  abstention thresholds  : {summary['abstention_policy']['status']}\n")

    df = summary["table"]
    print(f"  {'name':<24}{'acc':>8}{'macroF1':>9}{'balAcc':>8}{'ECE':>8}{'NLL':>8}{'ms/img':>9}")
    for _, r in df.iterrows():
        print(f"  {r['name']:<24}{_fmt(r['accuracy']):>8}{_fmt(r['macro_f1']):>9}"
              f"{_fmt(r['balanced_accuracy']):>8}{_fmt(r['ece_15bin']):>8}{_fmt(r['nll']):>8}"
              f"{_fmt(r['latency_ms_per_image'], 1):>9}")

    s = summary["selective"]
    print("\n  Selective prediction (decision ensemble, abstention policy applied):")
    print(f"    coverage {_fmt(s['coverage'])} | abstention rate {_fmt(s['abstention_rate'])} | "
          f"selective accuracy {_fmt(s['selective_accuracy'])} | selective risk {_fmt(s['selective_risk'])}")
    print(f"    abstention triggers: {s['trigger_counts']}")
    for w in summary["warnings"]:
        print(f"  WARNING: {w}")
    print(f"\n  Outputs written to: {summary['output_dir']}")


def print_calibration(report: Dict[str, Any]) -> None:
    print(f"\nCalibration fitted on the VALIDATION split ({report['n_validation_samples']} samples)")
    print(f"  models used: {report['models']}")
    if report["models_failed"]:
        print(f"  models that FAILED: {list(report['models_failed'])}")
    print(f"\n  {'model':<22}{'T':>7}{'NLL before':>12}{'NLL after':>11}{'ECE before':>12}{'ECE after':>11}")
    for m, r in report["per_model"].items():
        print(f"  {m:<22}{_fmt(r['temperature'], 3):>7}{_fmt(r['nll_before']):>12}{_fmt(r['nll_after']):>11}"
              f"{_fmt(r['ece_before']):>12}{_fmt(r['ece_after']):>11}")
    e = report["ensemble"]
    print(f"\n  ensemble (used for decisions): T={_fmt(e['temperature'], 3)}  "
          f"NLL {_fmt(e['nll_before'])} -> {_fmt(e['nll_after'])}   "
          f"ECE {_fmt(e['ece_before'])} -> {_fmt(e['ece_after'])}")
    print(f"  ensemble accuracy on validation: {_fmt(report['ensemble_accuracy_validation'])}")
    print(f"  thresholds fitted: {report['thresholds_fitted']}")
    for name, r in report["threshold_report"].items():
        if not r["fitted"]:
            print(f"    {name}: not fitted ({r['reason']}); stays provisional")
    print(f"  policy status: {report['policy_status']}")
    s = report["combined_policy_on_validation"]
    print(f"  combined policy on validation: coverage {_fmt(s['coverage'])}, "
          f"selective accuracy {_fmt(s['selective_accuracy'])}")
    for w in report["warnings"]:
        print(f"  NOTE: {w}")
    print(f"\n  Saved: {report['calibration_file']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate models and fit validation-only calibration.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fit-calibration", action="store_true",
                      help="Fit temperatures and abstention thresholds on the validation split")
    mode.add_argument("--check-manifest", action="store_true",
                      help="Only check that manifest files can be found (no models)")
    p.add_argument("--split", choices=["train", "val", "test"], default=None,
                   help="Split to evaluate (default: test). Not allowed with --fit-calibration.")
    p.add_argument("--config", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--limit", type=int, default=None, help="Use only N random rows (quick check)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--no-cache", action="store_true", help="Ignore cached logits and re-run the models")
    p.add_argument("--target-accuracy", type=float, default=0.99,
                   help="Selective-accuracy target for threshold fitting (default 0.99)")
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--split-column", default=None)
    p.add_argument("--label-column", default=None)
    p.add_argument("--path-column", nargs="*", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fit_calibration and args.split not in (None, "val"):
        print("ERROR: calibration is fitted on the validation split only; --split is not allowed here.",
              file=sys.stderr)
        return 1
    try:
        config = load_config(args.config)
        setup_logging(config["app"].get("log_level", "INFO"),
                      Path(config["paths"]["outputs"]) / "logs" / "evaluate.log")
        mapping = load_class_mapping(config)
        manifest = Path(args.manifest) if args.manifest else config["paths"]["manifest"]
        dataset_root = Path(args.dataset_root) if args.dataset_root else config["paths"]["dataset_root"]
        if dataset_root is None:
            print("NOTE: no dataset root set (paths.dataset_root / PD_DATASET_ROOT / --dataset-root). "
                  "Only absolute paths inside the manifest can be resolved.\n")
        co = {"split_col": args.split_column, "label_col": args.label_column, "path_cols": args.path_column}

        if args.check_manifest:
            print_manifest_report(check_manifest(config, mapping, manifest, dataset_root, args.limit, co))
            return 0
        common = dict(batch_size=args.batch_size, limit=args.limit, device=args.device,
                      use_cache=not args.no_cache, only_models=args.models, column_overrides=co)
        if args.fit_calibration:
            print_calibration(run_fit_calibration(
                config, mapping, manifest, dataset_root,
                target_accuracy=args.target_accuracy, min_coverage=args.min_coverage, **common))
            return 0
        print_evaluation(run_evaluation(config, mapping, args.split or "test", manifest, dataset_root, **common))
        return 0
    except (EvaluationError, ConfigError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
