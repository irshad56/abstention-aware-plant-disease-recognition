"""Build the evidence corpus from the sources listed in data/evidence/sources.yaml.

    python scripts\\build_evidence.py --check         # validate the list only (no internet)
    python scripts\\build_evidence.py                 # fetch pages and (re)build the corpus
    python scripts\\build_evidence.py --only unl-apple-scab ucipm-tomato-whiteflies
    python scripts\\build_evidence.py --rebuild       # drop previously stored copies first

This is the only script besides download_backbones.py that uses the internet. It
fetches only pages whose domain is in retrieval.allowed_domains, obeys robots.txt,
and stores the real extracted text with its URL, publisher and retrieval date.
A source that fails is reported and never turns into a document. If an older copy
of that source is already stored it is kept (and reported) unless --rebuild is used.
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

from plant_disease import retrieval as R  # noqa: E402
from plant_disease.config import (  # noqa: E402
    ConfigError,
    load_class_mapping,
    load_config,
    setup_logging,
)


def print_sources(entries: List[R.SourceEntry], problems: List[str]) -> None:
    print(f"{len(entries)} valid source(s):")
    for e in entries:
        kind = "local file" if e.local_file else "web"
        print(f"  - {e.id:<32}{e.source_type:<24}{kind:<11}{', '.join(e.classes) or '(no class tag)'}")
    if problems:
        print(f"\n{len(problems)} problem(s) in sources.yaml (these entries are ignored):")
        for p in problems:
            print(f"  ! {p}")


def coverage_report(config: Dict[str, Any], docs: List[R.EvidenceDoc]) -> Optional[Dict[str, Any]]:
    """Which disease classes have at least one stored source (needs class_mapping.json)."""
    try:
        mapping = load_class_mapping(config)
    except ConfigError:
        return None
    index = R.EvidenceIndex(R.build_passages(docs, int(config["retrieval"].get("chunk_chars", 900)),
                                             int(config["retrieval"].get("chunk_overlap", 150))))
    diseases = [n for n in mapping.names if not R.is_healthy_class(n)]
    covered = [n for n in diseases if index.has_evidence_for(n)]
    missing = [n for n in diseases if n not in covered]
    return {"disease_classes": len(diseases), "covered": covered, "without_evidence": missing,
            "passages": len(index.passages)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evidence corpus.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", default=None, help="Path to sources.yaml (default from config)")
    parser.add_argument("--only", nargs="*", default=None, help="Only these source ids")
    parser.add_argument("--check", action="store_true", help="Validate sources.yaml; no downloads")
    parser.add_argument("--rebuild", action="store_true", help="Discard previously stored documents")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between web requests")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Do not consult robots.txt (not recommended)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        setup_logging(config["app"].get("log_level", "INFO"),
                      Path(config["paths"]["outputs"]) / "logs" / "build_evidence.log")
        sources_path = Path(args.sources) if args.sources else config["paths"]["evidence_sources"]
        corpus_path: Path = config["paths"]["evidence_corpus"]
        entries, problems = R.load_sources(sources_path, config["retrieval"]["allowed_domains"])
    except (ConfigError, R.RetrievalError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Sources file: {sources_path}\n")
    print_sources(entries, problems)
    if args.only:
        unknown = [i for i in args.only if i not in {e.id for e in entries}]
        if unknown:
            print(f"\nERROR: unknown source id(s): {unknown}", file=sys.stderr)
            return 1
        entries = [e for e in entries if e.id in set(args.only)]
    if args.check:
        print("\nCheck only: nothing was downloaded.")
        return 1 if problems and not entries else 0
    if not entries:
        print("\nERROR: no valid sources to fetch. Edit data\\evidence\\sources.yaml.", file=sys.stderr)
        return 1

    print("\nFetching (internet required) ...\n")
    docs_new, report = R.ingest_sources(config, entries, delay_s=args.delay,
                                        respect_robots=not args.ignore_robots)

    # Merge with what is already stored
    listed_ids = {e.id for e in R.load_sources(sources_path, config["retrieval"]["allowed_domains"])[0]}
    previous = {} if args.rebuild else {d.source_id: d for d in R.load_corpus(corpus_path)}
    merged: Dict[str, R.EvidenceDoc] = {sid: d for sid, d in previous.items() if sid in listed_ids}
    for row in report:
        if row["status"] != "ok" and row["id"] in merged:
            row["status"] = "kept_previous"
    for d in docs_new:
        merged[d.source_id] = d
    final_docs = list(merged.values())

    width = max(len(r["id"]) for r in report)
    for r in report:
        extra = f"{r['chars']} chars, title: {r['title'][:50]}" if r["status"] == "ok" else r["reason"]
        if r["status"] == "kept_previous":
            extra = f"fetch failed ({r['reason']}); previous stored copy kept"
        print(f"[{r['status'].upper():<13}] {r['id']:<{width}}  {extra}")

    ok = sum(1 for r in report if r["status"] == "ok")
    failed = [r for r in report if r["status"] == "failed"]
    if not final_docs:
        print("\nERROR: nothing could be fetched, so no corpus was written.", file=sys.stderr)
        return 1

    R.save_corpus(corpus_path, final_docs)
    cov = coverage_report(config, final_docs)
    out_dir = Path(config["paths"]["outputs"]) / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build_report.json").write_text(
        json.dumps({"sources": report, "problems": problems, "coverage": cov,
                    "documents_stored": len(final_docs)}, indent=2), encoding="utf-8")

    print(f"\nCorpus written: {corpus_path}")
    print(f"  fetched ok: {ok} | failed: {len(failed)} | documents stored: {len(final_docs)}")
    if cov:
        print(f"  disease classes with evidence: {len(cov['covered'])} of {cov['disease_classes']} "
              f"({cov['passages']} searchable passages)")
        if cov["without_evidence"]:
            print("  classes WITHOUT evidence (the app will say so for these):")
            for name in cov["without_evidence"]:
                print(f"    - {name}")
    if failed:
        print("\nSome sources failed. Replace or fix those entries in sources.yaml, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
