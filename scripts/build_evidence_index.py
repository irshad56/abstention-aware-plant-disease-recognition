"""Evidence Index Builder and Corpus Coverage Validator."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plant_disease_agent.config import AppConfig
from plant_disease_agent.evidence.sources import is_trusted_url


def validate_corpus():
    config = AppConfig()
    corpus_p = Path("data/evidence/corpus.json")

    if not corpus_p.exists():
        print(f"Error: Evidence corpus not found at {corpus_p}")
        return

    with open(corpus_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_classes = {item.get("class_name", "").strip(): item for item in data}
    expected_classes = config.class_names

    print("\n========================================================")
    print(" EVIDENCE CORPUS COVERAGE AND AUTHENTICITY VALIDATION")
    print("========================================================\n")

    covered = 0
    trusted_count = 0

    for idx, c_name in enumerate(expected_classes):
        profile = corpus_classes.get(c_name.strip())
        if profile:
            covered += 1
            url = profile.get("url", "")
            trusted = is_trusted_url(url)
            if trusted:
                trusted_count += 1
            print(f"[{'OK' if trusted else '!'}] {c_name}: {profile.get('title')} ({profile.get('publisher')})")
        else:
            print(f"[MISSING] {c_name}")

    print(f"\nCorpus Summary:")
    print(f"- Total Classes in Registry: {len(expected_classes)}")
    print(f"- Covered by Corpus: {covered}/{len(expected_classes)} ({covered/len(expected_classes):.1%})")
    print(f"- Trusted Whitelisted Domains: {trusted_count}/{covered}\n")


if __name__ == "__main__":
    validate_corpus()
