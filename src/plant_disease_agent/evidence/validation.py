"""Evidence validation and citation authenticity verification."""

import re
from typing import List, Tuple
from urllib.parse import urlparse

from plant_disease_agent.schemas import EvidenceRecord
from plant_disease_agent.evidence.sources import is_trusted_url
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class EvidenceValidator:
    def __init__(self, allowed_domains: List[str] = None):
        self.allowed_domains = allowed_domains

    def validate_record(
        self,
        record: EvidenceRecord,
        target_plant: str,
        target_disease: str,
    ) -> Tuple[bool, str]:
        """Validates a retrieved evidence passage for domain trust, provenance, and relevance."""
        # 1. URL Structure Check
        try:
            parsed = urlparse(record.url)
            if not parsed.scheme or not parsed.netloc:
                return False, f"Malformed URL: {record.url}"
        except Exception:
            return False, f"Invalid URL syntax: {record.url}"

        # 2. Authoritative Domain Verification
        if not is_trusted_url(record.url, self.allowed_domains):
            return False, f"Domain '{parsed.netloc}' is not in trusted agricultural extension/research whitelist."

        # 3. Provenance Integrity Check
        if not record.title or len(record.title.strip()) < 5:
            return False, "Title is missing or suspiciously brief."
        if not record.publisher or len(record.publisher.strip()) < 3:
            return False, "Publisher attribution is missing."

        # 4. Contextual Relevance Check
        # Check that excerpt or title references the plant or the disease
        combined_text = f"{record.title} {record.excerpt}".lower()
        plant_clean = target_plant.lower().replace("(maize)", "").strip()
        disease_words = [w for w in re.split(r"[\s_]+", target_disease.lower()) if len(w) > 3]

        # Biological synonyms
        synonyms = {
            "corn": ["maize", "zea mays"],
            "bell pepper": ["pepper", "capsicum"],
            "tomato": ["solanum", "lycopersicum"],
            "potato": ["solanum", "tuberosum"],
            "apple": ["malus"],
            "cherry": ["prunus"],
            "peach": ["prunus", "persica"],
            "grape": ["vitis"],
            "strawberry": ["fragaria"],
        }
        relevant_plants = [plant_clean, "plant", "crop", "foliage"]
        if plant_clean in synonyms:
            relevant_plants.extend(synonyms[plant_clean])

        plant_matched = any(p in combined_text for p in relevant_plants)
        disease_matched = any(w in combined_text for w in disease_words) or "healthy" in target_disease.lower()

        if not (plant_matched or disease_matched):
            return False, f"Excerpt does not align with targeted host ({target_plant}) or disease ({target_disease})."

        return True, "Validated: Authoritative domain and content matches predicted pathology."

    def filter_records(
        self,
        records: List[EvidenceRecord],
        target_plant: str,
        target_disease: str,
    ) -> List[EvidenceRecord]:
        """Filters and annotates records with validation status."""
        validated = []
        for rec in records:
            is_valid, reason = self.validate_record(rec, target_plant, target_disease)
            rec.is_validated = is_valid
            rec.validation_notes = reason
            if is_valid:
                validated.append(rec)
            else:
                logger.warning(f"Rejected citation '{rec.title}' ({rec.url}): {reason}")
        return validated
