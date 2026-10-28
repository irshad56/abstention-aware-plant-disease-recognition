"""Tests for agricultural evidence validation, domain whitelisting, and citation filtering."""

import pytest
from plant_disease_agent.evidence.sources import is_trusted_url
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.schemas import EvidenceRecord


def test_domain_whitelist():
    assert is_trusted_url("https://extension.psu.edu/apple-scab") is True
    assert is_trusted_url("https://ipm.ucanr.edu/agriculture/grape/") is True
    assert is_trusted_url("https://edis.ifas.ufl.edu/vh052") is True
    assert is_trusted_url("https://ohioline.osu.edu/factsheet/plpath-fru-24") is True
    assert is_trusted_url("https://malicious-fake-site.com/cure") is False
    assert is_trusted_url("not_a_url") is False


def test_evidence_validator_accepts_valid_record():
    validator = EvidenceValidator()
    rec = EvidenceRecord(
        citation_id="CIT-1",
        title="Early Blight Management in Tomatoes",
        publisher="Penn State Extension",
        url="https://extension.psu.edu/tomato-early-blight",
        source_type="extension_service",
        excerpt="Alternaria solani causes concentric ring lesions on tomato foliage.",
    )
    is_valid, note = validator.validate_record(rec, target_plant="Tomato", target_disease="Early Blight")
    assert is_valid is True


def test_evidence_validator_rejects_untrusted_domain():
    validator = EvidenceValidator()
    rec = EvidenceRecord(
        citation_id="CIT-2",
        title="Unverified Blog Cure for Tomato Blight",
        publisher="GardeningBlog123",
        url="https://randomblog.com/tomato-blight",
        source_type="blog",
        excerpt="Spray vinegar on tomato foliage to cure blight.",
    )
    is_valid, note = validator.validate_record(rec, target_plant="Tomato", target_disease="Early Blight")
    assert is_valid is False
    assert "not in trusted" in note.lower()
