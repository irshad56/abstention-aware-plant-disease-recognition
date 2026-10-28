"""Agricultural evidence subsystem."""

from plant_disease_agent.evidence.sources import is_trusted_url, TRUSTED_DOMAINS
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.evidence.citations import format_markdown_citations

__all__ = [
    "is_trusted_url",
    "TRUSTED_DOMAINS",
    "EvidenceRetriever",
    "EvidenceValidator",
    "format_markdown_citations",
]
