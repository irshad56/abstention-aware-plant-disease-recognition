"""Citation generation and presentation formatting."""

from typing import List
from plant_disease_agent.schemas import EvidenceRecord


def format_markdown_citations(records: List[EvidenceRecord]) -> str:
    """Formats evidence records into GitHub-flavored Markdown citation cards with clickable links."""
    if not records:
        return "_No verified citations available for this query._"

    lines = []
    for idx, rec in enumerate(records, 1):
        lines.append(f"**[{idx}] [{rec.title}]({rec.url})**  ")
        lines.append(f"**Publisher / Institution:** {rec.publisher} | **Source:** `{rec.source_type}`  ")
        lines.append(f"> {rec.excerpt}  ")
        if rec.validation_notes:
            lines.append(f"*Validation Status:* {rec.validation_notes}\n")
        else:
            lines.append("\n")

    return "\n".join(lines)
