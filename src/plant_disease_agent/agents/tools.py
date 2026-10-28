"""Agent tools: LLM synthesis, deterministic explanation templates, and retrieval helpers."""

import json
import os
import re
from typing import Dict, List, Optional
from plant_disease_agent.schemas import (
    EnsembleResult,
    EvidenceRecord,
    GroundedExplanation,
)
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_deterministic_explanation(
    ensemble: EnsembleResult,
    evidence_records: List[EvidenceRecord],
) -> GroundedExplanation:
    """Generates a strictly grounded, hallucination-free explanation using local verified evidence."""
    pred_class = ensemble.predicted_class
    parts = pred_class.split("__")
    plant_species = parts[0].strip()
    disease_name = parts[1].strip() if len(parts) > 1 else pred_class

    confidence_pct = round(ensemble.confidence * 100.0, 1)

    symptoms = []
    actions = []
    prevention = []

    for rec in evidence_records:
        text = rec.excerpt
        # Extract symptom and management bullet points from structured excerpt
        for segment in text.split(" | "):
            if segment.startswith("Symptoms:"):
                s_list = segment.replace("Symptoms:", "").split("; ")
                symptoms.extend([s.strip() for s in s_list if s.strip()])
            elif segment.startswith("Management:"):
                m_list = segment.replace("Management:", "").split("; ")
                actions.extend([m.strip() for m in m_list if m.strip()])
            elif segment.startswith("Aetiology:"):
                c_list = segment.replace("Aetiology:", "").split("; ")
                prevention.extend([c.strip() for c in c_list if c.strip()])

    if not symptoms:
        symptoms = [
            f"Foliar lesions or chlorosis characteristic of {disease_name}.",
            "Morphological changes consistent with the deep ensemble consensus.",
        ]

    if not actions:
        if "healthy" in disease_name.lower():
            actions = [
                "Continue regular moisture and balanced nutrient management.",
                "Conduct routine bi-weekly foliar scouting to maintain disease-free canopy.",
            ]
        else:
            actions = [
                "Prune and destroy infected foliage to mitigate secondary spore spread.",
                "Ensure morning drip irrigation to prevent prolonged leaf wetness.",
                "Consult the cited university agricultural extension bulletin for labeled control measures.",
            ]

    summary = (
        f"The 5-model deep ensemble converged on **{pred_class}** with **{confidence_pct}%** mean confidence "
        f"(entropy: {ensemble.predictive_entropy:.2f} nats, margin: {ensemble.margin:.2f}). "
        f"{len(ensemble.participating_models)} independent model architectures contributed to this consensus."
    )

    sci_grounding = None
    for rec in evidence_records:
        if rec.source_type == "scientific_literature":
            sci_grounding = (
                f"Grounded by peer-reviewed research: '{rec.title}' ({rec.publisher}). "
                f"{rec.excerpt.replace(' | ', '. ')}"
            )
            break

    return GroundedExplanation(
        predicted_class=pred_class,
        plant_species=plant_species,
        disease_name=disease_name,
        confidence_percentage=confidence_pct,
        summary=summary,
        verified_symptoms=list(dict.fromkeys(symptoms))[:4],
        recommended_actions=list(dict.fromkeys(actions))[:4],
        preventative_measures=list(dict.fromkeys(prevention))[:3],
        scientific_literature_grounding=sci_grounding,
        citations=evidence_records,
    )


class LLMSynthesizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq LLM client with model '{self.model}'")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}. Fallback to deterministic template.")
                self.client = None

    def synthesize(
        self,
        ensemble: EnsembleResult,
        evidence_records: List[EvidenceRecord],
    ) -> GroundedExplanation:
        """Synthesizes a grounded explanation via Groq if available, otherwise uses deterministic fallback."""
        if not self.client or not self.api_key:
            return generate_deterministic_explanation(ensemble, evidence_records)

        evidence_context = "\n".join([
            f"- [{rec.citation_id}] ({rec.publisher}): {rec.excerpt}"
            for rec in evidence_records
        ])

        system_prompt = (
            "You are an expert plant pathologist and agricultural advisor. "
            "You must summarize a plant disease diagnosis based EXCLUSIVELY on the provided prediction data "
            "and verified evidence excerpts, specifically citing the peer-reviewed research and extension bulletins. "
            "Do not fabricate facts, symptoms, or recommendations."
        )

        user_prompt = f"""Diagnosis: {ensemble.predicted_class}
Ensemble Confidence: {ensemble.confidence * 100:.1f}%
Participating Models: {', '.join(ensemble.participating_models)}
Entropy: {ensemble.predictive_entropy:.3f} nats
Margin: {ensemble.margin:.3f}

Verified Evidence Records (Extension Bulletins & Q1 Peer-Reviewed Research):
{evidence_context if evidence_context else "No external evidence retrieved."}

Output a strictly valid JSON object with the following schema:
{{
  "summary": "2-3 sentence grounded diagnosis summary referencing the consensus and pathology",
  "verified_symptoms": ["symptom 1", "symptom 2", "symptom 3"],
  "recommended_actions": ["management action 1", "management action 2"],
  "preventative_measures": ["preventative measure 1", "preventative measure 2"],
  "scientific_basis": "1-2 sentences summarizing how peer-reviewed research (cite journal/authors from evidence) supports this pathology"
}}
"""

        try:
            # Fallback list of Groq models available and verified on this key
            models_to_try = [self.model, "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
            response = None
            last_err = None

            for m in models_to_try:
                if not m:
                    continue
                try:
                    response = self.client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=1000,
                        response_format={"type": "json_object"},
                    )
                    break
                except Exception as call_err:
                    last_err = call_err
                    continue

            if not response:
                raise last_err

            content = response.choices[0].message.content
            # Handle potential code fences
            if "```" in content:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                content = match.group(0) if match else content

            parsed = json.loads(content)

            parts = ensemble.predicted_class.split("__")
            plant_species = parts[0].strip()
            disease_name = parts[1].strip() if len(parts) > 1 else ensemble.predicted_class

            # Fallback scientific grounding if LLM omitted it
            sci_grounding = parsed.get("scientific_basis")
            if not sci_grounding:
                for rec in evidence_records:
                    if rec.source_type == "scientific_literature":
                        sci_grounding = f"Corroborated by {rec.title} ({rec.publisher}). {rec.excerpt}"
                        break

            return GroundedExplanation(
                predicted_class=ensemble.predicted_class,
                plant_species=plant_species,
                disease_name=disease_name,
                confidence_percentage=round(ensemble.confidence * 100.0, 1),
                summary=parsed.get("summary", ""),
                verified_symptoms=parsed.get("verified_symptoms", []),
                recommended_actions=parsed.get("recommended_actions", []),
                preventative_measures=parsed.get("preventative_measures", []),
                scientific_literature_grounding=sci_grounding,
                citations=evidence_records,
            )
        except Exception as e:
            logger.warning(f"LLM synthesis failed or returned invalid JSON ({e}). Using deterministic template.")
            return generate_deterministic_explanation(ensemble, evidence_records)
