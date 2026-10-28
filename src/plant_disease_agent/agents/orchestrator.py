"""Agent Orchestrator: High-level execution interface for the LangGraph workflow."""

import time
from typing import Optional, Union
from PIL import Image

from plant_disease_agent.agents.graph import build_plant_disease_graph
from plant_disease_agent.agents.state import AgentState
from plant_disease_agent.agents.tools import LLMSynthesizer
from plant_disease_agent.abstention import AbstentionPolicy
from plant_disease_agent.config import AppConfig
from plant_disease_agent.ensemble import DeepEnsembleAggregator
from plant_disease_agent.evidence.retrieval import EvidenceRetriever
from plant_disease_agent.evidence.validation import EvidenceValidator
from plant_disease_agent.image_validation import ImageValidator
from plant_disease_agent.inference import InferenceManager
from plant_disease_agent.model_registry import ModelRegistry
from plant_disease_agent.schemas import AbstentionState, FinalResponse
from plant_disease_agent.logging_config import setup_logger

logger = setup_logger(__name__)


class AgentOrchestrator:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.model_registry = ModelRegistry(self.config)
        self.load_status = self.model_registry.discover_and_load_all()

        settings = self.config.settings
        img_cfg = settings.get("app", {}).get("image_validation", {})
        self.validator = ImageValidator(
            min_width=img_cfg.get("min_width", 64),
            min_height=img_cfg.get("min_height", 64),
            max_aspect_ratio=img_cfg.get("max_aspect_ratio", 4.0),
            min_sharpness_variance=img_cfg.get("min_sharpness_variance", 15.0),
            min_mean_intensity=img_cfg.get("min_mean_intensity", 0.05),
            max_mean_intensity=img_cfg.get("max_mean_intensity", 0.95),
            max_file_size_mb=settings.get("app", {}).get("max_upload_size_mb", 15.0),
        )

        self.inference_manager = InferenceManager(self.model_registry.adapters)

        ens_cfg = settings.get("ensemble", {})
        self.aggregator = DeepEnsembleAggregator(
            class_names=self.config.class_names,
            weights=ens_cfg.get("weights"),
            min_participating_models=settings.get("abstention", {}).get("min_participating_models", 3),
        )

        abs_cfg = settings.get("abstention", {})
        self.policy = AbstentionPolicy(
            min_confidence=abs_cfg.get("min_confidence", 0.70),
            max_predictive_entropy=abs_cfg.get("max_predictive_entropy", 1.20),
            min_margin=abs_cfg.get("min_margin", 0.15),
            max_model_disagreement=abs_cfg.get("max_model_disagreement", 0.40),
            min_participating_models=abs_cfg.get("min_participating_models", 3),
            abstain_on_quality_failure=abs_cfg.get("abstain_on_quality_failure", True),
        )

        ret_cfg = settings.get("retrieval", {})
        self.retriever = EvidenceRetriever(
            corpus_path=settings.get("paths", {}).get("evidence_corpus", "data/evidence/corpus.json"),
            enable_live_search=ret_cfg.get("enable_live_search", False),
            timeout_seconds=ret_cfg.get("request_timeout_seconds", 15),
            allowed_domains=ret_cfg.get("allowed_domains"),
        )

        self.evidence_validator = EvidenceValidator(
            allowed_domains=ret_cfg.get("allowed_domains")
        )

        llm_cfg = settings.get("llm", {})
        self.synthesizer = LLMSynthesizer(
            api_key=self.config.groq_api_key,
            model=self.config.groq_model,
            temperature=llm_cfg.get("temperature", 0.1),
        )

        self.graph = build_plant_disease_graph(
            validator=self.validator,
            inference_manager=self.inference_manager,
            aggregator=self.aggregator,
            policy=self.policy,
            retriever=self.retriever,
            evidence_validator=self.evidence_validator,
            synthesizer=self.synthesizer,
        )
        logger.info("LangGraph Agent Orchestrator successfully initialized.")

    def run(self, image_input: Union[bytes, str]) -> FinalResponse:
        """Executes full diagnostic pipeline through LangGraph."""
        start_time = time.perf_counter()

        initial_state: AgentState = {
            "image_bytes": image_input if isinstance(image_input, bytes) else None,
            "image_path": image_input if isinstance(image_input, str) else None,
            "trace": [],
        }

        # Invoke the compiled LangGraph workflow
        final_state: AgentState = self.graph.invoke(initial_state)

        total_latency = (time.perf_counter() - start_time) * 1000.0

        decision = final_state.get("abstention_decision")
        decision_state = decision.state if decision else AbstentionState.ABSTAINED

        return FinalResponse(
            decision_state=decision_state,
            suitability=final_state.get("suitability"),
            ensemble_result=final_state.get("ensemble_result"),
            abstention_decision=decision,
            evidence_records=final_state.get("validated_evidence", []),
            explanation=final_state.get("explanation"),
            trace=final_state.get("trace", []),
            total_latency_ms=round(total_latency, 2),
        )
