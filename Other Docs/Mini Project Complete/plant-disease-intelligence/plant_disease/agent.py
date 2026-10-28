"""Evidence workflow: retrieve, validate, explain. LangGraph when available.

    image -> run_vision -> (accepted?) -> retrieve_evidence -> validate_evidence
          -> explain_evidence -> finalize

* The classifier decides first. If it abstains (or predicts a healthy class) the
  workflow skips retrieval and the language model is never called.
* The language model only receives the retrieved passages and the diagnosis. It
  cannot change the diagnosis, the abstention decision or any score.
* Its text is checked against the evidence. If a citation or quotation is not backed
  by a retrieved passage, or it states a confidence figure, it is rejected and a
  plain extract of the retrieved sources is shown instead.
* If retrieval, the language model or the whole workflow engine fails, the
  classification result is still returned.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict

from plant_disease import retrieval as R
from plant_disease.config import get_logger

log = get_logger("agent")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MIN_EXPLANATION_CHARS = 40
_CONFIDENCE_CLAIM = re.compile(
    r"(confiden|probabilit|likelihood|certain|accura)[^.\n]{0,60}\d"
    r"|\d+(\.\d+)?\s*%[^.\n]{0,40}(confiden|certain|sure|likely)",
    re.IGNORECASE,
)


class ExplanationError(Exception):
    """The language model could not be used."""


# ==========================================================================
# Results
# ==========================================================================
@dataclass
class Explanation:
    text: str
    source: str                      # "llm" | "extractive" | "none"
    notes: List[str] = field(default_factory=list)
    model: Optional[str] = None


@dataclass
class AgentResult:
    vision: Any
    evidence_status: str             # not_requested | no_corpus | no_evidence | available | error
    evidence_notes: List[str]
    hits: List[R.EvidenceHit]
    citations: List[R.Citation]
    hit_numbers: List[int]
    explanation: Explanation
    limitations: List[str]
    trace: List[str]
    engine: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vision": self.vision.to_dict() if hasattr(self.vision, "to_dict") else None,
            "evidence_status": self.evidence_status,
            "evidence_notes": self.evidence_notes,
            "passages": [
                {"passage_id": h.passage.passage_id, "source_id": h.passage.source_id,
                 "score": round(h.score, 4), "match": h.match, "citation": n, "text": h.passage.text}
                for h, n in zip(self.hits, self.hit_numbers)
            ],
            "citations": [c.__dict__ for c in self.citations],
            "explanation": {"text": self.explanation.text, "source": self.explanation.source,
                            "notes": self.explanation.notes, "model": self.explanation.model},
            "limitations": self.limitations,
            "trace": self.trace,
            "engine": self.engine,
        }


class AgentState(TypedDict, total=False):
    image: Any
    vision: Any
    route: str
    hits: list
    citations: list
    hit_numbers: list
    evidence_status: str
    evidence_notes: list
    explanation: Any
    limitations: list
    trace: list


# ==========================================================================
# Explanation
# ==========================================================================
def _excerpt(text: str, limit: int = 420) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= limit // 2:
        return cut[:end + 1]
    space = cut.rfind(" ")
    return cut[: space if space > 0 else limit].rstrip(",;:") + "..."


def check_llm_text(text: str, citations: List[R.Citation], hits: List[R.EvidenceHit]) -> List[str]:
    """Reasons to reject a language-model explanation (empty list = acceptable)."""
    problems: List[str] = []
    if len(text.strip()) < MIN_EXPLANATION_CHARS:
        problems.append("the response was empty or too short")
        return problems
    problems += R.citation_marker_problems(text, len(citations))
    unsupported = R.find_unsupported_quotes(text, [h.passage for h in hits])
    if unsupported:
        problems.append(f"it quoted text that is not in the sources ({len(unsupported)} quote(s))")
    if _CONFIDENCE_CLAIM.search(text):
        problems.append("it stated a confidence/probability figure")
    return problems


class Explainer:
    """Writes the explanation from retrieved evidence only.

    The language model is optional: with no GROQ_API_KEY, or if it fails or is rejected,
    the explanation is an extract of the retrieved passages.
    """

    def __init__(self, config: Dict[str, Any],
                 llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None) -> None:
        cfg = config.get("llm", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self._llm_call = llm_call

    @property
    def model(self) -> str:
        return os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL

    @property
    def llm_available(self) -> bool:
        has_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
        return self.enabled and (self._llm_call is not None or has_key)

    def status(self) -> str:
        if not self.enabled:
            return "disabled in config"
        return f"configured ({self.model})" if self.llm_available else "not configured (no GROQ_API_KEY)"

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _redact(message: str) -> str:
        key = os.environ.get("GROQ_API_KEY", "")
        return message.replace(key, "***") if key and len(key) > 8 else message

    def _groq(self, messages: List[Dict[str, str]]) -> str:
        try:
            from groq import Groq
        except ImportError as exc:
            raise ExplanationError("the 'groq' package is not installed") from exc
        client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30)
        response = client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def extractive(class_name: str, hits: List[R.EvidenceHit], hit_numbers: List[int]) -> str:
        crop, disease = R.split_class_name(class_name)
        lines = [f"Excerpts from the retrieved sources about {disease}"
                 + (f" ({crop})" if crop and crop.lower() not in disease.lower() else "") + ":", ""]
        for h, n in zip(hits, hit_numbers):
            lines.append(f"- {_excerpt(h.passage.text)} [{n}]")
        return "\n".join(lines)

    @staticmethod
    def _messages(class_name: str, hits: List[R.EvidenceHit], hit_numbers: List[int]) -> List[Dict[str, str]]:
        crop, disease = R.split_class_name(class_name)
        system = (
            "You write short, factual plant-disease guidance for farmers using ONLY the numbered "
            "evidence passages provided.\n"
            "Rules:\n"
            "1. Use only facts stated in the evidence. Add no outside knowledge.\n"
            "2. Cite every claim with its marker such as [1] or [2]. Use only the marker numbers given.\n"
            "3. If the evidence does not cover something, say so. Do not guess.\n"
            "4. The diagnosis was decided by an image classifier. Never question or change it, and never "
            "state or estimate any confidence, probability, accuracy or certainty.\n"
            "5. Give no pesticide products, doses or rates unless they appear in the evidence; if they "
            "do, tell the reader to follow the product label and local extension advice.\n"
            "6. Quote only exact words from the evidence, inside quotation marks; otherwise paraphrase.\n"
            "7. About 120-220 words, plain language, with short sections: What it is, Signs, What to do."
        )
        evidence = "\n\n".join(
            f"[{n}] ({h.passage.publisher}; {h.passage.title})\n{h.passage.text}"
            for h, n in zip(hits, hit_numbers)
        )
        user = (f"Diagnosis (already decided by an image classifier): {disease}"
                f"{' on ' + crop if crop and crop.lower() not in disease.lower() else ''}.\n\n"
                f"Evidence:\n{evidence}\n\nWrite the explanation.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ---- main -----------------------------------------------------------------
    def explain(self, class_name: str, hits: List[R.EvidenceHit], citations: List[R.Citation],
                hit_numbers: List[int]) -> Explanation:
        if not hits:
            return Explanation("", "none", ["No evidence passages were available."])
        fallback = self.extractive(class_name, hits, hit_numbers)
        if not self.llm_available:
            note = ("Language model disabled in config." if not self.enabled
                    else "No language model configured; showing excerpts from the sources.")
            return Explanation(fallback, "extractive", [note])
        try:
            call = self._llm_call or self._groq
            text = (call(self._messages(class_name, hits, hit_numbers)) or "").strip()
        except Exception as exc:  # noqa: BLE001 - never let the LLM break the result
            reason = self._redact(f"{type(exc).__name__}: {str(exc)[:160]}")
            log.warning("Language model call failed: %s", reason)
            return Explanation(fallback, "extractive",
                               [f"Language model unavailable ({reason}); showing excerpts instead."])
        problems = check_llm_text(text, citations, hits)
        if problems:
            log.warning("Language model text rejected: %s", "; ".join(problems))
            return Explanation(fallback, "extractive",
                               ["AI-written explanation rejected because " + "; ".join(problems)
                                + ". Showing excerpts from the sources instead."])
        return Explanation(text, "llm", [], self.model if self._llm_call is None else "injected")


# ==========================================================================
# Limitations shown with every result
# ==========================================================================
def build_limitations(vision: Any, evidence_status: str, explanation: Optional[Explanation]) -> List[str]:
    out = ["This is a decision aid for research, not a substitute for a plant pathologist or "
           "local agricultural extension advice."]
    status = getattr(vision, "policy_status", "") or ""
    if not status.startswith("validated"):
        out.append(f"Abstention thresholds are {status or 'unknown'}; they have not been fully "
                   "validated on held-out data.")
    if getattr(vision, "calibrated", False):
        out.append("Confidence figures use a temperature fitted on the validation split; they are "
                   "still estimates, not guarantees.")
    else:
        out.append("Confidence figures are raw model scores, not calibrated probabilities.")
    unverified = getattr(vision, "unverified_preprocessing", []) or []
    if unverified:
        out.append("Preprocessing is provisional (not confirmed from training metadata) for: "
                   + ", ".join(unverified) + ".")
    excluded = getattr(vision, "excluded", {}) or {}
    if excluded:
        out.append(f"{len(excluded)} model(s) did not contribute: {', '.join(excluded)}.")
    note = getattr(vision, "calibration_note", "")
    if note:
        out.append(note)
    for w in getattr(vision, "image_warnings", []) or []:
        out.append(f"Image: {w}")
    if evidence_status == "no_evidence":
        out.append("No evidence from the curated sources was available for this disease.")
    elif evidence_status == "no_corpus":
        out.append("The evidence corpus has not been built yet (run scripts\\build_evidence.py).")
    elif explanation is not None and explanation.source == "llm":
        out.append("The explanation was written by a language model from the retrieved sources; "
                   "check the cited sources.")
    elif explanation is not None and explanation.source == "extractive":
        out.append("The explanation consists of excerpts from the retrieved sources.")
    return out


# ==========================================================================
# Workflow
# ==========================================================================
class EvidenceWorkflow:
    """Runs the vision step and, when accepted, the evidence steps."""

    def __init__(self, vision_fn: Callable[[Any], Any], index: R.EvidenceIndex,
                 explainer: Explainer, config: Dict[str, Any], use_langgraph: bool = True) -> None:
        self.vision_fn = vision_fn
        self.index = index
        self.explainer = explainer
        self.config = config
        self.allowed = config["retrieval"]["allowed_domains"]
        self._graph = None
        self.engine = "sequential"
        if use_langgraph:
            try:
                self._graph = self._build_graph()
                self.engine = "langgraph"
            except Exception as exc:  # noqa: BLE001 - langgraph missing or incompatible
                log.warning("LangGraph unavailable (%s); using the sequential workflow.",
                            type(exc).__name__)
                self._graph = None

    # ---- nodes ---------------------------------------------------------------
    @staticmethod
    def _trace(state: AgentState, entry: str) -> List[str]:
        return list(state.get("trace", [])) + [entry]

    def _run_vision(self, state: AgentState) -> Dict[str, Any]:
        vision = self.vision_fn(state["image"])
        update: Dict[str, Any] = {"vision": vision, "hits": [], "citations": [], "hit_numbers": [],
                                  "evidence_notes": [], "route": "finish"}
        cls = getattr(vision, "predicted_class", None)
        if not getattr(vision, "accepted", False):
            update["evidence_status"] = "not_requested"
            update["evidence_notes"] = ["The classifier abstained, so no evidence was retrieved."]
            entry = "run_vision: abstained"
        elif cls is None or R.is_healthy_class(cls):
            update["evidence_status"] = "not_requested"
            update["evidence_notes"] = ["A healthy leaf was predicted; no disease guidance to retrieve."]
            entry = "run_vision: healthy"
        else:
            update["route"] = "retrieve"
            entry = f"run_vision: accepted {cls}"
        update["trace"] = self._trace(state, entry)
        return update

    def _retrieve(self, state: AgentState) -> Dict[str, Any]:
        cls = state["vision"].predicted_class
        notes: List[str] = []
        hits: List[R.EvidenceHit] = []
        try:
            if not self.index.ready:
                status = "no_corpus"
                notes.append(self.index.notice or "The evidence corpus is empty.")
            elif not self.index.has_evidence_for(cls):
                status = "no_evidence"
                notes.append(f"None of the curated sources is tagged for {cls}.")
            else:
                rc = self.config["retrieval"]
                hits = self.index.search(cls, int(rc.get("top_k", 4)), float(rc.get("min_score", 0.05)))
                status = "available" if hits else "no_evidence"
                if not hits:
                    notes.append("No passage was similar enough to the disease description.")
        except Exception as exc:  # noqa: BLE001
            status = "error"
            notes.append(f"Evidence retrieval failed: {type(exc).__name__}: {str(exc)[:120]}")
        return {"hits": hits, "evidence_status": status, "evidence_notes": notes,
                "trace": self._trace(state, f"retrieve_evidence: {len(hits)} passage(s), {status}")}

    def _validate(self, state: AgentState) -> Dict[str, Any]:
        hits = state.get("hits", [])
        notes = list(state.get("evidence_notes", []))
        good, problems = R.validate_hits(hits, self.allowed)
        notes += [f"Dropped a passage: {p}" for p in problems]
        citations, numbers = R.build_citations(good)
        status = state.get("evidence_status", "no_evidence")
        if hits and not good:
            status = "no_evidence"
        return {"hits": good, "citations": citations, "hit_numbers": numbers,
                "evidence_status": status, "evidence_notes": notes,
                "trace": self._trace(state, f"validate_evidence: {len(good)} valid passage(s)")}

    def _explain(self, state: AgentState) -> Dict[str, Any]:
        cls = state["vision"].predicted_class
        try:
            explanation = self.explainer.explain(cls, state["hits"], state["citations"],
                                                 state["hit_numbers"])
        except Exception as exc:  # noqa: BLE001
            explanation = Explanation("", "none", [f"Explanation step failed: {type(exc).__name__}"])
        return {"explanation": explanation,
                "trace": self._trace(state, f"explain_evidence: {explanation.source}")}

    def _finalize(self, state: AgentState) -> Dict[str, Any]:
        explanation = state.get("explanation") or Explanation("", "none", [])
        return {"explanation": explanation,
                "limitations": build_limitations(state["vision"], state.get("evidence_status", ""),
                                                 explanation),
                "trace": self._trace(state, "finalize")}

    # ---- graph ---------------------------------------------------------------
    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("run_vision", self._run_vision)
        graph.add_node("retrieve_evidence", self._retrieve)
        graph.add_node("validate_evidence", self._validate)
        graph.add_node("explain_evidence", self._explain)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("run_vision")
        graph.add_conditional_edges("run_vision", lambda s: s["route"],
                                    {"retrieve": "retrieve_evidence", "finish": "finalize"})
        graph.add_edge("retrieve_evidence", "validate_evidence")
        graph.add_conditional_edges("validate_evidence",
                                    lambda s: "explain" if s.get("hits") else "finish",
                                    {"explain": "explain_evidence", "finish": "finalize"})
        graph.add_edge("explain_evidence", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _run_sequential(self, state: AgentState) -> AgentState:
        state.update(self._run_vision(state))
        if state["route"] == "retrieve":
            state.update(self._retrieve(state))
            state.update(self._validate(state))
            if state.get("hits"):
                state.update(self._explain(state))
        state.update(self._finalize(state))
        return state

    def run(self, image: Any) -> AgentResult:
        state: AgentState = {"image": image, "trace": []}
        final = self._graph.invoke(state) if self._graph is not None else self._run_sequential(state)
        return AgentResult(
            vision=final["vision"],
            evidence_status=final.get("evidence_status", "not_requested"),
            evidence_notes=list(final.get("evidence_notes", [])),
            hits=list(final.get("hits", [])),
            citations=list(final.get("citations", [])),
            hit_numbers=list(final.get("hit_numbers", [])),
            explanation=final.get("explanation") or Explanation("", "none", []),
            limitations=list(final.get("limitations", [])),
            trace=list(final.get("trace", [])),
            engine=self.engine,
        )
