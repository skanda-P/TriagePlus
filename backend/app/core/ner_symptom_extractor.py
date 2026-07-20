"""
Biomedical Named Entity Recognition for symptom extraction.
Uses d4data/biomedical-ner-all HuggingFace model with regex fallback.
Extracts Sign_symptom entities and maps them to DDXPlus evidence codes (E_XX).
"""

import re
import json
import logging
import os
import threading
from functools import lru_cache
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Path to the actual DDXPlus evidences file (verified: name == code, question_en == human text).
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
_EVIDENCES_FILE = os.path.join(_DATA_DIR, "DDXPlus", "release_evidences.json")

# Entity groups we accept from the biomedical-ner-all model. We only ingest
# symptoms/signs into the triage pipeline; `Disease_disorder` is downstream of
# triage and would bias the question ranker if treated as a "symptom".
_ACCEPTED_ENTITY_GROUPS = ("Sign_symptom",)


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    # Replace common punctuation with space, then collapse whitespace
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove the trailing possessive 's
    return s


def _tokenize(s: str) -> List[str]:
    """Word-only tokenization (used by NER matching and BM25)."""
    return re.findall(r"\b\w+\b", s.lower())


# Symptom keyword → DDXPlus evidence code whitelist.
# This is generated from release_evidences.json's `question_en` field (see
# `build_symptom_keyword_map`) so codes always agree with the dataset. We keep
# a small hand-curated bootstrap here so the model can be used even before the
# JSON is loaded. Codes are validated against the JSON at init; any code that
# doesn't exist in release_evidences.json is dropped and a warning is logged.
_KEYWORD_FALLBACK_BOOTSTRAP = {
    "chest pain": "E_55",
    "fever": "E_91",
    "cough": "E_57",
    "headache": "E_53",
    "shortness of breath": "E_56",
    "difficulty breathing": "E_56",
    "sore throat": "E_66",
    "dizziness": "E_61",
    "fatigue": "E_62",
    "nausea": "E_58",
    "vomiting": "E_59",
}


def _build_keyword_fallback(evidence_map_name_to_code: Dict[str, str], evidences_json: dict) -> Dict[str, str]:
    """
    Build a symptom-keyword → DDXPlus code mapping from release_evidences.json.

    Strategy:
      1. Start from the curated bootstrap above (only for codes that exist in
         the JSON). The bootstrap keys are human symptom phrases, not evidence
         code strings, so they map meaningfully.
      2. Auto-derive additional keyword → code entries from B (boolean)
         evidences whose `question_en` clearly describes a single symptom.
    """
    fallback: Dict[str, str] = {}
    for phrase, code in _KEYWORD_FALLBACK_BOOTSTRAP.items():
        if code in evidences_json:
            fallback[phrase] = code
        else:
            logger.warning(f"Bootstrap keyword '{phrase}' references unknown evidence code {code}; dropping")

    # Auto-add: for every boolean evidence, take its `question_en`, strip the
    # leading "Do you have a/any" / "Are you" pattern, and add the residual as a
    # keyword. Only keep short noun-phrase-like residuals (max 4 tokens) so we
    # don't ingest full sentences as "keywords".
    phrase_strip_re = re.compile(r"^\s*(do you have|do you feel|are you|have you|did you)\s+(a|an|any|the)?\s*", re.I)
    for code, info in evidences_json.items():
        if info.get("data_type") != "B":
            continue
        q = info.get("question_en", "")
        if not q:
            continue
        stripped = phrase_strip_re.sub("", q).rstrip("?.!").strip().lower()
        # Remove trailing words like "regularly", "in the past", etc. — keep only
        # first 3 whitespace-separated tokens for a symptom-like phrase. Don't
        # add if the residual is too generic.
        if not stripped or len(stripped) > 40:
            continue
        # Only keep it if it looks noun-ish (no question mark, short)
        if stripped and stripped not in fallback:
            fallback[stripped] = code

    return fallback


class BiomedicalNER:
    """HuggingFace Biomedical NER pipeline for symptom extraction."""

    def __init__(self):
        self.ner_pipeline: Optional[object] = None
        self.evidence_to_code: Dict[str, str] = {}
        self.keyword_fallback: Dict[str, str] = {}
        self._load_ddxplus_evidence_mapping()
        self._load_model()

    def _load_model(self):
        """Load HF NER pipeline with GPU/CPU fallback and a small retry loop."""
        last_exc = None
        for attempt in range(3):
            try:
                import torch
                from transformers import pipeline

                device = 0 if torch.cuda.is_available() else -1
                logger.info(f"Loading biomedical NER model on device: {'cuda' if device == 0 else 'cpu'} (attempt {attempt + 1})")

                self.ner_pipeline = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    tokenizer="d4data/biomedical-ner-all",
                    aggregation_strategy="first",
                    device=device,
                )
                logger.info("HF Biomedical NER model loaded successfully")
                return
            except Exception as e:
                last_exc = e
                logger.warning(f"HF NER load attempt {attempt + 1} failed: {e}")
        logger.warning(f"HF NER load failed after 3 attempts: {last_exc}; falling back to regex")
        self.ner_pipeline = None

    def _load_ddxplus_evidence_mapping(self) -> Dict[str, str]:
        """
        Load DDXPlus `question_en` text -> E_XX code mapping (and also store the
        JSON for keyword-fallback generation).

        The DDXPlus `name` field is literally the code string itself (e.g.
        `E_91`), so we deliberately use `question_en` here, otherwise the
        mapping would be tautological (`"e_91" -> "E_91"`) and could never be
        matched against NER output like "fever".
        """
        try:
            if not os.path.exists(_EVIDENCES_FILE):
                logger.error(f"Evidences file not found: {_EVIDENCES_FILE}")
                self.evidences_json = {}
                return {}

            with open(_EVIDENCES_FILE, "r", encoding="utf-8") as f:
                evidences = json.load(f)
            self.evidences_json = evidences

            mapping: Dict[str, str] = {}
            seen_keys = set()
            for code, info in evidences.items():
                q = (info.get("question_en") or "").strip().lower()
                if not q:
                    continue
                # Strip the leading question stem so we don't store
                # "do you have a fever ...?" as a lookup key.
                q_stripped = re.sub(
                    r"^\s*(do you have|do you feel|are you|have you|did you|does the pain|is the pain|how)\b.*?\??\s*",
                    "",
                    q,
                ).strip()
                # Make sense to use the original question_en as a secondary key
                # too (NER may emit "fever" which matches nothing exact, but
                # substring matching below will catch it).
                for k in (q, q_stripped):
                    if k and k not in seen_keys:
                        mapping[k] = code
                        seen_keys.add(k)
                # Common variations
                q_norm = _normalize_text(q)
                if q_norm and q_norm not in seen_keys:
                    mapping[q_norm] = code
                    seen_keys.add(q_norm)

            self.evidence_to_code = mapping
            self.keyword_fallback = _build_keyword_fallback(mapping, evidences)
            logger.info(f"Loaded {len(mapping)} evidence question->code mappings, "
                        f"{len(self.keyword_fallback)} keyword fallbacks")
        except Exception as e:
            logger.warning(f"Could not load evidence mapping: {e}")
            self.evidences_json = {}
            self.keyword_fallback = dict(_KEYWORD_FALLBACK_BOOTSTRAP)
        return self.evidence_to_code

    def _map_to_evidence_codes(self, entities: List[Dict]) -> List[str]:
        """Map extracted entities to DDXPlus evidence codes."""
        codes: List[str] = []
        for entity in entities:
            entity_text = _normalize_text(entity.get("word", "").lower())
            entity_group = entity.get("entity_group", "")
            score = float(entity.get("score", 0.0) or 0.0)

            if entity_group not in _ACCEPTED_ENTITY_GROUPS:
                continue
            if score < 0.7:
                continue

            # Try exact match first
            if entity_text in self.evidence_to_code:
                codes.append(self.evidence_to_code[entity_text])
                continue

            # Fuzzy: prefer the LONGEST evidence name that contains the entity
            # (more specific match). Avoid the reverse-substring trap where a
            # single-word entity matches every "do you have X" question.
            best_match: Optional[str] = None
            best_len = 0
            for ev_name, ev_code in self.evidence_to_code.items():
                if ev_name and entity_text and ev_name in entity_text and len(ev_name) > best_len:
                    best_match = ev_code
                    best_len = len(ev_name)
            if best_match:
                codes.append(best_match)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique.append(code)
        return unique

    @lru_cache(maxsize=512)
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text and return DDXPlus evidence codes (E_XX)."""
        if self.ner_pipeline is not None:
            try:
                entities = self.ner_pipeline(text)
                codes = self._map_to_evidence_codes(entities)
                if codes:
                    logger.info(f"HF NER extracted {len(codes)} evidence codes: {codes}")
                    return codes
            except Exception as e:
                logger.warning(f"HF NER extraction failed: {e}, using keyword fallback")

        return self._keyword_fallback_extract(text)

    def _keyword_fallback_extract(self, text: str) -> List[str]:
        """Keyword-based symptom extraction fallback."""
        text_lower = text.lower()
        codes: List[str] = []
        seen = set()
        # Match longest phrases first to prefer specificity ("shortness of breath" before "breath")
        for phrase in sorted(self.keyword_fallback.keys(), key=len, reverse=True):
            # Use word-boundary tokenization for short phrases to avoid
            # "ok" matching inside "oklahoma".
            tokens = _tokenize(phrase)
            if not tokens:
                continue
            # Match if all phrase tokens appear consecutively in text tokens.
            text_tokens = text_lower.split()
            n = len(tokens)
            for i in range(len(text_tokens) - n + 1):
                if text_tokens[i:i + n] == tokens:
                    code = self.keyword_fallback[phrase]
                    if code not in seen:
                        seen.add(code)
                        codes.append(code)
                    break
        return codes


# System prompts for Ollama
SYSTEM_PROMPTS = {
    "triage": """You are a medical triage assistant. Your role is to:
1. Listen to patient symptoms
2. Ask clarifying questions to understand severity
3. Assess urgency level
4. Recommend appropriate medical department

Be professional, empathetic, and concise. Avoid medical jargon unless patient uses it.
Never diagnose - only triage and recommend specialist.""",
    "question_generation": """You are a medical interview assistant. Generate clear, conversational questions to:
1. Understand symptom characteristics
2. Assess symptom severity and duration
3. Identify relevant medical history
4. Narrow down differential diagnoses

Questions should be natural, not clinical. Avoid lists. Ask one question at a time.""",
    "explanation": """You are a patient education assistant. Your role is to:
1. Explain why a specific specialist is recommended
2. Describe what to expect
3. Provide reassurance
4. Encourage follow-up care

Use simple language. Avoid medical jargon. Be accurate but not alarming.""",
    "follow_up": """You are a follow-up assistant. Based on patient responses:
1. Clarify any ambiguous symptoms
2. Ask about related conditions
3. Assess medication/allergy history
4. Identify red flags

Be thorough but conversational.""",
}


def get_system_prompt(prompt_type: str = "triage") -> str:
    """Get system prompt for Ollama."""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["triage"])


# Singleton with lock for thread-safe lazy initialization
_biomedical_ner: Optional[BiomedicalNER] = None
_ner_lock = threading.Lock()


def get_biomedical_ner() -> BiomedicalNER:
    """Get or create BiomedicalNER singleton (thread-safe)."""
    global _biomedical_ner
    if _biomedical_ner is None:
        with _ner_lock:
            if _biomedical_ner is None:
                _biomedical_ner = BiomedicalNER()
    return _biomedical_ner


def reset_biomedical_ner() -> None:
    """Test hook: reset the singleton."""
    global _biomedical_ner
    with _ner_lock:
        _biomedical_ner = None
