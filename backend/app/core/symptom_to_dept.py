"""
Runtime department prediction from symptom evidence codes.
Loads pre-computed mapping and provides prediction function.
"""

import json
import logging
import threading
from collections import Counter
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BACKEND_DIR / "model"
MAPPING_FILE = MODEL_DIR / "symptom_dept_mapping.json"
NAME_MAPPING_FILE = MODEL_DIR / "symptom_name_dept_mapping.json"

# Fallback keyword mapping — keyed by human symptom phrase, not by E_XX code,
# so we can also feed NER raw text into it. Kept in sync with the keyword
# fallback in scripts/create_symptom_dept_mapping.py (DEPT_KEYWORDS).
# NOTE: Pulmonology/Ophthalmology/ENT/etc. that aren't seeded in the
# `specialty` table are remapped to General Medicine / Internal Medicine here
# too, so we don't dead-end the slot-fetch downstream.
DEPT_KEYWORDS = {
    "Cardiology": ["chest pain", "palpitation", "heart", "cardiac", "bp", "blood pressure", "hypertension"],
    "Neurology": ["headache", "migraine", "dizziness", "seizure", "confusion", "weakness", "numbness", "tingling", "stroke", "memory"],
    "Respiratory": ["cough", "shortness of breath", "wheezing", "breathing", "lung", "asthma", "copd", "pneumonia"],
    "Gastroenterology": ["abdominal pain", "stomach pain", "nausea", "vomiting", " diarrhea", "constipation", "bloating", "acid reflux", "heartburn"],
    "Dermatology": ["rash", "itching", "hives", "skin", "acne", "eczema", "psoriasis", "mole", "lesion"],
    "Orthopedics": ["joint pain", "back pain", "knee pain", "shoulder pain", "neck pain", "fracture", "sprain", "arthritis", "muscle pain"],
    "Psychiatry": ["anxiety", "depression", "panic", "mood", "sleep", "insomnia", "stress"],
    "General Medicine / Internal Medicine": ["fever", "infection", "chills", "sweats", "feverish", "fatigue", "weight"],
}


class SymptomToDepartmentPredictor:
    """Predict department from symptom evidence codes."""

    def __init__(self):
        self.evidence_to_dept = {}
        self.name_to_dept = {}
        self._load_mappings()

    def _load_mappings(self):
        """Load pre-computed mappings."""
        if MAPPING_FILE.exists():
            with open(MAPPING_FILE, "r", encoding="utf-8") as f:
                self.evidence_to_dept = json.load(f)
            logger.info(f"Loaded {len(self.evidence_to_dept)} evidence->dept mappings")
        else:
            logger.warning(f"Mapping file not found: {MAPPING_FILE}")

        if NAME_MAPPING_FILE.exists():
            with open(NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
                self.name_to_dept = json.load(f)
            logger.info(f"Loaded {len(self.name_to_dept)} name->dept mappings")

    def predict(self, symptom_codes: List[str]) -> str:
        """
        Predict department from list of evidence codes (E_XX).
        Uses voting by frequency, falls back to General Medicine.
        """
        if not symptom_codes:
            return "General Medicine / Internal Medicine"

        # Vote by frequency over the pre-computed mapping. Codes that aren't in
        # the mapping don't contribute (no more hand-typed `E_XX` sets that
        # silently mis-routed patients — see `symptom_to_dept.py` history).
        dept_votes = Counter()
        for code in symptom_codes:
            base = code.split("_@_")[0] if "_@_" in code else code
            if base in self.evidence_to_dept:
                dept_votes[self.evidence_to_dept[base]] += 1

        if dept_votes:
            most_common = dept_votes.most_common(1)[0][0]
            logger.info(f"Predicted department from evidence codes: {most_common} (votes: {dict(dept_votes)})")
            return most_common

        return "General Medicine / Internal Medicine"

    def predict_from_text(self, text: str) -> str:
        """Predict department from free text via the keyword table."""
        text_lower = text.lower()
        for dept, keywords in DEPT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return dept
        return "General Medicine / Internal Medicine"


# Singleton instance with lock
_predictor = None
_predictor_lock = threading.Lock()


def get_predictor() -> SymptomToDepartmentPredictor:
    """Get or create predictor singleton (thread-safe)."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = SymptomToDepartmentPredictor()
    return _predictor


def predict_department_from_symptoms(symptom_codes: List[str]) -> str:
    """
    Predict department from symptom evidence codes.

    Args:
        symptom_codes: List of DDXPlus evidence codes (e.g., ["E_55", "E_53"])

    Returns:
        Department name string (always in the Supabase seed set)
    """
    predictor = get_predictor()
    return predictor.predict(symptom_codes)
