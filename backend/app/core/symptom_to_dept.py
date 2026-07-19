"""
Runtime department prediction from symptom evidence codes.
Loads pre-computed mapping and provides prediction function.
"""

import json
import os
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BACKEND_DIR / "model"
MAPPING_FILE = MODEL_DIR / "symptom_dept_mapping.json"
NAME_MAPPING_FILE = MODEL_DIR / "symptom_name_dept_mapping.json"

# Fallback keyword mapping
DEPT_KEYWORDS = {
    "Cardiology": ["chest pain", "palpitation", "heart", "cardiac", "bp", "blood pressure", "hypertension"],
    "Neurology": ["headache", "migraine", "dizziness", "seizure", "confusion", "weakness", "numbness", "tingling", "stroke", "memory"],
    "Pulmonology": ["cough", "shortness of breath", "wheezing", "breathing", "lung", "asthma", "copd", "pneumonia"],
    "Gastroenterology": ["abdominal pain", "stomach pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating", "acid reflux", "heartburn"],
    "Dermatology": ["rash", "itching", "hives", "skin", "acne", "eczema", "psoriasis", "mole", "lesion"],
    "Orthopedics": ["joint pain", "back pain", "knee pain", "shoulder pain", "neck pain", "fracture", "sprain", "arthritis", "muscle pain"],
    "Rheumatology": ["joint pain", "stiffness", "swelling", "autoimmune", "lupus", "rheumatoid"],
    "Endocrinology": ["diabetes", "thyroid", "hormone", "weight gain", "weight loss", "fatigue", "excessive thirst"],
    "Nephrology": ["kidney", "urine", "urination", "blood in urine", "kidney stone"],
    "Hematology": ["anemia", "bleeding", "bruising", "blood", "clotting"],
    "Infectious Disease": ["fever", "infection", "chills", "sweats", "feverish"],
    "Psychiatry": ["anxiety", "depression", "panic", "mood", "sleep", "insomnia", "stress"],
    "Ophthalmology": ["eye", "vision", "blurry", "eye pain", "red eye"],
    "ENT": ["ear", "throat", "nose", "sinus", "hearing", "sore throat", "tonsil"],
    "Urology": ["urinary", "prostate", "erectile", "testicular", "kidney stone"],
    "Gynecology": ["menstrual", "period", "pregnancy", "vaginal", "pelvic", "obgyn"],
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
            with open(MAPPING_FILE, 'r') as f:
                self.evidence_to_dept = json.load(f)
            logger.info(f"Loaded {len(self.evidence_to_dept)} evidence->dept mappings")
        else:
            logger.warning(f"Mapping file not found: {MAPPING_FILE}")
        
        if NAME_MAPPING_FILE.exists():
            with open(NAME_MAPPING_FILE, 'r') as f:
                self.name_to_dept = json.load(f)
            logger.info(f"Loaded {len(self.name_to_dept)} name->dept mappings")
    
    def predict(self, symptom_codes: List[str]) -> str:
        """
        Predict department from list of evidence codes (E_XX).
        Uses voting by frequency, falls back to keyword matching.
        """
        if not symptom_codes:
            return "General Medicine / Internal Medicine"
        
        # Vote by frequency
        dept_votes = Counter()
        
        for code in symptom_codes:
            if code in self.evidence_to_dept:
                dept_votes[self.evidence_to_dept[code]] += 1
        
        if dept_votes:
            most_common = dept_votes.most_common(1)[0][0]
            logger.info(f"Predicted department from evidence codes: {most_common} (votes: {dict(dept_votes)})")
            return most_common
        
        # Fallback: keyword matching on symptom names (if we had them)
        # For now, use a simple heuristic based on common codes
        return self._keyword_fallback(symptom_codes)
    
    def _keyword_fallback(self, symptom_codes: List[str]) -> str:
        """Fallback using known code patterns."""
        # Common code patterns
        cardio_codes = {"E_55", "E_56", "E_70", "E_71", "E_72"}  # chest pain, SOB, palpitations
        neuro_codes = {"E_53", "E_61", "E_73", "E_74", "E_75"}  # headache, dizziness, seizure
        pulmo_codes = {"E_56", "E_57", "E_76", "E_77"}  # cough, dyspnea, wheezing
        gi_codes = {"E_58", "E_59", "E_60", "E_68", "E_69"}  # nausea, vomiting, abdominal pain, diarrhea
        derm_codes = {"E_63", "E_78", "E_79"}  # rash, itching, hives
        ortho_codes = {"E_64", "E_65", "E_80", "E_81"}  # joint pain, back pain
        
        code_set = set(symptom_codes)
        
        if code_set & cardio_codes:
            return "Cardiology"
        if code_set & neuro_codes:
            return "Neurology"
        if code_set & pulmo_codes:
            return "Pulmonology"
        if code_set & gi_codes:
            return "Gastroenterology"
        if code_set & derm_codes:
            return "Dermatology"
        if code_set & ortho_codes:
            return "Orthopedics"
        
        return "General Medicine / Internal Medicine"


# Singleton instance
_predictor = None

def get_predictor() -> SymptomToDepartmentPredictor:
    """Get or create predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = SymptomToDepartmentPredictor()
    return _predictor


def predict_department_from_symptoms(symptom_codes: List[str]) -> str:
    """
    Predict department from symptom evidence codes.
    
    Args:
        symptom_codes: List of DDXPlus evidence codes (e.g., ["E_55", "E_53"])
    
    Returns:
        Department name string
    """
    predictor = get_predictor()
    return predictor.predict(symptom_codes)


# Add Counter import
from collections import Counter