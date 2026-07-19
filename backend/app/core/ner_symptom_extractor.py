"""
Biomedical Named Entity Recognition for symptom extraction.
Uses d4data/biomedical-ner-all HuggingFace model with regex fallback.
Extracts Sign_symptom, Disease_disorder, Detailed_description entities
and maps them to DDXPlus evidence codes (E_XX).
"""

import re
import logging
import json
import os
from typing import List, Dict, Set
from enum import Enum

logger = logging.getLogger(__name__)


class SymptomSeverity(Enum):
    """Severity levels extracted from symptom descriptions."""
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class BiomedicalNER:
    """HuggingFace Biomedical NER pipeline for symptom extraction."""
    
    def __init__(self):
        self.ner_pipeline = None
        self.evidence_to_code = self._load_ddxplus_evidence_mapping()
        self._load_model()
    
    def _load_model(self):
        """Load HF NER pipeline with GPU/CPU fallback."""
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading biomedical NER model on device: {'cuda' if device == 0 else 'cpu'}")
            
            self.ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=device
            )
            logger.info("HF Biomedical NER model loaded successfully")
        except Exception as e:
            logger.warning(f"HF NER load failed: {e}, falling back to regex")
            self.ner_pipeline = None
    
    def _load_ddxplus_evidence_mapping(self) -> Dict[str, str]:
        """Load DDXPlus evidence name to code mapping (E_XX)."""
        mapping = {}
        try:
            kg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/ddxplus_evidences.json"))
            if os.path.exists(kg_path):
                with open(kg_path, 'r') as f:
                    evidences = json.load(f)
                for code, info in evidences.items():
                    name = info.get("name", "").lower()
                    if name:
                        mapping[name] = code
                        # Also map variations
                        mapping[name.replace("_", " ")] = code
                        mapping[name.replace("-", " ")] = code
            logger.info(f"Loaded {len(mapping)} evidence name mappings")
        except Exception as e:
            logger.warning(f"Could not load evidence mapping: {e}")
        return mapping
    
    def _map_to_evidence_codes(self, entities: List[Dict]) -> List[str]:
        """Map extracted entities to DDXPlus evidence codes."""
        codes = []
        for entity in entities:
            entity_text = entity.get('word', '').lower().strip()
            entity_group = entity.get('entity_group', '')
            
            if entity_group in ['Sign_symptom', 'Disease_disorder', 'Detailed_description']:
                # Try exact match first
                if entity_text in self.evidence_to_code:
                    codes.append(self.evidence_to_code[entity_text])
                else:
                    # Try fuzzy match - check if evidence name is contained in entity
                    for ev_name, ev_code in self.evidence_to_code.items():
                        if ev_name in entity_text or entity_text in ev_name:
                            codes.append(ev_code)
                            break
        
        # Deduplicate while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        return unique_codes
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text and return DDXPlus evidence codes (E_XX)."""
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                # Filter for relevant entity types
                symptom_entities = [
                    e for e in entities 
                    if e.get('entity_group') in ['Sign_symptom', 'Disease_disorder', 'Detailed_description']
                ]
                codes = self._map_to_evidence_codes(symptom_entities)
                if codes:
                    logger.info(f"HF NER extracted {len(codes)} evidence codes: {codes}")
                    return codes
            except Exception as e:
                logger.warning(f"HF NER extraction failed: {e}, using regex fallback")
        
        # Fallback to regex-based extraction
        return self._regex_fallback(text)
    
    def _regex_fallback(self, text: str) -> List[str]:
        """Regex-based symptom extraction fallback (original SymptomsExtractor logic)."""
        text_lower = text.lower()
        codes = []
        
        # Direct mapping of common symptoms to evidence codes
        symptom_to_code = {
            "chest pain": "E_55",
            "headache": "E_53", 
            "fever": "E_91",
            "shortness of breath": "E_56",
            "difficulty breathing": "E_56",
            "cough": "E_57",
            "nausea": "E_58",
            "vomiting": "E_59",
            "abdominal pain": "E_60",
            "dizziness": "E_61",
            "fatigue": "E_62",
            "rash": "E_63",
            "joint pain": "E_64",
            "back pain": "E_65",
            "sore throat": "E_66",
            "runny nose": "E_67",
            "congestion": "E_67",
            "diarrhea": "E_68",
            "constipation": "E_69",
        }
        
        for symptom, code in symptom_to_code.items():
            if symptom in text_lower:
                codes.append(code)
        
        return codes


# Keep the original regex-based extractor as a backup
class SymptomsExtractor:
    """Extract symptoms from patient descriptions with severity and context."""
    
    # Medical symptom vocabulary
    SYMPTOM_PATTERNS = {
        # Pain patterns
        r'\b(pain|ache|soreness|tenderness|discomfort)\b': 'pain',
        r'\b(severe|intense|unbearable|excruciating)\s+(pain|ache)\b': 'severe_pain',
        r'\b(dull|sharp|stabbing|throbbing|burning|aching)\b': 'pain_type',
        
        # Respiratory
        r'\b(cough|coughing)\b': 'cough',
        r'\b(breathless|dyspnea|difficulty breathing|shortness of breath|can\'t breathe|SOB)\b': 'dyspnea',
        r'\b(wheezing|wheeze|wheezy)\b': 'wheezing',
        r'\b(runny nose|nasal congestion|stuffy nose|sneezing)\b': 'nasal_symptoms',
        
        # Gastrointestinal
        r'\b(nausea|nauseous|nauseated)\b': 'nausea',
        r'\b(vomit|vomiting|threw up|throwing up)\b': 'vomiting',
        r'\b(diarrhea|loose stool|diarrheal)\b': 'diarrhea',
        r'\b(constipation|constipated|blocked)\b': 'constipation',
        r'\b(abdominal|stomach|belly)\s+(pain|ache|cramps)\b': 'abdominal_pain',
        
        # Fever/Systemic
        r'\b(fever|feverish|febrile|temperature|temp)\b': 'fever',
        r'\b(chills|chilling)\b': 'chills',
        r'\b(fatigue|tired|exhaustion|weakness)\b': 'fatigue',
        r'\b(malaise|generally unwell)\b': 'malaise',
        
        # Neurological
        r'\b(headache|migraine|head pain)\b': 'headache',
        r'\b(dizzy|dizziness|vertigo|lightheaded)\b': 'dizziness',
        r'\b(confusion|confused|disorientation|disoriented)\b': 'confusion',
        r'\b(seizure|seizures|convulsion)\b': 'seizure',
        
        # Cardiovascular
        r'\b(chest pain|chest ache|chest discomfort)\b': 'chest_pain',
        r'\b(palpitation|heart pounding|pounding heart)\b': 'palpitations',
        r'\b(shortness of breath|dyspnea)\b': 'dyspnea_cardiac',
        
        # Skin
        r'\b(rash|rashes|skin rash)\b': 'rash',
        r'\b(itching|itchy|pruritus|scratch)\b': 'itching',
        r'\b(hives|urticaria)\b': 'hives',
        r'\b(swelling|edema|swollen|puffiness)\b': 'swelling',
    }
    
    # Severity indicators
    SEVERITY_MODIFIERS = {
        'severe': SymptomSeverity.SEVERE,
        'intense': SymptomSeverity.SEVERE,
        'unbearable': SymptomSeverity.SEVERE,
        'excruciating': SymptomSeverity.SEVERE,
        'moderate': SymptomSeverity.MODERATE,
        'mild': SymptomSeverity.MILD,
        'slight': SymptomSeverity.MILD,
        'light': SymptomSeverity.MILD,
        'little': SymptomSeverity.MILD,
    }
    
    # Duration patterns
    DURATION_PATTERNS = [
        (r'(\d+)\s*(second|minute|min|hour|hr|day|week|month|year)s?\b', 'duration'),
        (r'\b(since|for|over|past)\s+(this\s+)?(morning|afternoon|evening|night|today|yesterday)\b', 'recent'),
        (r'\b(started|began|came on)\s+((\d+\s*)?(days|hours|weeks|minutes)\s+ago)\b', 'recent'),
    ]
    
    @staticmethod
    def extract_symptoms(text: str) -> List[Dict]:
        """
        Extract symptoms from patient description.
        
        Returns:
            List of {
                'symptom': str,
                'severity': SymptomSeverity,
                'context': str,
                'location': str (if applicable)
            }
        """
        text_lower = text.lower()
        extracted_symptoms = []
        matched_patterns = set()
        
        # Match patterns
        for pattern, symptom_type in SymptomsExtractor.SYMPTOM_PATTERNS.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if match.start() not in matched_patterns:
                    matched_patterns.add(match.start())
                    
                    # Extract severity from context
                    severity = SymptomsExtractor._extract_severity_for_match(text_lower, match.start())
                    
                    # Extract location if applicable
                    location = SymptomsExtractor._extract_location_for_match(text_lower, match.start())
                    
                    extracted_symptoms.append({
                        'symptom': symptom_type,
                        'raw_text': match.group(0),
                        'severity': severity,
                        'location': location,
                        'confidence': 0.9
                    })
        
        # Extract duration
        duration = SymptomsExtractor._extract_duration(text)
        
        logger.info(f"Extracted {len(extracted_symptoms)} symptoms from text (regex fallback)")
        for sym in extracted_symptoms:
            logger.debug(f"  - {sym['symptom']}: {sym['severity'].name}")
        
        return extracted_symptoms
    
    @staticmethod
    def _extract_severity_for_match(text: str, match_pos: int) -> SymptomSeverity:
        """Extract severity modifier near match position."""
        context_start = max(0, match_pos - 20)
        context = text[context_start:match_pos]
        
        for modifier, severity in SymptomsExtractor.SEVERITY_MODIFIERS.items():
            if modifier in context:
                return severity
        
        return SymptomSeverity.MODERATE
    
    @staticmethod
    def _extract_location_for_match(text: str, match_pos: int) -> str:
        """Extract anatomical location if present."""
        location_patterns = [
            'head', 'neck', 'shoulder', 'chest', 'back', 'abdomen', 'stomach',
            'leg', 'arm', 'hand', 'foot', 'joint', 'knee', 'elbow',
            'throat', 'mouth', 'eye', 'ear', 'nose', 'right', 'left',
            'upper', 'lower', 'front', 'side'
        ]
        
        context_start = max(0, match_pos - 30)
        context_end = min(len(text), match_pos + 30)
        context = text[context_start:context_end]
        
        for location in location_patterns:
            if location in context:
                return location
        
        return None
    
    @staticmethod
    def _extract_duration(text: str) -> Dict:
        """Extract symptom duration from text."""
        for pattern, duration_type in SymptomsExtractor.DURATION_PATTERNS:
            match = re.search(pattern, text.lower())
            if match:
                return {
                    'type': duration_type,
                    'raw_text': match.group(0),
                    'confidence': 0.8
                }
        return None
    
    @staticmethod
    def get_symptom_summary(symptoms: List[Dict]) -> str:
        """Generate summary of extracted symptoms."""
        if not symptoms:
            return "No symptoms detected"
        
        grouped = {}
        for sym in symptoms:
            symptom_type = sym['symptom']
            if symptom_type not in grouped:
                grouped[symptom_type] = []
            grouped[symptom_type].append(sym)
        
        summary_parts = []
        for symptom_type, instances in grouped.items():
            severity = max(s['severity'].value for s in instances)
            severity_name = SymptomSeverity(severity).name.lower()
            summary_parts.append(f"{symptom_type} ({severity_name})")
        
        return ", ".join(summary_parts)


# System prompts for Ollama
SYSTEM_PROMPTS = {
    'triage': """You are a medical triage assistant. Your role is to:
1. Listen to patient symptoms
2. Ask clarifying questions to understand severity
3. Assess urgency level
4. Recommend appropriate medical department

Be professional, empathetic, and concise. Avoid medical jargon unless patient uses it.
Never diagnose - only triage and recommend specialist.""",
    
    'question_generation': """You are a medical interview assistant. Generate clear, conversational questions to:
1. Understand symptom characteristics
2. Assess symptom severity and duration
3. Identify relevant medical history
4. Narrow down differential diagnoses

Questions should be natural, not clinical. Avoid lists. Ask one question at a time.""",
    
    'explanation': """You are a patient education assistant. Your role is to:
1. Explain why a specific specialist is recommended
2. Describe what to expect
3. Provide reassurance
4. Encourage follow-up care

Use simple language. Avoid medical jargon. Be accurate but not alarming.""",
    
    'follow_up': """You are a follow-up assistant. Based on patient responses:
1. Clarify any ambiguous symptoms
2. Ask about related conditions
3. Assess medication/allergy history
4. Identify red flags

Be thorough but conversational."""
}


def get_system_prompt(prompt_type: str = 'triage') -> str:
    """Get system prompt for Ollama."""
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS['triage'])


# Singleton instance for HF NER
_biomedical_ner = None

def get_biomedical_ner() -> BiomedicalNER:
    """Get or create BiomedicalNER singleton."""
    global _biomedical_ner
    if _biomedical_ner is None:
        _biomedical_ner = BiomedicalNER()
    return _biomedical_ner
