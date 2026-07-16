"""
Improved Named Entity Recognition for symptom extraction.
Uses pattern-based matching + contextual extraction rather than just keyword lists.
"""

import re
import logging
from typing import List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SymptomSeverity(Enum):
    """Severity levels extracted from symptom descriptions."""
    MILD = 1
    MODERATE = 2
    SEVERE = 3


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
    
    # Duration patterns (extract timeframe)
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
        
        logger.info(f"Extracted {len(extracted_symptoms)} symptoms from text")
        for sym in extracted_symptoms:
            logger.debug(f"  - {sym['symptom']}: {sym['severity'].name}")
        
        return extracted_symptoms
    
    @staticmethod
    def _extract_severity_for_match(text: str, match_pos: int) -> SymptomSeverity:
        """Extract severity modifier near match position."""
        # Look at words before the match (within 20 chars)
        context_start = max(0, match_pos - 20)
        context = text[context_start:match_pos]
        
        for modifier, severity in SymptomsExtractor.SEVERITY_MODIFIERS.items():
            if modifier in context:
                return severity
        
        # Default to moderate
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
        
        # Look at context around match
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
