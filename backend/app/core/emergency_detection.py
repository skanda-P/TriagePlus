"""
Multi-layer emergency detection with conservative approach.
- Layer 1 (Floor): Explicit critical keywords - can never be overridden
- Layer 2 (Enrichment): Severity escalation from 3+ severe symptoms
- Layer 3 (Deep): KG-based severity lookup
- Each layer can escalate urgency but never lower it
- Failures default to HIGHER urgency, never lower
"""

import logging
import re
from typing import Dict, Tuple, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class UrgencyLevel(Enum):
    """ESI-like urgency levels."""
    EMERGENT = 1  # Immediate threat to life
    URGENT = 2    # Serious conditions requiring rapid evaluation
    SEMI_URGENT = 3  # Stable acute conditions
    NON_URGENT = 4  # Minor acute/chronic conditions
    SAFE = 5      # Routine/preventive care


# Layer 1: Conservative explicit keywords only
CRITICAL_KEYWORDS = {
    'chest pain': UrgencyLevel.EMERGENT,
    'difficulty breathing': UrgencyLevel.EMERGENT,
    'shortness of breath': UrgencyLevel.EMERGENT,
    'unconscious': UrgencyLevel.EMERGENT,
    'unresponsive': UrgencyLevel.EMERGENT,
    'heavy bleeding': UrgencyLevel.EMERGENT,
    'suicidal': UrgencyLevel.EMERGENT,
    'suicide': UrgencyLevel.EMERGENT,
    'self-harm': UrgencyLevel.EMERGENT,
    'choking': UrgencyLevel.EMERGENT,
    'can\'t breathe': UrgencyLevel.EMERGENT,
    'not breathing': UrgencyLevel.EMERGENT,
    'no pulse': UrgencyLevel.EMERGENT,
    'severe hemorrhage': UrgencyLevel.EMERGENT,
    'severe bleeding': UrgencyLevel.EMERGENT,
}

# Layer 2: Severe symptoms for escalation
SEVERE_SYMPTOMS = {
    'severe pain': 1,
    'loss of consciousness': 1,
    'disorientation': 1,
    'confusion': 1,
    'difficulty speaking': 1,
    'weakness': 0.5,
    'dizziness': 0.5,
    'loss of vision': 1,
    'high fever': 0.5,  # 39°C+
    'seizure': 1,
    'seizures': 1,
    'vomiting': 0.5,
    'severe nausea': 0.5,
}

# Symptom combinations that escalate
DANGER_COMBINATIONS = [
    (['chest pain', 'shortness of breath'], UrgencyLevel.EMERGENT),
    (['fever', 'confusion'], UrgencyLevel.URGENT),
    (['severe pain', 'high fever'], UrgencyLevel.URGENT),
    (['vomiting', 'severe abdominal pain'], UrgencyLevel.URGENT),
]


class EmergencyDetector:
    """Multi-layer emergency detection system."""
    
    @staticmethod
    def detect_emergency(user_message: str, kg=None) -> Tuple[UrgencyLevel, Dict]:
        """
        Multi-layer emergency detection.
        
        Returns:
            (UrgencyLevel, detection_details_dict)
        """
        
        message_lower = user_message.lower()
        details = {
            'keyword_match': None,
            'severity_escalation': None,
            'temporal_factors': None,
            'kg_severity': None,
            'final_urgency': UrgencyLevel.NON_URGENT,
            'reasoning': []
        }
        
        # Layer 1: Explicit keywords (floor - cannot be lowered)
        keyword_level = EmergencyDetector._layer_keywords(message_lower)
        details['keyword_match'] = keyword_level
        details['final_urgency'] = keyword_level
        
        if keyword_level == UrgencyLevel.EMERGENT:
            details['reasoning'].append("Critical keyword detected (Layer 1)")
            logger.warning(f"EMERGENCY detected via keyword: {user_message[:100]}")
            return keyword_level, details
        
        # Layer 2: Severity escalation (can only increase urgency)
        severity_level = EmergencyDetector._layer_severity_escalation(message_lower)
        details['severity_escalation'] = severity_level
        
        if severity_level.value < details['final_urgency'].value:  # Lower value = higher urgency
            details['final_urgency'] = severity_level
            details['reasoning'].append(f"Severity escalation: {severity_level.name} (Layer 2)")
            logger.warning(f"URGENT via severity escalation: {user_message[:100]}")
        
        # Layer 3: Temporal extraction (can escalate for rapid onset)
        temporal_level = EmergencyDetector._layer_temporal_factors(message_lower)
        details['temporal_factors'] = temporal_level
        
        if temporal_level and temporal_level.value < details['final_urgency'].value:
            details['final_urgency'] = temporal_level
            details['reasoning'].append(f"Temporal escalation: {temporal_level.name} (Layer 3)")
        
        # Layer 4: KG-based severity (deepest layer, can escalate)
        if kg:
            try:
                kg_level = EmergencyDetector._layer_kg_severity(message_lower, kg)
                details['kg_severity'] = kg_level
                
                if kg_level and kg_level.value < details['final_urgency'].value:
                    details['final_urgency'] = kg_level
                    details['reasoning'].append(f"KG-based severity: {kg_level.name} (Layer 4)")
            except Exception as e:
                logger.error(f"KG severity check failed: {e}")
                # Don't lower urgency on KG failure - default to higher
                if details['final_urgency'] == UrgencyLevel.NON_URGENT:
                    details['final_urgency'] = UrgencyLevel.SEMI_URGENT
                    details['reasoning'].append("KG check failed, defaulting to conservative urgency")
        
        if details['reasoning']:
            logger.info(f"Emergency detection result: {details['final_urgency'].name} - {details['reasoning']}")
        
        return details['final_urgency'], details
    
    @staticmethod
    def _layer_keywords(message: str) -> Optional[UrgencyLevel]:
        """Layer 1: Conservative explicit keywords only."""
        for keyword, level in CRITICAL_KEYWORDS.items():
            if keyword in message:
                logger.info(f"Keyword match: '{keyword}' -> {level.name}")
                return level
        return None
    
    @staticmethod
    def _layer_severity_escalation(message: str) -> Optional[UrgencyLevel]:
        """
        Layer 2: Escalate if 3+ severe symptoms detected.
        Never returns EMERGENT - only URGENT or SEMI_URGENT.
        """
        severe_count = 0
        matched_symptoms = []
        
        for symptom, weight in SEVERE_SYMPTOMS.items():
            if symptom in message:
                severe_count += weight
                matched_symptoms.append(symptom)
        
        # Check danger combinations first
        for symptom_list, level in DANGER_COMBINATIONS:
            if all(any(s in message for s in [symptom]) for s in symptom_list):
                logger.info(f"Danger combination detected: {symptom_list} -> {level.name}")
                return level
        
        # Escalate if 3+ severe symptoms
        if severe_count >= 3:
            logger.info(f"Severe symptom escalation: {severe_count} points from {matched_symptoms}")
            return UrgencyLevel.URGENT
        elif severe_count >= 1.5:
            logger.info(f"Moderate severity: {severe_count} points")
            return UrgencyLevel.SEMI_URGENT
        
        return None
    
    @staticmethod
    def _layer_temporal_factors(message: str) -> Optional[UrgencyLevel]:
        """
        Layer 3: Extract temporal patterns.
        Rapid onset + severe symptoms = escalate.
        """
        rapid_onset_keywords = ['sudden', 'acute', 'abrupt', 'just', 'minutes ago', 'came on quickly']
        
        has_rapid_onset = any(keyword in message for keyword in rapid_onset_keywords)
        
        if has_rapid_onset:
            # Check for severity indicators with rapid onset
            severe_indicators = ['severe', 'intense', 'unbearable', 'worst', 'can\'t']
            has_severe = any(word in message for word in severe_indicators)
            
            if has_severe:
                logger.info("Rapid onset + severe indicators detected")
                return UrgencyLevel.URGENT
        
        return None
    
    @staticmethod
    def _layer_kg_severity(message: str, kg) -> Optional[UrgencyLevel]:
        """
        Layer 4: Query KG for condition severity.
        This is optional and can fail gracefully.
        """
        try:
            # Try to extract candidate conditions from message
            # This would integrate with your KG
            logger.debug("KG severity check - not yet implemented")
            return None
        except Exception as e:
            logger.error(f"KG severity check error: {e}")
            return None
    
    @staticmethod
    def is_emergency(urgency: UrgencyLevel) -> bool:
        """Check if situation is emergency (level 1 or 2)."""
        return urgency.value <= 2
    
    @staticmethod
    def get_urgency_color(urgency: UrgencyLevel) -> str:
        """Get UI color for urgency level."""
        color_map = {
            UrgencyLevel.EMERGENT: 'red',
            UrgencyLevel.URGENT: 'orange',
            UrgencyLevel.SEMI_URGENT: 'yellow',
            UrgencyLevel.NON_URGENT: 'green',
            UrgencyLevel.SAFE: 'green'
        }
        return color_map.get(urgency, 'green')
