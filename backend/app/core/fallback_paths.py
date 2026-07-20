"""
Hardcoded fallback conversation paths when Ollama (LLM) is unavailable.
Designed to fill all required slots sequentially: symptoms → severity → duration → medications → final diagnosis.
"""

import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackSlot(Enum):
    """Slots that need to be filled in fallback mode."""
    SYMPTOMS = "symptoms"
    SEVERITY = "severity"
    DURATION = "duration"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    COMORBIDITIES = "comorbidities"
    FINAL_DIAGNOSIS = "final_diagnosis"


class FallbackConversationManager:
    """Manages hardcoded conversation flow when LLM unavailable.
    
    Instantiated per-session (not a global singleton) to avoid cross-session
    state pollution.
    """
    
    # Templated questions that work for ANY symptom/condition
    UNIVERSAL_QUESTIONS = {
        FallbackSlot.SYMPTOMS: [
            "What symptoms are you experiencing? Please describe them.",
            "Can you tell me more about what's bothering you?",
        ],
        FallbackSlot.SEVERITY: [
            "On a scale of 1-10, how severe is your condition? (1=mild, 10=severe)",
            "How much does this affect your daily activities?",
        ],
        FallbackSlot.DURATION: [
            "How long have you been experiencing these symptoms?",
            "When did this start?",
        ],
        FallbackSlot.MEDICATIONS: [
            "Are you currently taking any medications?",
            "What medications do you take regularly?",
        ],
        FallbackSlot.ALLERGIES: [
            "Do you have any known allergies to medications?",
            "Are you allergic to anything?",
        ],
        FallbackSlot.COMORBIDITIES: [
            "Do you have any chronic conditions like diabetes, high blood pressure, or heart disease?",
            "Do you have any pre-existing health conditions?",
        ],
    }
    
    # Generic follow-up questions to maintain conversation flow
    FOLLOW_UP_QUESTIONS = [
        "Anything else you'd like to tell me?",
        "Have you seen a doctor before for this?",
        "Is this your first time experiencing this?",
        "Have you tried any treatment for this?",
    ]
    
    # Generic explanations when diagnosis is made
    GENERIC_EXPLANATIONS = {
        "respiratory": "Based on your symptoms, this could be a respiratory condition. This is an area where respiratory specialists can provide the best care.",
        "gastrointestinal": "Your symptoms suggest a gastrointestinal issue. A gastroenterologist can evaluate this properly.",
        "cardiac": "Your symptoms could be cardiac-related. This requires urgent evaluation by a cardiologist.",
        "general": "Your symptoms suggest a medical condition that needs evaluation. A general practitioner can help determine the exact cause.",
        "dermatological": "Based on your description, this appears to be a skin-related issue. A dermatologist can provide specialized care.",
        "neurological": "Your symptoms could indicate a neurological condition. A neurologist can perform the necessary evaluations.",
        "musculoskeletal": "This sounds like a musculoskeletal issue. An orthopedist or physical medicine specialist can help.",
        "infectious": "Your symptoms suggest a possible infection. A primary care doctor can evaluate and treat this.",
    }
    
    def __init__(self):
        self.slot_order = [
            FallbackSlot.SYMPTOMS,
            FallbackSlot.SEVERITY,
            FallbackSlot.DURATION,
            FallbackSlot.MEDICATIONS,
            FallbackSlot.ALLERGIES,
            FallbackSlot.COMORBIDITIES,
        ]
        self.current_slot_index = 0
        self.filled_slots = {}
    
    def get_next_question(self) -> str:
        """Get the next question in the fallback sequence."""
        if self.current_slot_index >= len(self.slot_order):
            # All slots filled
            return self._generate_final_response()
        
        current_slot = self.slot_order[self.current_slot_index]
        
        # Get appropriate question
        if current_slot in self.UNIVERSAL_QUESTIONS:
            questions = self.UNIVERSAL_QUESTIONS[current_slot]
            # Cycle through questions for variety
            question_idx = self.current_slot_index % len(questions)
            return questions[question_idx]
        
        return "Can you tell me more?"
    
    def fill_slot(self, slot: FallbackSlot, value: str) -> None:
        """Fill a slot with user response."""
        self.filled_slots[slot] = value
        logger.debug(f"Filled slot {slot.value}: {value[:100]}")
    
    def process_user_response(self, response: str) -> Dict:
        """
        Process user response and advance to next slot.
        
        Returns:
            {
                'filled_slot': FallbackSlot,
                'next_question': str,
                'all_slots_filled': bool,
                'can_diagnose': bool
            }
        """
        if self.current_slot_index >= len(self.slot_order):
            return {
                'filled_slot': None,
                'next_question': None,
                'all_slots_filled': True,
                'can_diagnose': True
            }
        
        current_slot = self.slot_order[self.current_slot_index]
        self.fill_slot(current_slot, response)
        
        self.current_slot_index += 1
        
        # Check if all slots are filled
        all_filled = self.current_slot_index >= len(self.slot_order)
        
        if all_filled:
            next_question = None
        else:
            next_question = self.get_next_question()
        
        return {
            'filled_slot': current_slot,
            'next_question': next_question,
            'all_slots_filled': all_filled,
            'can_diagnose': all_filled
        }
    
    def generate_diagnosis(self, symptoms: Optional[str] = None) -> Dict:
        """
        Generate generic diagnosis based on symptoms.
        
        Works by:
        1. Categorizing symptoms into broad categories
        2. Routing to appropriate specialty
        3. Generating generic explanation
        """
        if not self.filled_slots.get(FallbackSlot.SYMPTOMS):
            symptoms = symptoms or "general discomfort"
        else:
            symptoms = self.filled_slots[FallbackSlot.SYMPTOMS]
        
        # Symptom categorization (simple keyword matching)
        category = self._categorize_symptoms(symptoms)
        
        explanation = self.GENERIC_EXPLANATIONS.get(category, self.GENERIC_EXPLANATIONS["general"])
        
        diagnosis = {
            'category': category,
            'explanation': explanation,
            'severity': self.filled_slots.get(FallbackSlot.SEVERITY, 'moderate'),
            'duration': self.filled_slots.get(FallbackSlot.DURATION, 'unknown'),
            'recommended_department': self._map_category_to_department(category),
            'triage_level': self._estimate_triage_level(),
            'is_fallback': True
        }
        
        logger.info(f"Generated fallback diagnosis: {diagnosis}")
        return diagnosis
    
    def _categorize_symptoms(self, symptoms: Optional[str]) -> str:
        """Categorize symptoms using simple keyword matching."""
        if not symptoms:
            return 'general'
        symptoms_lower = symptoms.lower()
        
        # Respiratory
        if any(word in symptoms_lower for word in ['cough', 'breath', 'cold', 'flu', 'asthma', 'wheeze']):
            return 'respiratory'
        
        # Gastrointestinal
        if any(word in symptoms_lower for word in ['stomach', 'nausea', 'vomit', 'diarrhea', 'constipation', 'gas', 'bloating']):
            return 'gastrointestinal'
        
        # Cardiac
        if any(word in symptoms_lower for word in ['heart', 'chest', 'palpitation', 'arrhythmia']):
            return 'cardiac'
        
        # Dermatological
        if any(word in symptoms_lower for word in ['rash', 'skin', 'itching', 'hives', 'acne', 'eczema']):
            return 'dermatological'
        
        # Neurological
        if any(word in symptoms_lower for word in ['headache', 'migraine', 'dizzy', 'vertigo', 'seizure', 'neuropathy']):
            return 'neurological'
        
        # Musculoskeletal
        if any(word in symptoms_lower for word in ['pain', 'joint', 'muscle', 'arthritis', 'fracture', 'sprain']):
            return 'musculoskeletal'
        
        # Infectious
        if any(word in symptoms_lower for word in ['fever', 'infection', 'sepsis', 'malaria', 'covid']):
            return 'infectious'
        
        return 'general'
    
    def _map_category_to_department(self, category: str) -> str:
        """Map symptom category to medical department."""
        mapping = {
            'respiratory': 'Respiratory',
            'gastrointestinal': 'Gastroenterology',
            'cardiac': 'Cardiology',
            'dermatological': 'Dermatology',
            'neurological': 'Neurology',
            'musculoskeletal': 'Orthopedics',
            'infectious': 'General Medicine / Internal Medicine',
            'general': 'General Medicine / Internal Medicine'
        }
        return mapping.get(category, 'General Medicine / Internal Medicine')
    
    def _estimate_triage_level(self) -> int:
        """
        Estimate ESI triage level (1=emergent, 5=safe).
        Based on severity slot.
        """
        severity_str = self.filled_slots.get(FallbackSlot.SEVERITY, '5').lower()
        
        try:
            severity_num = int(severity_str.split()[0])  # Extract first number
            if severity_num >= 8:
                return 2  # URGENT
            elif severity_num >= 5:
                return 3  # SEMI_URGENT
            else:
                return 4  # NON_URGENT
        except Exception:
            return 3  # Default to semi-urgent
    
    def reset(self):
        """Reset conversation state for new patient."""
        self.current_slot_index = 0
        self.filled_slots = {}
        logger.debug("Fallback conversation state reset")
    
    def _generate_final_response(self) -> str:
        """Generate final response when all slots are filled."""
        return "Thank you for providing all that information. Based on your symptoms and history, let me recommend an appropriate specialist for you."
    
    def get_conversation_status(self) -> Dict:
        """Get current conversation status."""
        return {
            'current_slot': self.slot_order[self.current_slot_index].value if self.current_slot_index < len(self.slot_order) else None,
            'slots_filled': len(self.filled_slots),
            'total_slots': len(self.slot_order),
            'progress_percent': (len(self.filled_slots) / len(self.slot_order)) * 100,
            'filled_slots': {k.value: v for k, v in self.filled_slots.items()}
        }


# REMOVED: Global singleton `_fallback_manager` and `get_fallback_manager()`.
# The fallback manager is now instantiated per-session by the graph nodes
# that need it, avoiding cross-session state pollution.