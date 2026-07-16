"""
Multi-intent router for TriagePlus LangGraph

Maps 6 user intents to dedicated workflow paths:
1. symptom_triage: "I have chest pain" → Full clinical triage flow
2. direct_booking_department: "I want to book with cardiology" → Select doctor from dept
3. direct_booking_doctor: "I want to book with Dr. Smith" → Direct doctor search
4. appointment_status: "What's my appointment status?" → Query existing appointments
5. followup_appointment: "I need a follow-up" → Schedule follow-up based on previous visit
6. generic_inquiry: "What is hypertension?" → General medical Q&A
"""

import re
from typing import Optional, Dict, Any
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)

# Intent keywords grouped by priority
INTENT_KEYWORDS = {
    "direct_booking_doctor": {
        "keywords": ["book", "appointment", "schedule", "reserve", "with", "doctor", "dr."],
        "patterns": [
            r"book.*with.*dr\.?\s*(\w+)",
            r"appointment.*with.*dr\.?\s*(\w+)",
            r"i\s+want\s+(an\s+)?appointment.*dr\.?\s*(\w+)",
            r"schedule.*dr\.?\s*(\w+)",
        ],
        "priority": 1
    },
    "direct_booking_department": {
        "keywords": ["book", "appointment", "schedule", "reserve", "department", "cardiology", "dermatology", "orthopedic", "specialist"],
        "patterns": [
            r"book.*(?:with\s+)?(\w+)\s+(department|specialist)",
            r"i\s+(?:want|need).*(?:appointment|consultation).*(?:with\s+)?(\w+)",
            r"schedule.*(\w+)\s+(?:department|specialist)",
        ],
        "priority": 2
    },
    "appointment_status": {
        "keywords": ["status", "appointment", "when", "scheduled", "confirm", "check", "what's my"],
        "patterns": [
            r"(?:what's|what\s+is).*(?:my\s+)?appointment",
            r"appointment\s+status",
            r"when\s+is\s+my\s+appointment",
            r"confirm.*appointment",
        ],
        "priority": 3
    },
    "followup_appointment": {
        "keywords": ["follow", "follow-up", "followup", "next appointment", "reschedule", "another appointment"],
        "patterns": [
            r"follow[\s-]?up\s+appointment",
            r"schedule.*follow[\s-]?up",
            r"(?:i\s+)?need.*another\s+appointment",
            r"next\s+appointment",
        ],
        "priority": 4
    },
    "generic_inquiry": {
        "keywords": ["what", "how", "why", "explain", "tell", "information", "about"],
        "patterns": [
            r"what\s+is\s+(\w+)",
            r"(?:how|why|explain).*(\w+)",
            r"tell\s+me.*(\w+)",
            r"information\s+about\s+(\w+)",
        ],
        "priority": 5
    },
    "symptom_triage": {
        # This is the default catch-all for medical symptoms
        "keywords": ["pain", "ache", "symptom", "sick", "ill", "fever", "cough", "hurt"],
        "patterns": [],
        "priority": 6  # Lowest priority, default fallback
    }
}

class MultiIntentRouter:
    """Routes user messages to appropriate LangGraph intent paths"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.doctor_names = self._load_doctor_names()
    
    def _load_doctor_names(self) -> set:
        """Load doctor names from database for matching"""
        # TODO: Fetch from Supabase doctors table
        # For now, return empty set - will be populated at runtime
        return set()
    
    def detect_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Detect user intent from message
        
        Returns:
            {
                "intent": "direct_booking_doctor",
                "confidence": 0.95,
                "extracted_entity": "Smith",  # doctor name, dept, condition, etc.
                "reasoning": "Pattern match: 'book with Dr. Smith'"
            }
        """
        user_message = user_message.lower().strip()
        
        # Check for emergency first (emergency check happens before intent routing in LangGraph)
        # This is just for completeness
        
        scores = {}
        
        # Score each intent
        for intent, config in INTENT_KEYWORDS.items():
            score = 0
            entity = None
            
            # Pattern matching (highest priority)
            for pattern in config["patterns"]:
                match = re.search(pattern, user_message)
                if match:
                    score = max(score, 0.95)
                    if match.groups():
                        entity = match.group(1)
                    break
            
            # Keyword matching (lower priority)
            if score == 0:
                keyword_count = sum(1 for kw in config["keywords"] if kw in user_message)
                if keyword_count > 0:
                    score = min(0.5 + (keyword_count * 0.1), 0.8)
            
            # For direct_booking_doctor, try to match against known doctors
            if intent == "direct_booking_doctor" and score > 0:
                doctor_match = self._match_doctor_name(user_message)
                if doctor_match:
                    score = 0.98
                    entity = doctor_match
            
            scores[intent] = {
                "score": score,
                "priority": config["priority"],
                "entity": entity
            }
        
        # Select best intent
        best_intent = max(scores.items(), 
                         key=lambda x: (x[1]["score"], -x[1]["priority"]))
        
        intent_name, intent_data = best_intent
        confidence = intent_data["score"]
        
        # If confidence is too low, default to symptom_triage
        if confidence < self.confidence_threshold:
            intent_name = "symptom_triage"
            confidence = 0.5
            entity = None
            reasoning = "Low confidence match, defaulting to symptom triage"
        else:
            reasoning = self._get_reasoning(intent_name, user_message, entity)
        
        return {
            "intent": intent_name,
            "confidence": confidence,
            "extracted_entity": entity,
            "reasoning": reasoning,
            "all_scores": {k: v["score"] for k, v in scores.items()}
        }
    
    def _match_doctor_name(self, text: str) -> Optional[str]:
        """
        Try to match doctor names in text using fuzzy matching
        
        TODO: Replace with database query
        """
        # Placeholder - in production this would query the doctors table
        doctors = ["smith", "johnson", "patel", "kumar", "sharma"]
        
        for doctor in doctors:
            if fuzz.ratio(doctor, text) > 85:
                return doctor
        
        return None
    
    def _get_reasoning(self, intent: str, message: str, entity: Optional[str]) -> str:
        """Generate human-readable reasoning for intent detection"""
        
        reasoning_map = {
            "direct_booking_doctor": f"User wants to book with a specific doctor (Dr. {entity})",
            "direct_booking_department": f"User wants to book with {entity} department",
            "appointment_status": "User asking about existing appointment status",
            "followup_appointment": "User needs a follow-up appointment",
            "generic_inquiry": f"User asking general medical question about {entity}",
            "symptom_triage": "User describing medical symptoms for triage",
        }
        
        return reasoning_map.get(intent, "Intent detected")

# Function to be called from triage graph
def detect_user_intent(user_message: str) -> Dict[str, Any]:
    """Wrapper function for LangGraph integration"""
    router = MultiIntentRouter(confidence_threshold=0.65)
    return router.detect_intent(user_message)
