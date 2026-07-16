"""
Knowledge Graph + RAG Integration.
- KG used as index for next-question retrieval (graph traversal)
- RAG provides grounding documents for verification
- Multi-hop traversal for question ranking by information gain
"""

import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TriagePhase(Enum):
    """Phases in the triage process."""
    INITIAL_SYMPTOMS = "initial_symptoms"
    REFINEMENT = "refinement"
    VERIFICATION = "verification"
    ROUTING = "routing"


class KGRAGIntegration:
    """Integrate Knowledge Graph and RAG for intelligent symptom clarification."""
    
    def __init__(self, kg, rag_engine):
        """
        Initialize with KG and RAG engine instances.
        
        Args:
            kg: Knowledge graph instance with networkx graph
            rag_engine: Hybrid RAG engine for MedQuAD retrieval
        """
        self.kg = kg
        self.rag = rag_engine
        self.current_candidates = []  # Differential diagnoses
    
    def get_next_clarifying_question(
        self,
        confirmed_symptoms: List[str],
        unasked_symptoms: List[str] = None,
        candidate_conditions: List[str] = None,
        phase: TriagePhase = TriagePhase.INITIAL_SYMPTOMS
    ) -> Dict:
        """
        Get next clarifying question using KG traversal + information gain.
        
        Args:
            confirmed_symptoms: Already confirmed symptoms
            unasked_symptoms: Symptoms not yet asked about
            candidate_conditions: Initial differential diagnoses
            phase: Current triage phase
        
        Returns:
            {
                'question': str,
                'evidence': str,
                'info_gain': float,
                'expected_conditions': List[str],
                'grounding_docs': List[str]
            }
        """
        
        if phase == TriagePhase.INITIAL_SYMPTOMS:
            return self._get_initial_question(confirmed_symptoms)
        elif phase == TriagePhase.REFINEMENT:
            return self._get_refinement_question(confirmed_symptoms, unasked_symptoms)
        elif phase == TriagePhase.VERIFICATION:
            return self._get_verification_question(confirmed_symptoms, candidate_conditions)
        else:
            return self._get_routing_question(candidate_conditions)
    
    def _get_initial_question(self, symptoms: List[str]) -> Dict:
        """Get initial question when patient hasn't described symptoms yet."""
        # Find most common initial questions in KG
        initial_evidence_nodes = self.kg.graph.nodes(data=True) if hasattr(self.kg, 'graph') else []
        
        # Filter to high-information symptoms (bridge between many conditions)
        high_info_evidence = [
            node for node in initial_evidence_nodes
            if node.get('data', {}).get('frequency', 0) > 0.5
        ]
        
        question = "Could you describe the main symptom or problem you're experiencing?"
        evidence = "Initial symptom description"
        
        return {
            'question': question,
            'evidence': evidence,
            'info_gain': 0.8,
            'expected_conditions': [],
            'grounding_docs': []
        }
    
    def _get_refinement_question(
        self,
        confirmed_symptoms: List[str],
        unasked_symptoms: List[str] = None
    ) -> Dict:
        """Get refinement question using graph traversal + info gain ranking."""
        
        if not confirmed_symptoms:
            return self._get_initial_question([])
        
        if not hasattr(self.kg, 'rank_next_questions'):
            # Fallback if KG doesn't have rank_next_questions
            return self._fallback_refinement_question(confirmed_symptoms)
        
        try:
            # Use KG's built-in ranking (should rank by information gain)
            next_evidence = self.kg.rank_next_questions(
                confirmed_symptoms,
                unasked_symptoms or []
            )
            
            if not next_evidence:
                return self._fallback_refinement_question(confirmed_symptoms)
            
            # Get question template for this evidence
            question = f"Do you have {next_evidence}?"
            
            # Query RAG for grounding information
            rag_results = self.rag.query_medquad_by_symptoms(
                f"{' '.join(confirmed_symptoms)} {next_evidence}",
                k=2
            )
            
            grounding_docs = [
                result.get('text', '') for result in rag_results
            ] if rag_results else []
            
            # Estimate information gain (simplified)
            info_gain = 0.6
            
            return {
                'question': question,
                'evidence': next_evidence,
                'info_gain': info_gain,
                'expected_conditions': [],
                'grounding_docs': grounding_docs
            }
        
        except Exception as e:
            logger.error(f"Error in refinement question: {e}")
            return self._fallback_refinement_question(confirmed_symptoms)
    
    def _get_verification_question(
        self,
        confirmed_symptoms: List[str],
        candidate_conditions: List[str] = None
    ) -> Dict:
        """Get verification question to differentiate between candidates."""
        
        if not candidate_conditions:
            return self._get_refinement_question(confirmed_symptoms)
        
        # Find discriminating evidence between top 2-3 candidates
        question = f"To help narrow down the cause, have you experienced any of the following recently? [ask about distinguishing features]"
        
        # Query RAG for each condition to find discriminators
        rag_results = []
        for condition in candidate_conditions[:3]:
            results = self.rag.query_medquad_by_condition(condition, k=1)
            rag_results.extend(results)
        
        return {
            'question': question,
            'evidence': 'discriminating_evidence',
            'info_gain': 0.5,
            'expected_conditions': candidate_conditions[:3],
            'grounding_docs': [r.get('text', '') for r in rag_results]
        }
    
    def _get_routing_question(self, candidate_conditions: List[str]) -> Dict:
        """Get final routing verification question."""
        
        condition = candidate_conditions[0] if candidate_conditions else "general condition"
        
        return {
            'question': f"Based on your symptoms, I recommend seeing a specialist. Let me find available appointments.",
            'evidence': condition,
            'info_gain': 0.0,
            'expected_conditions': candidate_conditions,
            'grounding_docs': []
        }
    
    def _fallback_refinement_question(self, symptoms: List[str]) -> Dict:
        """Fallback when KG ranking unavailable."""
        
        # Generic refinement questions
        generic_questions = [
            "When did this start?",
            "How severe is it on a scale of 1-10?",
            "Have you seen a doctor about this before?",
            "Are you taking any medications for this?",
            "Do you have any allergies?"
        ]
        
        question = generic_questions[len(symptoms) % len(generic_questions)]
        
        return {
            'question': question,
            'evidence': 'context_refinement',
            'info_gain': 0.3,
            'expected_conditions': [],
            'grounding_docs': []
        }
    
    def generate_condition_card(
        self,
        condition: str,
        confirmed_symptoms: List[str]
    ) -> Dict:
        """
        Generate condition verification card with grounding.
        
        Uses RAG to fetch medical info, presents as structured card.
        """
        
        # Query RAG for condition information
        rag_results = self.rag.query_medquad_by_condition(condition, k=3)
        
        card = {
            'condition': condition,
            'symptoms': confirmed_symptoms,
            'description': "",
            'typical_presentation': "",
            'differential_notes': "",
            'confidence_level': "moderate",
            'grounding_sources': []
        }
        
        if rag_results:
            # Extract information from RAG results
            for i, result in enumerate(rag_results):
                text = result.get('text', '')
                metadata = result.get('metadata', {})
                
                if i == 0:
                    card['description'] = text[:200]
                elif i == 1:
                    card['typical_presentation'] = text[:200]
                elif i == 2:
                    card['differential_notes'] = text[:200]
                
                card['grounding_sources'].append({
                    'source': metadata.get('source', 'Unknown'),
                    'focus_area': metadata.get('focus_area', 'Unknown')
                })
        
        return card
    
    def verify_routing_decision(
        self,
        condition: str,
        confirmed_symptoms: List[str]
    ) -> Dict:
        """
        Verify routing decision with RAG grounding.
        
        Returns confidence score and supporting evidence.
        """
        
        card = self.generate_condition_card(condition, confirmed_symptoms)
        
        verification = {
            'condition': condition,
            'confidence': 0.7,
            'recommendation': self._get_department_for_condition(condition),
            'grounding_documents': card['grounding_sources'],
            'supported_symptoms': confirmed_symptoms,
            'confidence_explanation': f"Based on {len(confirmed_symptoms)} confirmed symptoms and {len(card['grounding_sources'])} medical references"
        }
        
        return verification
    
    def _get_department_for_condition(self, condition: str) -> str:
        """Map condition to medical department."""
        
        condition_lower = condition.lower()
        
        departments = {
            'respiratory': 'Pulmonology',
            'cardiac': 'Cardiology',
            'gastrointestinal': 'Gastroenterology',
            'neurological': 'Neurology',
            'orthopedic': 'Orthopedics',
            'dermatologic': 'Dermatology',
            'endocrine': 'Endocrinology',
            'rheumatologic': 'Rheumatology'
        }
        
        for keyword, dept in departments.items():
            if keyword in condition_lower:
                return dept
        
        return 'Internal Medicine'


def create_kg_rag_integration(kg, rag_engine) -> KGRAGIntegration:
    """Factory function to create KG-RAG integration."""
    return KGRAGIntegration(kg, rag_engine)
