import json
import os
import pickle
import numpy as np
import networkx as nx
from functools import lru_cache
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
KG_FILE = os.path.join(DATA_DIR, "ddxplus_kg.pkl")
CONDITIONS_FILE = os.path.join(DATA_DIR, "ddxplus_conditions.json")
EVIDENCES_FILE = os.path.join(DATA_DIR, "ddxplus_evidences.json")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.conditions = {}
        self.evidences = {}
        self.evidence_condition_counts = defaultdict(Counter)
        self.condition_evidence_counts = defaultdict(Counter)
        self._loaded = False
        self._load_data()

    def _load_data(self):
        """Load pre-built KG or fall back to JSON"""
        try:
            # Try loading pre-built pickled KG
            if os.path.exists(KG_FILE):
                logger.info(f"Loading KG from {KG_FILE}")
                with open(KG_FILE, 'rb') as f:
                    kg_data = pickle.load(f)
                    self.graph = kg_data['graph']
                    self.conditions = kg_data['conditions']
                    self.evidences = kg_data['evidences']
                    self.evidence_condition_counts = kg_data['evidence_condition_counts']
                    self.condition_evidence_counts = kg_data['condition_evidence_counts']
                    self._loaded = True
                    logger.info(f"Loaded KG: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            else:
                logger.warning(f"KG not found at {KG_FILE}, loading from JSON")
                self._load_from_json()
        except Exception as e:
            logger.error(f"Error loading KG: {e}")
            self._load_from_json()

    def _load_from_json(self):
        """Fallback: Load from JSON files"""
        try:
            if os.path.exists(CONDITIONS_FILE):
                with open(CONDITIONS_FILE, 'r') as f:
                    self.conditions = json.load(f)
                logger.info(f"Loaded {len(self.conditions)} conditions from JSON")
            
            if os.path.exists(EVIDENCES_FILE):
                with open(EVIDENCES_FILE, 'r') as f:
                    self.evidences = json.load(f)
                logger.info(f"Loaded {len(self.evidences)} evidences from JSON")
            
            # Build basic graph from JSON
            for cond_id, cond_data in self.conditions.items():
                self.graph.add_node(cond_id, type="condition", name=cond_data.get("name", ""))
            
            for ev_id, ev_data in self.evidences.items():
                self.graph.add_node(ev_id, type="evidence", name=ev_data.get("name", ""))
            
            self._loaded = True
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
            self._loaded = False

    def get_condition_info(self, condition_id: str) -> Dict:
        """Get condition metadata"""
        return self.conditions.get(str(condition_id), {})

    def get_evidence_info(self, evidence_id: str) -> Dict:
        """Get evidence metadata"""
        return self.evidences.get(str(evidence_id), {})

    def rank_next_questions(self, present_symptoms: List[str], asked_symptoms: List[str] = None) -> List[Tuple[str, float]]:
        """
        Rank next questions by information gain using KG traversal
        Returns top 10 evidences sorted by discriminative power
        """
        if not self._loaded:
            logger.warning("KG not loaded, returning empty results")
            return []
        
        if asked_symptoms is None:
            asked_symptoms = []
        
        asked_set = set(asked_symptoms)
        present_set = set(present_symptoms)
        
        # Find compatible conditions based on present symptoms
        compatible_conditions = set(self.conditions.keys())
        
        if present_symptoms:
            compatible_conditions = set()
            for symptom in present_symptoms:
                # Find conditions with this evidence
                if symptom in self.evidence_condition_counts:
                    compatible_conditions.update(self.evidence_condition_counts[symptom].keys())
        
        if not compatible_conditions:
            compatible_conditions = set(self.conditions.keys())
        
        # Score unanswered evidences by information gain
        evidence_scores = defaultdict(float)
        
        for condition_id in compatible_conditions:
            # Get evidences for this condition
            outgoing_edges = list(self.graph.out_edges(condition_id, data=True))
            for _, evidence_id, _ in outgoing_edges:
                if evidence_id not in asked_set and evidence_id not in present_set:
                    # Information gain = how well this evidence discriminates
                    if evidence_id in self.evidence_condition_counts:
                        condition_dist = self.evidence_condition_counts[evidence_id]
                        total = sum(condition_dist.values())
                        if total > 0:
                            # Entropy (higher entropy = more discriminative)
                            probs = np.array([count / total for count in condition_dist.values()])
                            entropy = -np.sum(probs * np.log2(probs + 1e-10))
                            evidence_scores[evidence_id] += entropy / len(compatible_conditions)
        
        # Sort by score
        ranked = sorted(evidence_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:10]  # Return top 10

    def get_condition_specialty(self, condition_id: str) -> str:
        """Map condition to medical specialty"""
        specialty_mapping = {
            'cardio': 'Cardiology',
            'pulmon': 'Pulmonology',
            'gastro': 'Gastroenterology',
            'neuro': 'Neurology',
            'derm': 'Dermatology',
            'ortho': 'Orthopedics',
            'rheum': 'Rheumatology',
            'endo': 'Endocrinology',
            'nephro': 'Nephrology',
            'hemo': 'Hematology',
        }
        
        cond_name = self.get_condition_info(condition_id).get("name", "").lower()
        for key, specialty in specialty_mapping.items():
            if key in cond_name:
                return specialty
        
        return "General Medicine"

@lru_cache(maxsize=1)
def get_kg() -> KnowledgeGraph:
    """Get singleton KG instance"""
    return KnowledgeGraph()
