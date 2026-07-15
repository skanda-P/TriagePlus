import json
import os
import networkx as nx
from functools import lru_cache

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/DDXPlus"))
CONDITIONS_FILE = os.path.join(DATA_DIR, "release_conditions.json")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.conditions = {}
        self.symptoms = set()
        self._load_data()

    def _load_data(self):
        try:
            with open(CONDITIONS_FILE, 'r', encoding='utf-8') as f:
                self.conditions = json.load(f)
                
            for condition_key, data in self.conditions.items():
                self.graph.add_node(condition_key, type="condition", severity=data.get("severity", 5))
                
                # Add symptoms as nodes and edges from condition -> symptom
                for symptom_key in data.get("symptoms", {}):
                    self.symptoms.add(symptom_key)
                    if not self.graph.has_node(symptom_key):
                        self.graph.add_node(symptom_key, type="symptom")
                    self.graph.add_edge(condition_key, symptom_key)
                    
                # Add antecedents as nodes and edges from condition -> antecedent
                for ant_key in data.get("antecedents", {}):
                    self.symptoms.add(ant_key)
                    if not self.graph.has_node(ant_key):
                        self.graph.add_node(ant_key, type="antecedent")
                    self.graph.add_edge(condition_key, ant_key)
        except Exception as e:
            print(f"Error loading DDXPlus conditions: {e}")

    def get_condition_severity(self, condition_name):
        # Default to 5 (least severe) if not found
        node_data = self.graph.nodes.get(condition_name, {})
        return node_data.get("severity", 5)
        
    def rank_next_questions(self, present_symptoms, asked_symptoms):
        """
        Rank the next best symptom to ask based on information gain to split remaining candidates.
        """
        # Find candidate conditions that have the present symptoms
        candidate_conditions = set()
        for symptom in present_symptoms:
            if self.graph.has_node(symptom):
                # Diseases that have this symptom (predecessors in DiGraph)
                candidate_conditions.update(self.graph.predecessors(symptom))
                
        if not candidate_conditions:
            candidate_conditions = set([n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'condition'])
            
        # Count frequency of unasked symptoms among candidate conditions
        symptom_counts = {}
        for condition in candidate_conditions:
            for neighbor in self.graph.successors(condition):
                if neighbor not in present_symptoms and neighbor not in asked_symptoms:
                    symptom_counts[neighbor] = symptom_counts.get(neighbor, 0) + 1
                    
        if not symptom_counts:
            return None
            
        # Sort by frequency (closest to splitting the candidate set in half gives most info gain)
        # Ideal split is frequency == len(candidate_conditions) / 2
        target_freq = len(candidate_conditions) / 2.0
        
        ranked_symptoms = sorted(
            symptom_counts.items(), 
            key=lambda x: abs(x[1] - target_freq)
        )
        
        return ranked_symptoms[0][0] if ranked_symptoms else None

@lru_cache(maxsize=1)
def get_kg() -> KnowledgeGraph:
    return KnowledgeGraph()
