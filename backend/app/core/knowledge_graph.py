import json
import os
import networkx as nx
from typing import List, Dict, Any, Tuple

class KnowledgeGraph:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        self.conditions = {}
        self.evidences = {}
        self._load_data()
        self._build_graph()

    def _load_data(self):
        conditions_path = os.path.join(self.data_dir, "release_conditions.json")
        evidences_path = os.path.join(self.data_dir, "release_evidences.json")
        
        with open(conditions_path, "r", encoding="utf-8") as f:
            self.conditions = json.load(f)
            
        with open(evidences_path, "r", encoding="utf-8") as f:
            self.evidences = json.load(f)

    def _build_graph(self):
        # Nodes for Evidences
        for ev_id, ev_data in self.evidences.items():
            self.graph.add_node(ev_id, type="evidence", data=ev_data)
            
        # Nodes for Conditions and Edges from Condition to Evidence
        for cond_name, cond_data in self.conditions.items():
            cond_id = f"C_{cond_name}"
            self.graph.add_node(cond_id, type="condition", data=cond_data)
            
            for symp_id in cond_data.get("symptoms", {}):
                if symp_id in self.evidences:
                    self.graph.add_edge(cond_id, symp_id, relation="has_symptom")
                    
            for ant_id in cond_data.get("antecedents", {}):
                if ant_id in self.evidences:
                    self.graph.add_edge(cond_id, ant_id, relation="has_antecedent")

    def get_question_for_evidence(self, ev_id: str) -> str:
        ev_node = self.graph.nodes.get(ev_id)
        if not ev_node or ev_node["type"] != "evidence":
            return ""
        return ev_node["data"].get("question_en", f"Do you have {ev_id}?")

    def get_evidence_name(self, ev_id: str) -> str:
        ev_node = self.graph.nodes.get(ev_id)
        if not ev_node or ev_node["type"] != "evidence":
            return ev_id
        return ev_node["data"].get("name", ev_id)

    def rank_next_questions(self, present_symptoms: List[str], absent_symptoms: List[str], top_k: int = 1) -> List[str]:
        """
        Information-gain inspired greedy selection:
        1. Find all conditions that have at least one of the present_symptoms.
        2. Rank those conditions by how many present_symptoms match.
        3. For the top conditions, find symptoms they have that haven't been asked yet.
        4. Return the most frequent unasked symptom among those top conditions.
        """
        if not present_symptoms:
            # If no symptoms yet, just ask a common general symptom or return empty
            return []
            
        condition_scores = {}
        for cond_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") == "condition":
                # count matches
                cond_evidences = [v for u, v in self.graph.out_edges(cond_id)]
                matches = sum(1 for s in present_symptoms if s in cond_evidences)
                if matches > 0:
                    condition_scores[cond_id] = matches
                    
        # Sort conditions by match count descending
        sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
        
        # We only care about the top N conditions to narrow down
        top_conditions = [c[0] for c in sorted_conditions[:5]]
        
        # Count frequency of unasked symptoms in these top conditions
        asked_symptoms = set(present_symptoms + absent_symptoms)
        unasked_counts = {}
        
        for cond_id in top_conditions:
            for _, ev_id in self.graph.out_edges(cond_id):
                if ev_id not in asked_symptoms:
                    unasked_counts[ev_id] = unasked_counts.get(ev_id, 0) + 1
                    
        # Return top K unasked symptoms
        sorted_unasked = sorted(unasked_counts.items(), key=lambda x: x[1], reverse=True)
        return [ev for ev, count in sorted_unasked[:top_k]]
