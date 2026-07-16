#!/usr/bin/env python3
"""
DDXPlus Knowledge Graph Builder
Converts DDXPlus dataset into NetworkX graph for efficient traversal
Computes information gain for intelligent question ranking
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDXPlusKGBuilder:
    """Build knowledge graph from DDXPlus dataset"""
    
    def __init__(self, data_dir: str = "backend/data"):
        self.data_dir = data_dir
        self.conditions_file = os.path.join(data_dir, "ddxplus_conditions.json")
        self.evidences_file = os.path.join(data_dir, "ddxplus_evidences.json")
        self.eval_set_file = os.path.join(data_dir, "ddxplus_eval_set.json")
        
        self.conditions = {}
        self.evidences = {}
        self.eval_cases = []
        self.graph = nx.DiGraph()
        
        self.evidence_condition_counts = defaultdict(Counter)  # P(condition | evidence)
        self.condition_evidence_counts = defaultdict(Counter)  # P(evidence | condition)
        
    def load_data(self):
        """Load DDXPlus JSON files"""
        logger.info("Loading DDXPlus data...")
        
        with open(self.conditions_file, 'r') as f:
            self.conditions = json.load(f)
        logger.info(f"Loaded {len(self.conditions)} conditions")
        
        with open(self.evidences_file, 'r') as f:
            self.evidences = json.load(f)
        logger.info(f"Loaded {len(self.evidences)} evidences")
        
        with open(self.eval_set_file, 'r') as f:
            self.eval_cases = json.load(f)
        logger.info(f"Loaded {len(self.eval_cases)} evaluation cases")
    
    def build_graph(self):
        """Build NetworkX directed graph with nodes and edges"""
        logger.info("Building knowledge graph...")
        
        # Add condition nodes
        for cond_id, cond_data in self.conditions.items():
            self.graph.add_node(
                cond_id,
                node_type='condition',
                name=cond_data.get('name', ''),
                prevalence=cond_data.get('prevalence', 0)
            )
        
        # Add evidence nodes
        for ev_id, ev_data in self.evidences.items():
            self.graph.add_node(
                ev_id,
                node_type='evidence',
                name=ev_data.get('name', ''),
                icd_code=ev_data.get('icd_code', '')
            )
        
        # Build edges: condition -> evidence relationships
        logger.info("Adding edges (condition -> evidence relationships)...")
        edge_count = 0
        
        for case in self.eval_cases:
            condition_id = str(case['condition_id'])
            present_evidences = case['present_evidences']
            absent_evidences = case['absent_evidences']
            
            # Present evidences: strong connection
            for ev_id in present_evidences:
                ev_id_str = str(ev_id)
                self.graph.add_edge(
                    condition_id, 
                    ev_id_str,
                    weight=1.0,  # Present
                    type='present'
                )
                self.evidence_condition_counts[ev_id_str][condition_id] += 1
                self.condition_evidence_counts[condition_id][ev_id_str] += 1
                edge_count += 1
            
            # Absent evidences: weak negative connection (for discriminating)
            for ev_id in absent_evidences:
                ev_id_str = str(ev_id)
                if not self.graph.has_edge(condition_id, ev_id_str):
                    self.graph.add_edge(
                        condition_id,
                        ev_id_str,
                        weight=0.0,  # Absent
                        type='absent'
                    )
                    edge_count += 1
        
        logger.info(f"Added {edge_count} edges")
    
    def compute_information_gain(self, evidence_id: str, current_evidences: Set[str] = None) -> float:
        """
        Compute information gain of asking about a specific evidence
        Higher IG = more discriminative evidence
        """
        if current_evidences is None:
            current_evidences = set()
        
        if evidence_id not in self.evidence_condition_counts:
            return 0.0
        
        condition_counts = self.evidence_condition_counts[evidence_id]
        if not condition_counts:
            return 0.0
        
        # Probability of each condition given this evidence
        total = sum(condition_counts.values())
        probs = np.array([count / total for count in condition_counts.values()])
        
        # Shannon entropy (information gain metric)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    def rank_next_questions(self, current_symptoms: List[str], asked_symptoms: List[str] = None) -> List[Tuple[str, float]]:
        """
        Rank next questions by information gain
        Returns list of (evidence_id, score) tuples
        """
        if asked_symptoms is None:
            asked_symptoms = []
        
        asked_set = set(asked_symptoms)
        current_set = set(current_symptoms)
        
        # Find candidate next evidences
        candidates = {}
        
        # Get all conditions compatible with current symptoms
        if current_symptoms:
            compatible_conditions = set()
            for symptom in current_symptoms:
                if symptom in self.evidences:
                    # Find conditions with this evidence
                    for condition_id in self.evidence_condition_counts.get(symptom, {}).keys():
                        if symptom not in asked_set:
                            compatible_conditions.add(condition_id)
        else:
            compatible_conditions = set(self.conditions.keys())
        
        # Find most informative unanswered evidence for remaining conditions
        for condition_id in compatible_conditions:
            outgoing_edges = self.graph.out_edges(condition_id, data=True)
            for _, evidence_id, edge_data in outgoing_edges:
                if evidence_id not in asked_set and evidence_id not in current_set:
                    if evidence_id not in candidates:
                        candidates[evidence_id] = self.compute_information_gain(evidence_id, current_set)
        
        # Sort by information gain
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:10]  # Return top 10
    
    def get_condition_evidence_profile(self, condition_id: str) -> Dict:
        """Get all evidences associated with a condition"""
        evidences = []
        for _, evidence_id, edge_data in self.graph.out_edges(condition_id, data=True):
            evidences.append({
                'evidence_id': evidence_id,
                'evidence_name': self.evidences.get(evidence_id, {}).get('name', ''),
                'presence': edge_data.get('type', 'unknown')
            })
        return {
            'condition_id': condition_id,
            'condition_name': self.conditions.get(condition_id, {}).get('name', ''),
            'evidences': evidences
        }
    
    def save_graph(self, output_path: str = "backend/data/ddxplus_kg.pkl"):
        """Save graph and metadata to pickle"""
        logger.info(f"Saving knowledge graph to {output_path}")
        
        kg_data = {
            'graph': self.graph,
            'conditions': self.conditions,
            'evidences': self.evidences,
            'evidence_condition_counts': self.evidence_condition_counts,
            'condition_evidence_counts': self.condition_evidence_counts,
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(kg_data, f)
        
        logger.info(f"Graph saved: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def build(self):
        """Execute full pipeline"""
        self.load_data()
        self.build_graph()
        self.save_graph()
        logger.info("Knowledge graph built successfully!")

if __name__ == "__main__":
    builder = DDXPlusKGBuilder()
    builder.build()
