#!/usr/bin/env python3
"""
DDXPlus Knowledge Graph Builder
Converts DDXPlus dataset into NetworkX graph for efficient traversal
Computes information gain for intelligent question ranking
"""

import json
import pickle
import logging
import ast
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional

import numpy as np
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _base_evidence(evid: str) -> str:
    """Strip the value-suffix from an evidence code: 'E_204_@_V_10' -> 'E_204'.

    The runtime NER/XGBoost pipelines emit *base* codes (without the value
    suffix), so the KG must store base codes too — otherwise
    `evidence_condition_counts[E_55]` lookups miss because the only keys
    present are `E_55_@_V_89`, `E_55_@_V_90`, etc.
    """
    return evid.split("_@_")[0] if "_@_" in evid else evid


class DDXPlusKGBuilder:
    """Build knowledge graph from DDXPlus dataset"""

    def __init__(self, data_dir: str = "backend/data"):
        backend_dir = Path(__file__).parent.parent
        self.data_dir = backend_dir / data_dir
        self.conditions_file = self.data_dir / "DDXPlus" / "release_conditions.json"
        self.evidences_file = self.data_dir / "DDXPlus" / "release_evidences.json"
        self.eval_set_file = self.data_dir / "DDXPlus" / "eval_set.json"

        self.conditions: Dict[str, Dict] = {}
        self.evidences: Dict[str, Dict] = {}
        self.eval_cases: List[Dict] = []
        self.graph = nx.DiGraph()

        # present counts
        self.evidence_condition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.condition_evidence_counts: Dict[str, Counter] = defaultdict(Counter)
        # NEW: absent counts (needed for proper expected-posterior-entropy IG)
        self.evidence_condition_absent_counts: Dict[str, Counter] = defaultdict(Counter)
        # total cases per pathology, needed to compute P(E|C).
        self.condition_case_counts: Counter = Counter()

    def load_data(self):
        """Load DDXPlus JSON files"""
        logger.info("Loading DDXPlus data...")

        with open(self.conditions_file, "r", encoding="utf-8") as f:
            self.conditions = json.load(f)
        logger.info(f"Loaded {len(self.conditions)} conditions")

        with open(self.evidences_file, "r", encoding="utf-8") as f:
            self.evidences = json.load(f)
        logger.info(f"Loaded {len(self.evidences)} evidences")

        with open(self.eval_set_file, "r", encoding="utf-8") as f:
            self.eval_cases = json.load(f)
        logger.info(f"Loaded {len(self.eval_cases)} evaluation cases")

    def _condition_display_name(self, cond_data: Dict) -> str:
        return cond_data.get("condition_name") or cond_data.get("name") or ""

    def _evidence_display_text(self, ev_data: Dict) -> str:
        # `name` in release_evidences.json is the code itself — use question_en
        return ev_data.get("question_en") or ev_data.get("name") or ""

    def build_graph(self):
        """Build NetworkX directed graph with nodes and edges"""
        logger.info("Building knowledge graph...")

        # Add condition nodes
        for cond_id, cond_data in self.conditions.items():
            self.graph.add_node(
                cond_id,
                node_type="condition",
                name=self._condition_display_name(cond_data),
                prevalence=cond_data.get("prevalence", 0),
                severity=cond_data.get("severity", 3),
            )

        # Add evidence nodes (base codes only)
        for ev_id, ev_data in self.evidences.items():
            base_ev = _base_evidence(ev_id)
            self.graph.add_node(
                base_ev,
                node_type="evidence",
                text=self._evidence_display_text(ev_data),
                name=base_ev,
                icd_code=ev_data.get("icd_code", ""),
            )

        # All baseline evidences per condition — for computing absent set.
        # release_conditions.json lists candidate evidences per pathology under
        # `symptoms` and `antecedents`. We'll treat any evidence in
        # `condition_evidence_counts[C]` as a candidate; the present/absent
        # distinction comes from the eval_set cases.
        cond_to_candidate_evidences: Dict[str, Set[str]] = {}
        for cond_id, cond_data in self.conditions.items():
            cand = set()
            for k in cond_data.get("symptoms", {}):
                cand.add(_base_evidence(k))
            for k in cond_data.get("antecedents", {}):
                cand.add(_base_evidence(k))
            cond_to_candidate_evidences[cond_id] = cand

        logger.info("Adding edges (condition -> evidence relationships)...")
        edge_count = 0

        for case in self.eval_cases:
            condition_id = str(case.get("PATHOLOGY", ""))
            if not condition_id:
                continue
            self.condition_case_counts[condition_id] += 1

            try:
                present_evidences_raw = ast.literal_eval(case.get("EVIDENCES", "[]"))
            except Exception as e:
                logger.warning(f"Could not parse EVIDENCES for case (cond={condition_id}): {e}")
                present_evidences_raw = []
            present_evidences = [_base_evidence(str(e)) for e in present_evidences_raw]
            present_evidences_set = set(present_evidences)

            # All candidate evidences that *could* have been asked for this
            # condition. The DDXPlus eval cases only record what was asked (and
            # the answer); absent evidence here means "asked and answered no"
            # vs. "not asked at all". We infer absence by taking the union of
            # all candidate evidences for this condition (across the whole
            # eval set) minus the ones recorded as present in this case. That's
            # a noisy proxy but better than nothing.
            candidate_set = cond_to_candidate_evidences.get(condition_id, set()) | present_evidences_set
            absent_evidences = candidate_set - present_evidences_set

            # Present evidences: strong connection
            for ev_id in present_evidences:
                self.graph.add_edge(
                    condition_id,
                    ev_id,
                    weight=1.0,
                    type="present",
                )
                self.evidence_condition_counts[ev_id][condition_id] += 1
                self.condition_evidence_counts[condition_id][ev_id] += 1
                edge_count += 1

            # Absent evidences: weak negative connection (for discriminating)
            for ev_id in absent_evidences:
                if not self.graph.has_edge(condition_id, ev_id):
                    self.graph.add_edge(
                        condition_id,
                        ev_id,
                        weight=0.0,
                        type="absent",
                    )
                    edge_count += 1
                self.evidence_condition_absent_counts[ev_id][condition_id] += 1

        logger.info(f"Added {edge_count} edges across {sum(self.condition_case_counts.values())} cases")

    def compute_information_gain(
        self,
        evidence_id: str,
        present_symptoms: Optional[Set[str]] = None,
        asked_symptoms: Optional[Set[str]] = None,
    ) -> float:
        """Compute expected-posterior-entropy IG (kept for standalone use)."""
        # This is a thin wrapper used by tooling — the runtime IG lives in kg.py.
        if present_symptoms is None:
            present_symptoms = set()
        if asked_symptoms is None:
            asked_symptoms = set()

        if evidence_id not in self.evidence_condition_counts:
            return 0.0

        present_counts = self.evidence_condition_counts[evidence_id]
        absent_counts = self.evidence_condition_absent_counts.get(evidence_id, Counter())

        total_present = sum(present_counts.values())
        total_absent = sum(absent_counts.values())
        total = total_present + total_absent
        if total == 0:
            return 0.0

        # P(condition | E present) and P(condition | E absent)
        cond_set = set(present_counts) | set(absent_counts)
        p_present = total_present / total
        p_absent = total_absent / total

        def _entropy(counter: Counter) -> float:
            tot = sum(counter.values())
            if tot == 0:
                return 0.0
            probs = np.array([c / tot for c in counter.values()])
            return float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))

        return _entropy(Counter({c: present_counts[c] + absent_counts[c] for c in cond_set})) - (
            p_present * _entropy(present_counts) + p_absent * _entropy(absent_counts)
        )

    def save_graph(self, output_path: str = "backend/data/ddxplus_kg.pkl"):
        """Save graph and metadata to pickle"""
        backend_dir = Path(__file__).parent.parent
        full_path = backend_dir / output_path
        logger.info(f"Saving knowledge graph to {full_path}")

        kg_data = {
            "graph": self.graph,
            "conditions": self.conditions,
            "evidences": self.evidences,
            "evidence_condition_counts": self.evidence_condition_counts,
            "condition_evidence_counts": self.condition_evidence_counts,
            "evidence_condition_absent_counts": self.evidence_condition_absent_counts,
            "condition_case_counts": self.condition_case_counts,
        }

        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(kg_data, f)

        logger.info(
            f"Graph saved: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def build(self):
        """Execute full pipeline"""
        self.load_data()
        self.build_graph()
        self.save_graph()
        logger.info("Knowledge graph built successfully!")


if __name__ == "__main__":
    builder = DDXPlusKGBuilder(data_dir="data")
    builder.build()
