import json
import os
import pickle
import threading
import logging
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
# Pre-built pickled KG (built by scripts/build_ddxplus_kg.py) is the canonical
# artifact at runtime. The JSON files alongside are the source-of-truth schema,
# used both by the build script and as a degraded runtime fallback.
KG_FILE = os.path.join(DATA_DIR, "ddxplus_kg.pkl")
CONDITIONS_FILE = os.path.join(DATA_DIR, "DDXPlus", "release_conditions.json")
EVIDENCES_FILE = os.path.join(DATA_DIR, "DDXPlus", "release_evidences.json")

# Canonical mapping from any department this code might emit to the canonical
# specialty name as seeded in supabase/migrations/0001_init.sql. Keep this
# table as the single source of truth for KG→Supabase specialty translation.
SPECIALTY_CANONICALIZATION = {
    "Pulmonology": "Respiratory",
    "Emergency Medicine": "General Medicine / Internal Medicine",
    "General Medicine": "General Medicine / Internal Medicine",
    "Internal Medicine": "General Medicine / Internal Medicine",
    # Already-canonical names are passed through (added here for clarity).
    "Cardiology": "Cardiology",
    "Dermatology": "Dermatology",
    "Orthopedics": "Orthopedics",
    "Gastroenterology": "Gastroenterology",
    "Neurology": "Neurology",
    "Pediatrics": "Pediatrics",
    "Psychiatry": "Psychiatry",
    "Respiratory": "Respiratory",
    "General Medicine / Internal Medicine": "General Medicine / Internal Medicine",
}
# Seeded specialties in Supabase (supabase/migrations/0001_init.sql line 234).
# If KG returns a specialty not in this set, we canonicalize via the table
# above; if it *still* isn't seeded, we fall back to General Medicine /
# Internal Medicine so booking-slot fetch never silently dead-ends.
SEEDED_SPECIALTIES = {
    "Cardiology",
    "Dermatology",
    "Orthopedics",
    "Gastroenterology",
    "Neurology",
    "Pediatrics",
    "Psychiatry",
    "Respiratory",
    "General Medicine / Internal Medicine",
}


def _canonicalize_specialty(s: Optional[str]) -> Optional[str]:
    """Map any specialty-ish string to the seed name in the Supabase specialty table.

    Returns None if the input is None; otherwise always returns a value in
    SEEDED_SPECIALTIES (so the caller's downstream Supabase slot-fetch never
    silently finds zero doctors).
    """
    if not s:
        return None
    if s in SEEDED_SPECIALTIES:
        return s
    canonical = SPECIALTY_CANONICALIZATION.get(s)
    if canonical and canonical in SEEDED_SPECIALTIES:
        return canonical
    return "General Medicine / Internal Medicine"


def _condition_display_name(cond_info: Dict) -> str:
    """Get the human-readable condition name.

    DDXPlus `release_conditions.json` uses `condition_name` (which mirrors the
    JSON top-level key, e.g. "Spontaneous pneumothorax"). Older code paths
    sometimes used `.get('name')` and got the empty string.
    """
    return cond_info.get("condition_name") or cond_info.get("name") or ""


def _evidence_display_text(ev_info: Dict) -> str:
    """Get the human-readable text for an evidence.

    DDXPlus `release_evidences.json` has `name` == the E_XX code itself, so
    we MUST read `question_en` for any human-text operation.
    """
    return ev_info.get("question_en", "") or ev_info.get("name", "")


def _base_evidence(evid: str) -> str:
    """Strip the value-suffix from an evidence code: 'E_204_@_V_10' -> 'E_204'."""
    return evid.split("_@_")[0] if "_@_" in evid else evid


# Tokens that carry no discriminative meaning for the semantic-duplicate
# filter below; comparing these across two evidences tells us nothing about
# whether they're really the same question.
_NON_DISCRIMINATIVE_TOKENS = {
    "a", "an", "the", "you", "your", "do", "does", "did", "have", "has",
    "had", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
    "with", "for", "any", "been", "recently", "lately", "currently", "now",
    "i", "me", "my", "there", "here", "what", "which", "be", "feel", "feeling",
}


def _evidence_question_tokens(ev_info: Dict) -> Set[str]:
    """Return the discriminative token set for an evidence's `question_en`.

    Lowercased, alnum-only, with stop/question boilerplate removed. Used by
    the semantic-duplicate filter in `rank_next_questions` so that near-
    identical questions (e.g. "Do you have chest pain?" vs "Do you have pain
    in your chest?") are detected as duplicates instead of relying on the
    LLM to self-censor.
    """
    text = (ev_info.get("question_en") or ev_info.get("name") or "").lower()
    tokens = set()
    for tok in text.split():
        # Strip punctuation so "pain?" and "pain" match.
        cleaned = "".join(ch for ch in tok if ch.isalnum())
        if cleaned and cleaned not in _NON_DISCRIMINATIVE_TOKENS and len(cleaned) > 1:
            tokens.add(cleaned)
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity over two token sets; 0 if either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.conditions: Dict[str, Dict] = {}
        self.evidences: Dict[str, Dict] = {}
        # evidence_condition_counts[E][C] = # cases where evidence E was present
        # in a case whose pathology was C. So evidence_condition_counts[E]
        # is a Counter of conditions where E appeared.
        self.evidence_condition_counts: Dict[str, Counter] = defaultdict(Counter)
        # condition_evidence_counts[C][E] = # cases where evidence E was present
        # in a case whose pathology was C.
        self.condition_evidence_counts: Dict[str, Counter] = defaultdict(Counter)
        # evidence_condition_absent_counts[E][C] = # cases where evidence E was
        # absent in a case whose pathology was C. Needed for proper expected-
        # posterior-entropy IG (see `rank_next_questions`).
        self.evidence_condition_absent_counts: Dict[str, Counter] = defaultdict(Counter)
        # condition_cases[C] = total # cases for pathology C. Needed for the
        # P(E|C) normalization in the IG calc.
        self.condition_case_counts: Counter = Counter()
        self._loaded = False
        self._load_data()

    def _load_data(self):
        """Load pre-built KG or fall back to JSON."""
        try:
            if os.path.exists(KG_FILE):
                logger.info(f"Loading KG from {KG_FILE}")
                with open(KG_FILE, "rb") as f:
                    kg_data = pickle.load(f)
                    self.graph = kg_data["graph"]
                    self.conditions = kg_data["conditions"]
                    self.evidences = kg_data["evidences"]
                    self.evidence_condition_counts = kg_data["evidence_condition_counts"]
                    self.condition_evidence_counts = kg_data["condition_evidence_counts"]
                    # New fields added for proper IG. Backwards-compat: if an
                    # older pickle is loaded, these will be missing and the
                    # IG calc will fall back gracefully.
                    self.evidence_condition_absent_counts = kg_data.get(
                        "evidence_condition_absent_counts", defaultdict(Counter)
                    )
                    self.condition_case_counts = kg_data.get(
                        "condition_case_counts", Counter()
                    )
                    self._loaded = True
                    logger.info(
                        f"Loaded KG: {self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges"
                    )
            else:
                logger.warning(f"KG not found at {KG_FILE}, loading from JSON")
                self._load_from_json()
        except Exception as e:
            logger.error(f"Error loading KG: {e}")
            self._load_from_json()

    def _load_from_json(self):
        """Fallback: load from release_conditions.json + release_evidences.json
        and rebuild the condition↔evidence adjacency from the `symptoms` and
        `antecedents` blocks in each condition. This makes the JSON fallback
        actually usable for `rank_next_questions` (previous version left
        evidence_condition_counts empty, silently degrading the Q-loop).
        """
        try:
            if os.path.exists(CONDITIONS_FILE):
                with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
                    self.conditions = json.load(f)
                logger.info(f"Loaded {len(self.conditions)} conditions from JSON")

            if os.path.exists(EVIDENCES_FILE):
                with open(EVIDENCES_FILE, "r", encoding="utf-8") as f:
                    self.evidences = json.load(f)
                logger.info(f"Loaded {len(self.evidences)} evidences from JSON")

            # Build adjacency from the per-condition `symptoms` block. Each
            # condition lists the evidence codes that are relevant to it.
            for cond_id, cond_data in self.conditions.items():
                self.graph.add_node(
                    cond_id,
                    type="condition",
                    name=_condition_display_name(cond_data),
                    severity=cond_data.get("severity", 3),
                )
                # Symptom evidences
                for ev_id, ev_value in (cond_data.get("symptoms") or {}).items():
                    base_ev = _base_evidence(ev_id)
                    if base_ev not in self.evidences:
                        continue
                    self.graph.add_node(
                        base_ev, type="evidence", text=_evidence_display_text(self.evidences[base_ev])
                    )
                    # We don't have case-level counts from JSON, so use a
                    # uniform weight of 1 per condition. IG will degrade to
                    # equal-entropy heuristics, but at least compatible
                    # condition lookup will work.
                    self.graph.add_edge(cond_id, base_ev, weight=1.0, type="present")
                    self.evidence_condition_counts[base_ev][cond_id] += 1
                    self.condition_evidence_counts[cond_id][base_ev] += 1
                # Antecedents
                for ev_id, ev_value in (cond_data.get("antecedents") or {}).items():
                    base_ev = _base_evidence(ev_id)
                    if base_ev not in self.evidences:
                        continue
                    self.graph.add_node(
                        base_ev, type="evidence", text=_evidence_display_text(self.evidences[base_ev])
                    )
                    self.graph.add_edge(cond_id, base_ev, weight=1.0, type="antecedent")
                    self.evidence_condition_counts[base_ev][cond_id] += 1
                    self.condition_evidence_counts[cond_id][base_ev] += 1

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

    def _entropy(self, counts: Counter) -> float:
        """Shannon entropy over a Counter's values."""
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        probs = np.array([c / total for c in counts.values()], dtype=float)
        # Only positive probs contribute (other terms are 0 * log(0))
        nonzero = probs[probs > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))

    def rank_next_questions(
        self,
        present_symptoms: List[str],
        asked_symptoms: Optional[List[str]] = None,
        absent_symptoms: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank next questions by expected-posterior-entropy (true Information Gain).

        For each candidate evidence E not yet asked/present/absent, score = the
        expected entropy of the posterior distribution over compatible
        conditions after the patient answers whether they have E. Lower is
        better. We negate so higher score → more discriminative (consistent
        with the historical return ordering).

        Args:
          present_symptoms: evidence codes the patient has confirmed.
          asked_symptoms:   evidence codes we've already asked about.
          absent_symptoms:  evidence codes the patient has answered "no" to.
        """
        if not self._loaded:
            logger.warning("KG not loaded, returning empty results")
            return []

        if asked_symptoms is None:
            asked_symptoms = []
        if absent_symptoms is None:
            absent_symptoms = []

        present_set = set(_base_evidence(e) for e in present_symptoms)
        asked_set = set(_base_evidence(e) for e in asked_symptoms)
        absent_set = set(_base_evidence(e) for e in absent_symptoms)

        # Pre-compute the discriminative-token sets for every evidence we've
        # already surfaced (present / asked / absent). Candidates whose
        # `question_en` is highly similar to any of these are dropped below so
        # the Q-loop doesn't ask "Do you have chest pain?" right after "Do you
        # have pain in your chest?" — the KG layer suppresses near-duplicates
        # rather than relying on the LLM to self-censor. (see GH issue: semantic
        # duplicate next-questions.)
        already_covered_tokens: List[Set[str]] = []
        for evid in present_set | asked_set | absent_set:
            info = self.evidences.get(evid)
            if info:
                toks = _evidence_question_tokens(info)
                if toks:
                    already_covered_tokens.append(toks)

        # Find conditions compatible with the *positive* evidences observed.
        if present_set:
            compatible: Set[str] = set()
            for symptom in present_set:
                if symptom in self.evidence_condition_counts:
                    compatible.update(self.evidence_condition_counts[symptom].keys())
        else:
            compatible = set(self.conditions.keys())

        if not compatible:
            compatible = set(self.conditions.keys())

        # Prior probability over compatible conditions, weighted by the case
        # count per condition so common pathologies aren't over-asked.
        if self.condition_case_counts and compatible:
            weights = {c: float(self.condition_case_counts.get(c, 0)) or 1.0 for c in compatible}
        else:
            weights = {c: 1.0 for c in compatible}
        total_weight = sum(weights.values()) or 1.0
        prior = {c: weights[c] / total_weight for c in compatible}
        prior_entropy = -sum(p * np.log2(p) for p in prior.values() if p > 0)

        candidate_evidences: Set[str] = set()
        for condition_id in compatible:
            for _, evidence_id, _ in self.graph.out_edges(condition_id, data=True):
                if (
                    evidence_id not in asked_set
                    and evidence_id not in present_set
                    and evidence_id not in absent_set
                ):
                    candidate_evidences.add(evidence_id)

        # Semantic-duplicate suppression: drop candidates whose `question_en`
        # is highly similar (Jaccard >= SEMANTIC_DUP_THRESHOLD) to any evidence
        # already surfaced. This is the KG-layer fix for the "asks the same
        # question twice" problem — correlated evidence codes that rephrase the
        # same symptom are grouped here instead of being surfaced one after the
        # other. The threshold is intentionally lenient: exact code equality was
        # already filtered by the set checks above, so this only catches genuine
        # paraphrases.
        SEMANTIC_DUP_THRESHOLD = 0.6
        if already_covered_tokens and candidate_evidences:
            survivor: Set[str] = set()
            for evidence_id in candidate_evidences:
                cand_tokens = _evidence_question_tokens(self.evidences.get(evidence_id, {}))
                if not cand_tokens:
                    survivor.add(evidence_id)
                    continue
                is_dup = False
                for ref_tokens in already_covered_tokens:
                    if _jaccard(cand_tokens, ref_tokens) >= SEMANTIC_DUP_THRESHOLD:
                        is_dup = True
                        break
                if not is_dup:
                    survivor.add(evidence_id)
            candidate_evidences = survivor

        # Score each candidate by expected posterior entropy.
        # We need P(E=present | C=c) and P(E=absent | C=c) for each compatible
        # condition c. Build these counts once from the build-time stats.
        evidence_scores: Dict[str, float] = {}
        for evidence_id in candidate_evidences:
            present_counts = self.evidence_condition_counts.get(evidence_id, Counter())
            absent_counts = self.evidence_condition_absent_counts.get(evidence_id, Counter())
            # If we don't have absent-count stats (older pickle / JSON fallback),
            # estimate absent counts: total cases of C minus present cases of E
            # over C. Absent of len == 0 ⇒ P(E=present|C) = present_counts[C] /
            # condition_case_counts[C]; absent = 1 − that.
            p_present_given_c: Dict[str, float] = {}
            p_absent_given_c: Dict[str, float] = {}
            for c in compatible:
                if self.condition_case_counts:
                    total_c = self.condition_case_counts.get(c, 0)
                else:
                    # JSON-fallback: assume 1 case per mentioned condition
                    total_c = max(present_counts.get(c, 0), 1)
                p_c = total_c if total_c > 0 else 1
                p_present = present_counts.get(c, 0) / p_c
                if absent_counts:
                    p_absent = absent_counts.get(c, 0) / p_c
                else:
                    p_absent = 1.0 - p_present
                # Laplace smoothing so unseen combos don't produce 0 probabilities
                # that collapse posterior entropy artificially.
                alpha = 0.5
                p_present = (p_present * p_c + alpha) / (p_c + 2 * alpha)
                p_absent   = (p_absent   * p_c + alpha) / (p_c + 2 * alpha)
                p_present_given_c[c] = p_present
                p_absent_given_c[c] = p_absent

            # Marginal probability of E present / absent over compatible conditions
            p_e_present = sum(prior[c] * p_present_given_c[c] for c in compatible)
            p_e_absent = sum(prior[c] * p_absent_given_c[c] for c in compatible)
            if p_e_present <= 0 and p_e_absent <= 0:
                continue

            # Posterior P(C | E=present) ∝ P(E=present | C) * P(C)
            post_present = {c: prior[c] * p_present_given_c[c] for c in compatible}
            s_present = sum(post_present.values())
            post_absent = {c: prior[c] * p_absent_given_c[c] for c in compatible}
            s_absent = sum(post_absent.values())

            ent_present = 0.0
            if s_present > 0:
                probs = np.array([v / s_present for v in post_present.values()])
                nonzero = probs[probs > 0]
                ent_present = float(-np.sum(nonzero * np.log2(nonzero)))
            ent_absent = 0.0
            if s_absent > 0:
                probs = np.array([v / s_absent for v in post_absent.values()])
                nonzero = probs[probs > 0]
                ent_absent = float(-np.sum(nonzero * np.log2(nonzero)))

            expected_posterior_entropy = (
                p_e_present * ent_present + p_e_absent * ent_absent
            )
            # Information gain = prior_entropy − expected_posterior_entropy.
            # Higher IG → better question. Store IG so `reverse=True` keeps the
            # best questions on top.
            ig = prior_entropy - expected_posterior_entropy
            evidence_scores[evidence_id] = ig

        ranked = sorted(evidence_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:10]

    def get_condition_specialty(self, condition_id: str) -> Optional[str]:
        """Map a condition to a canonical Supabase specialty name.

        Returns a value that is guaranteed to be in SEEDED_SPECIALTIES (or None
        if the condition is unknown). Previously this returned raw enum names
        like 'Pulmonology' / 'Emergency Medicine' that aren't in the seed
        table, which made the booking-flow slot fetch silently dead-end.
        """
        cond_info = self.get_condition_info(condition_id)
        if not cond_info:
            return None
        # Prefer an explicit per-condition specialty if the build script
        # stored one (faster than substring-matching); fall back to that.
        if "specialty" in cond_info and cond_info["specialty"]:
            return _canonicalize_specialty(cond_info["specialty"])

        cond_name = _condition_display_name(cond_info).lower()

        specialty_mapping = {
            # Cardiology
            "cardio": "Cardiology", "heart": "Cardiology", "cardiac": "Cardiology",
            "angina": "Cardiology", "myocard": "Cardiology", "atrial fibrillation": "Cardiology",
            "hypertension": "Cardiology", "arrhythmia": "Cardiology", "coronary": "Cardiology",
            "nstem": "Cardiology", "stemi": "Cardiology", "pericarditis": "Cardiology",
            # Pulmonology → Respiratory (canonical)
            "pulmon": "Pulmonology", "pneumothorax": "Pulmonology", "asthma": "Pulmonology",
            "copd": "Pulmonology", "bronch": "Pulmonology", "pneumonia": "Pulmonology",
            "respiratory": "Pulmonology", "lung": "Pulmonology", "pleural": "Pulmonology",
            "emphysema": "Pulmonology", "sarcoid": "Pulmonology", "laryngospasm": "Pulmonology",
            "laryngitis": "Pulmonology", "larynx": "Pulmonology", "tracheitis": "Pulmonology",
            "epiglottitis": "Pulmonology",
            # Gastroenterology
            "gastro": "Gastroenterology", "gerd": "Gastroenterology", "reflux": "Gastroenterology",
            "boerhaave": "Gastroenterology", "esophag": "Gastroenterology", "gastric": "Gastroenterology",
            "peptic": "Gastroenterology", "ulcer": "Gastroenterology", "ibd": "Gastroenterology",
            "crohn": "Gastroenterology", "colitis": "Gastroenterology", "hepat": "Gastroenterology",
            "cirrhosis": "Gastroenterology", "pancrea": "Gastroenterology", "cholecyst": "Gastroenterology",
            "appendic": "Gastroenterology", "diverticul": "Gastroenterology",
            # Neurology
            "neuro": "Neurology", "headache": "Neurology", "migraine": "Neurology",
            "cluster": "Neurology", "seizure": "Neurology", "epilepsy": "Neurology",
            "stroke": "Neurology", "meningitis": "Neurology", "encephalitis": "Neurology",
            "multiple sclerosis": "Neurology", "parkins": "Neurology", "guillain": "Neurology",
            "myasthenia": "Neurology",
            # Dermatology
            "derm": "Dermatology", "skin": "Dermatology", "rash": "Dermatology",
            "eczema": "Dermatology", "psoriasis": "Dermatology", "melanoma": "Dermatology",
            "urticaria": "Dermatology", "acne": "Dermatology", "cellulitis": "Dermatology",
            # Orthopedics
            "ortho": "Orthopedics", "fracture": "Orthopedics", "rib": "Orthopedics",
            "bone": "Orthopedics", "joint": "Orthopedics", "sprain": "Orthopedics",
            "strain": "Orthopedics", "arthrit": "Orthopedics", "disloc": "Orthopedics",
            # Psychiatry
            "psych": "Psychiatry", "depression": "Psychiatry", "anxiety": "Psychiatry",
            "bipolar": "Psychiatry", "schizophrenia": "Psychiatry", "panic": "Psychiatry",
            "ptsd": "Psychiatry", "eating disorder": "Psychiatry",
        }

        # 'tumor' is no longer blanket-routed to Neurology; neoplasms will fall
        # through to General Medicine / Internal Medicine which is the right
        # default for an unspecified oncology case where we don't have an
        # Oncology specialty seeded.

        for key, specialty in specialty_mapping.items():
            if key in cond_name:
                return _canonicalize_specialty(specialty)

        return _canonicalize_specialty("General Medicine / Internal Medicine")

    def get_condition_severity(self, condition_id: str) -> int:
        """Get severity/triage level for a condition (1-5, where 1 is most severe)."""
        cond_info = self.get_condition_info(condition_id)
        if "severity" in cond_info:
            sev = cond_info["severity"]
            if isinstance(sev, int) and 1 <= sev <= 5:
                return sev
            logger.warning(f"Condition {condition_id} has malformed severity={sev!r}, using default")
        # Default — mid-urgency. (Previously fell back to a substring map which
        # was almost always dead code given DDXPlus ships `severity` per
        # condition. Keep the default here so we degrade visibly.)
        return 3


_kg_lock = threading.Lock()
_kg_instance: Optional[KnowledgeGraph] = None


def get_kg() -> KnowledgeGraph:
    """Get singleton KG instance (thread-safe)."""
    global _kg_instance
    if _kg_instance is None:
        with _kg_lock:
            if _kg_instance is None:
                _kg_instance = KnowledgeGraph()
    return _kg_instance


def reset_kg() -> None:
    """Test hook: reset the singleton."""
    global _kg_instance
    with _kg_lock:
        _kg_instance = None


# Keep the historical `@lru_cache(maxsize=1)` API surface for any code that
# imported it directly (renamed to get_kg above; both work).
@lru_cache(maxsize=1)
def _get_kg_lru() -> KnowledgeGraph:
    return KnowledgeGraph()
