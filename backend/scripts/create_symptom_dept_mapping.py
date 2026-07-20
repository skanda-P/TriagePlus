#!/usr/bin/env python3
"""
Create symptom-to-department mapping from multiple sources:
1. DDXPlus conditions -> specialty via KG
2. Symptom2Disease.csv -> disease -> specialty
3. Keyword fallback mapping
"""

import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
KG_FILE = DATA_DIR / "ddxplus_kg.pkl"
SYMPTOM2DISEASE_FILE = DATA_DIR / "Symptom2Disease.csv"
EVIDENCES_FILE = DATA_DIR / "DDXPlus" / "release_evidences.json"
CONDITIONS_FILE = DATA_DIR / "DDXPlus" / "release_conditions.json"
OUTPUT_FILE = BACKEND_DIR / "model" / "symptom_dept_mapping.json"


# Department keyword fallback (used as final fallback). Each keyword is matched
# against the human-readable evidence `question_en` text (NOT the evidence
# code, which is what `name` actually contains in release_evidences.json).
DEPT_KEYWORDS = {
    "Cardiology": ["chest pain", "palpitation", "heart", "cardiac", "bp", "blood pressure", "hypertension"],
    "Neurology": ["headache", "migraine", "dizziness", "seizure", "confusion", "weakness", "numbness", "tingling", "stroke", "memory"],
    "Respiratory": ["cough", "shortness of breath", "wheezing", "breathing", "lung", "asthma", "copd", "pneumonia"],
    "Gastroenterology": ["abdominal pain", "stomach pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating", "acid reflux", "heartburn"],
    "Dermatology": ["rash", "itching", "hives", "skin", "acne", "eczema", "psoriasis", "mole", "lesion"],
    "Orthopedics": ["joint pain", "back pain", "knee pain", "shoulder pain", "neck pain", "fracture", "sprain", "arthritis", "muscle pain"],
    "Psychiatry": ["anxiety", "depression", "panic", "mood", "sleep", "insomnia", "stress"],
    "General Medicine / Internal Medicine": ["fever", "infection", "chills", "sweats", "feverish", "fatigue", "weight"],
}


def _base_evidence(evid: str) -> str:
    return evid.split("_@_")[0] if "_@_" in evid else evid


def load_kg():
    """Load the knowledge graph."""
    if not KG_FILE.exists():
        print(f"KG file not found: {KG_FILE}")
        return None

    with open(KG_FILE, "rb") as f:
        kg_data = pickle.load(f)

    print(f"Loaded KG: {len(kg_data['conditions'])} conditions, {len(kg_data['evidences'])} evidences")
    return kg_data


def load_evidences_and_conditions():
    """Load DDXPlus evidences and conditions from JSON."""
    evidences = {}
    conditions = {}

    if EVIDENCES_FILE.exists():
        with open(EVIDENCES_FILE, "r", encoding="utf-8") as f:
            evidences = json.load(f)
        print(f"Loaded {len(evidences)} evidences")

    if CONDITIONS_FILE.exists():
        with open(CONDITIONS_FILE, "r", encoding="utf-8") as f:
            conditions = json.load(f)
        print(f"Loaded {len(conditions)} conditions")

    return evidences, conditions


def _condition_name(info: dict) -> str:
    return info.get("condition_name") or info.get("name") or ""


def _evidence_text(info: dict) -> str:
    # release_evidences.json: `name` is the code itself; the human text is in `question_en`.
    return info.get("question_en") or info.get("name") or ""


def get_specialty_from_condition_name(condition_name: str) -> str:
    """Map condition name to specialty using keyword matching."""
    condition_lower = condition_name.lower()

    specialty_mapping = {
        "cardio": "Cardiology",
        "pulmon": "Respiratory",
        "gastro": "Gastroenterology",
        "neuro": "Neurology",
        "derm": "Dermatology",
        "ortho": "Orthopedics",
        "rheum": "General Medicine / Internal Medicine",  # No Rheumatology seeded
        "endo": "General Medicine / Internal Medicine",   # No Endocrinology seeded
        "nephro": "General Medicine / Internal Medicine",  # No Nephrology seeded
        "hemo": "General Medicine / Internal Medicine",   # No Hematology seeded
        "infect": "General Medicine / Internal Medicine",  # No Infectious Disease seeded
        "psych": "Psychiatry",
        "ophthal": "General Medicine / Internal Medicine",  # No Ophthalmology seeded
        "oto": "General Medicine / Internal Medicine",  # No ENT seeded
        "uro": "General Medicine / Internal Medicine",  # No Urology seeded
        "gyne": "General Medicine / Internal Medicine",  # No Gynecology seeded
        "oncology": "General Medicine / Internal Medicine",  # No Oncology seeded
        "emergency": "General Medicine / Internal Medicine",
    }

    for key, specialty in specialty_mapping.items():
        if key in condition_lower:
            return specialty

    return "General Medicine / Internal Medicine"


def build_mapping_from_kg(kg_data) -> dict:
    """Build evidence->department mapping from KG conditions.

    Fixes the prior bug where `cond_info.get('name', '')` returned the empty
    string for every condition (the correct field is `condition_name`), which
    forced every KG specialty to fall back to `get_specialty_from_condition_name('')`
    and uniformly return `General Medicine`.
    """
    mapping = defaultdict(Counter)

    for cond_id, cond_info in kg_data["conditions"].items():
        cond_name = _condition_name(cond_info)
        specialty = get_specialty_from_condition_name(cond_name)

        # Get evidences for this condition (base codes only)
        for ev_id in kg_data["condition_evidence_counts"].get(cond_id, {}):
            mapping[_base_evidence(ev_id)][specialty] += 1

    result = {}
    for ev_id, specialty_counts in mapping.items():
        most_common = specialty_counts.most_common(1)
        if most_common:
            result[ev_id] = most_common[0][0]

    print(f"Built mapping for {len(result)} evidence codes from KG")
    return result


def build_mapping_from_symptom2disease(evidences: dict) -> dict:
    """Build evidence->department mapping from Symptom2Disease.csv.

    Fixes the prior bug where the join key was `ev_info.get('name')` (which is
    the evidence code itself, never matching a symptom phrase). Now uses
    `question_en` so it actually matches.
    """
    if not SYMPTOM2DISEASE_FILE.exists():
        print(f"Symptom2Disease file not found: {SYMPTOM2DISEASE_FILE}")
        return {}

    df = pd.read_csv(SYMPTOM2DISEASE_FILE)
    print(f"Loaded {len(df)} rows from Symptom2Disease.csv")

    # Group symptoms by disease first so we don't loop evidences per row.
    disease_to_symptoms = df.groupby("Disease")["Symptom"].apply(
        lambda xs: [str(x).lower().strip() for x in xs]
    ).to_dict()

    # Build disease -> specialty once per disease.
    disease_to_specialty = {
        disease: get_specialty_from_condition_name(disease)
        for disease in disease_to_symptoms
    }

    # For each evidence, find diseases whose symptom tokens appear in the
    # evidence question_en. Use token containment rather than bidirectional
    # substring (which had false positives like "kin" inside "skin").
    evidence_to_diseases = defaultdict(set)
    for ev_id, ev_info in evidences.items():
        ev_text = _evidence_text(ev_info).lower()
        if not ev_text:
            continue
        ev_tokens = set(ev_text.split())
        for disease, symptoms in disease_to_symptoms.items():
            for symptom in symptoms:
                symptom_tokens = symptom.split()
                # Match if 2+ symptom tokens are present in the evidence text
                overlap = sum(1 for t in symptom_tokens if t in ev_tokens)
                if overlap >= max(1, len(symptom_tokens) // 2):
                    evidence_to_diseases[ev_id].add(disease)
                    break

    mapping = {}
    for ev_id, diseases in evidence_to_diseases.items():
        specialties = [disease_to_specialty.get(d, "General Medicine / Internal Medicine") for d in diseases]
        if specialties:
            most_common = Counter(specialties).most_common(1)[0][0]
            mapping[ev_id] = most_common

    print(f"Built mapping for {len(mapping)} evidence codes from Symptom2Disease")
    return mapping


def combine_mappings(mapping_list) -> dict:
    """Combine multiple mappings by declared priority (earlier = higher)."""
    combined = {}
    for mapping in mapping_list:
        for ev_id, dept in mapping.items():
            if ev_id not in combined:
                combined[ev_id] = dept
    return combined


def add_keyword_fallback(mapping: dict, evidences: dict) -> dict:
    """Add keyword-based fallback for unmapped evidences.

    Uses `question_en` for the keyword match, NOT `name` (which is the code).
    """
    result = mapping.copy()

    for ev_id, ev_info in evidences.items():
        if ev_id in result:
            continue

        ev_text = _evidence_text(ev_info).lower()
        if not ev_text:
            continue

        for dept, keywords in DEPT_KEYWORDS.items():
            if any(kw in ev_text for kw in keywords):
                result[ev_id] = dept
                break

    print(f"After keyword fallback: {len(result)} evidence codes mapped")
    return result


def main():
    print("=" * 60)
    print("Building Symptom-to-Department Mapping")
    print("=" * 60)

    # Load data
    kg_data = load_kg()
    evidences, conditions = load_evidences_and_conditions()

    # Source priority: KG > Symptom2Disease > Keywords
    # Use an explicit ordered list so subsequent refactor won't silently invert.
    mappings_in_order = []

    if kg_data:
        mappings_in_order.append(build_mapping_from_kg(kg_data))
    if evidences:
        mappings_in_order.append(build_mapping_from_symptom2disease(evidences))

    combined = combine_mappings(mappings_in_order)

    if evidences:
        combined = add_keyword_fallback(combined, evidences)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved mapping to {OUTPUT_FILE}")
    print(f"Total evidence codes mapped: {len(combined)}")

    # Print sample
    print("\nSample mappings:")
    for ev_id, dept in list(combined.items())[:20]:
        ev_text = _evidence_text(evidences.get(ev_id, {}))
        print(f"  {ev_id}: {ev_text[:60]} -> {dept}")

    # Also create reverse mapping for symptom names
    name_to_dept = {}
    for ev_id, dept in combined.items():
        ev_text = _evidence_text(evidences.get(ev_id, {})).lower()
        if ev_text:
            name_to_dept[ev_text] = dept

    name_file = OUTPUT_FILE.parent / "symptom_name_dept_mapping.json"
    with open(name_file, "w", encoding="utf-8") as f:
        json.dump(name_to_dept, f, indent=2)
    print(f"Saved name-based mapping to {name_file}")


if __name__ == "__main__":
    main()
