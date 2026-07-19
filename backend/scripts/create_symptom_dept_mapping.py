#!/usr/bin/env python3
"""
Create symptom-to-department mapping from multiple sources:
1. DDXPlus conditions -> specialty via KG
2. Symptom2Disease.csv -> disease -> specialty
3. Keyword fallback mapping
"""

import json
import os
import pickle
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
KG_FILE = DATA_DIR / "ddxplus_kg.pkl"
SYMPTOM2DISEASE_FILE = DATA_DIR / "Symptom2Disease.csv"
EVIDENCES_FILE = DATA_DIR / "DDXPlus" / "release_evidences.json"
CONDITIONS_FILE = DATA_DIR / "DDXPlus" / "release_conditions.json"
OUTPUT_FILE = BACKEND_DIR / "model" / "symptom_dept_mapping.json"


# Department keyword fallback
DEPT_KEYWORDS = {
    "Cardiology": ["chest pain", "palpitation", "heart", "cardiac", "bp", "blood pressure", "hypertension"],
    "Neurology": ["headache", "migraine", "dizziness", "seizure", "confusion", "weakness", "numbness", "tingling", "stroke", "memory"],
    "Pulmonology": ["cough", "shortness of breath", "wheezing", "breathing", "lung", "asthma", "copd", "pneumonia"],
    "Gastroenterology": ["abdominal pain", "stomach pain", "nausea", "vomiting", "diarrhea", "constipation", "bloating", "acid reflux", "heartburn"],
    "Dermatology": ["rash", "itching", "hives", "skin", "acne", "eczema", "psoriasis", "mole", "lesion"],
    "Orthopedics": ["joint pain", "back pain", "knee pain", "shoulder pain", "neck pain", "fracture", "sprain", "arthritis", "muscle pain"],
    "Rheumatology": ["joint pain", "stiffness", "swelling", "autoimmune", "lupus", "rheumatoid"],
    "Endocrinology": ["diabetes", "thyroid", "hormone", "weight gain", "weight loss", "fatigue", "excessive thirst"],
    "Nephrology": ["kidney", "urine", "urination", "blood in urine", "kidney stone"],
    "Hematology": ["anemia", "bleeding", "bruising", "blood", "clotting"],
    "Infectious Disease": ["fever", "infection", "chills", "sweats", "feverish"],
    "Psychiatry": ["anxiety", "depression", "panic", "mood", "sleep", "insomnia", "stress"],
    "Ophthalmology": ["eye", "vision", "blurry", "eye pain", "red eye"],
    "ENT": ["ear", "throat", "nose", "sinus", "hearing", "sore throat", "tonsil"],
    "Urology": ["urinary", "prostate", "erectile", "testicular", "kidney stone"],
    "Gynecology": ["menstrual", "period", "pregnancy", "vaginal", "pelvic", "obgyn"],
}


def load_kg():
    """Load the knowledge graph."""
    if not KG_FILE.exists():
        print(f"KG file not found: {KG_FILE}")
        return None
    
    with open(KG_FILE, 'rb') as f:
        kg_data = pickle.load(f)
    
    print(f"Loaded KG: {len(kg_data['conditions'])} conditions, {len(kg_data['evidences'])} evidences")
    return kg_data


def load_evidences_and_conditions():
    """Load DDXPlus evidences and conditions from JSON."""
    evidences = {}
    conditions = {}
    
    if EVIDENCES_FILE.exists():
        with open(EVIDENCES_FILE, 'r') as f:
            evidences = json.load(f)
        print(f"Loaded {len(evidences)} evidences")
    
    if CONDITIONS_FILE.exists():
        with open(CONDITIONS_FILE, 'r') as f:
            conditions = json.load(f)
        print(f"Loaded {len(conditions)} conditions")
    
    return evidences, conditions


def get_specialty_from_condition_name(condition_name: str) -> str:
    """Map condition name to specialty using keyword matching."""
    condition_lower = condition_name.lower()
    
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
        'infect': 'Infectious Disease',
        'psych': 'Psychiatry',
        'ophthal': 'Ophthalmology',
        'oto': 'ENT',
        'uro': 'Urology',
        'gyne': 'Gynecology',
        'oncology': 'Oncology',
        'emergency': 'Emergency Medicine',
    }
    
    for key, specialty in specialty_mapping.items():
        if key in condition_lower:
            return specialty
    
    return "General Medicine"


def build_mapping_from_kg(kg_data) -> dict:
    """Build evidence->department mapping from KG conditions."""
    mapping = defaultdict(Counter)
    
    for cond_id, cond_info in kg_data['conditions'].items():
        cond_name = cond_info.get('name', '')
        specialty = get_specialty_from_condition_name(cond_name)
        
        # Get evidences for this condition
        for ev_id in kg_data['condition_evidence_counts'].get(cond_id, {}):
            mapping[ev_id][specialty] += 1
    
    # Convert to simple mapping: evidence -> most common specialty
    result = {}
    for ev_id, specialty_counts in mapping.items():
        most_common = specialty_counts.most_common(1)
        if most_common:
            result[ev_id] = most_common[0][0]
    
    print(f"Built mapping for {len(result)} evidence codes from KG")
    return result


def build_mapping_from_symptom2disease(evidences: dict) -> dict:
    """Build evidence->department mapping from Symptom2Disease.csv."""
    if not SYMPTOM2DISEASE_FILE.exists():
        print(f"Symptom2Disease file not found: {SYMPTOM2DISEASE_FILE}")
        return {}
    
    df = pd.read_csv(SYMPTOM2DISEASE_FILE)
    print(f"Loaded {len(df)} rows from Symptom2Disease.csv")
    
    # Create disease -> specialty mapping
    disease_to_specialty = {}
    for disease in df['Disease'].unique():
        disease_to_specialty[disease] = get_specialty_from_condition_name(disease)
    
    # Map evidence codes to diseases based on name matching
    evidence_to_diseases = defaultdict(set)
    
    for _, row in df.iterrows():
        disease = row['Disease']
        symptom = row['Symptom'].lower()
        
        # Match symptom to evidence name
        for ev_id, ev_info in evidences.items():
            ev_name = ev_info.get('name', '').lower()
            if symptom in ev_name or ev_name in symptom:
                evidence_to_diseases[ev_id].add(disease)
    
    # Convert to evidence -> specialty
    mapping = {}
    for ev_id, diseases in evidence_to_diseases.items():
        specialties = [disease_to_specialty.get(d, "General Medicine") for d in diseases]
        if specialties:
            most_common = Counter(specialties).most_common(1)[0][0]
            mapping[ev_id] = most_common
    
    print(f"Built mapping for {len(mapping)} evidence codes from Symptom2Disease")
    return mapping


def combine_mappings(*mappings) -> dict:
    """Combine multiple mappings, preferring KG > Symptom2Disease > Keywords."""
    combined = {}
    
    # Priority order
    for mapping in mappings:
        for ev_id, dept in mapping.items():
            if ev_id not in combined:
                combined[ev_id] = dept
    
    return combined


def add_keyword_fallback(mapping: dict, evidences: dict) -> dict:
    """Add keyword-based fallback for unmapped evidences."""
    result = mapping.copy()
    
    for ev_id, ev_info in evidences.items():
        if ev_id in result:
            continue
        
        ev_name = ev_info.get('name', '').lower()
        
        # Check keywords
        for dept, keywords in DEPT_KEYWORDS.items():
            if any(kw in ev_name for kw in keywords):
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
    
    mappings = []
    
    # 1. From KG
    if kg_data:
        kg_mapping = build_mapping_from_kg(kg_data)
        mappings.append(kg_mapping)
    
    # 2. From Symptom2Disease
    if evidences:
        s2d_mapping = build_mapping_from_symptom2disease(evidences)
        mappings.append(s2d_mapping)
    
    # 3. Combine with priority
    combined = combine_mappings(*mappings)
    
    # 4. Add keyword fallback
    if evidences:
        combined = add_keyword_fallback(combined, evidences)
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nSaved mapping to {OUTPUT_FILE}")
    print(f"Total evidence codes mapped: {len(combined)}")
    
    # Print sample
    print("\nSample mappings:")
    for ev_id, dept in list(combined.items())[:20]:
        ev_name = evidences.get(ev_id, {}).get('name', 'Unknown')
        print(f"  {ev_id}: {ev_name} -> {dept}")
    
    # Also create reverse mapping for symptom names
    name_to_dept = {}
    for ev_id, dept in combined.items():
        ev_name = evidences.get(ev_id, {}).get('name', '').lower()
        if ev_name:
            name_to_dept[ev_name] = dept
    
    name_file = OUTPUT_FILE.parent / "symptom_name_dept_mapping.json"
    with open(name_file, 'w') as f:
        json.dump(name_to_dept, f, indent=2)
    print(f"Saved name-based mapping to {name_file}")


if __name__ == "__main__":
    main()