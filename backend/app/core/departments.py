import json
from typing import List, Dict

# The core 17 departments typically found in major hospitals
DEPARTMENTS = [
    "General Medicine / Internal Medicine",
    "Pediatrics",
    "Emergency Medicine",
    "Cardiology",
    "Neurology",
    "Gastroenterology",
    "Orthopedics",
    "Dermatology",
    "Obstetrics and Gynecology (OB/GYN)",
    "Psychiatry",
    "Ophthalmology",
    "ENT (Otolaryngology)",
    "Urology",
    "Pulmonology",
    "Hematology",
    "Rheumatology",
    "General Surgery"
]

# When smaller hospitals don't have specialized departments, we fallback to more general ones.
DEPARTMENT_FALLBACKS = {
    "Hematology": "General Medicine / Internal Medicine",
    "Pulmonology": "General Medicine / Internal Medicine",
    "Rheumatology": "General Medicine / Internal Medicine",
    "Cardiology": "General Medicine / Internal Medicine",
    "Neurology": "General Medicine / Internal Medicine",
    "Gastroenterology": "General Medicine / Internal Medicine"
}

# A mapping from DDXPlus/MedQuAD common conditions to their primary department.
# Common stuff like headache, fever, cold go to General Medicine / Pediatrics (based on age).
CONDITION_TO_DEPARTMENT = {
    "URTI": "General Medicine / Internal Medicine", # Cold/Cough
    "Influenza": "General Medicine / Internal Medicine", # Fever/Aches
    "Viral pharyngitis": "ENT (Otolaryngology)", # Or General Med
    "Cluster headache": "Neurology", # Or General Med for basic headache
    "Acute otitis media": "ENT (Otolaryngology)",
    "Acute rhinosinusitis": "ENT (Otolaryngology)",
    "Bronchitis": "General Medicine / Internal Medicine", 
    "Pneumonia": "Pulmonology",
    "Allergic sinusitis": "ENT (Otolaryngology)",
    "Anemia": "Hematology",
    "GERD": "Gastroenterology",
    "Inguinal hernia": "General Surgery",
    "Myasthenia gravis": "Neurology",
    "Whooping cough": "Pediatrics",
    "Anaphylaxis": "Emergency Medicine",
    "Epiglottitis": "Emergency Medicine",
    "Guillain-Barré syndrome": "Neurology",
    "Atrial fibrillation": "Cardiology",
    "Pulmonary embolism": "Emergency Medicine",
    "Spontaneous pneumothorax": "Emergency Medicine",
    "Boerhaave": "Emergency Medicine",
    "Spontaneous rib fracture": "Orthopedics",
    "HIV (initial infection)": "General Medicine / Internal Medicine",
    "Chagas": "General Medicine / Internal Medicine",
    "Scombroid food poisoning": "Emergency Medicine",
    "Myocarditis": "Cardiology",
    "Larygospasm": "Emergency Medicine",
    "Acute dystonic reactions": "Emergency Medicine",
    "Localized edema": "General Medicine / Internal Medicine",
    "SLE": "Rheumatology",
    "Tuberculosis": "Pulmonology",
    "Unstable angina": "Emergency Medicine",
    "Stable angina": "Cardiology",
    "Ebola": "Emergency Medicine",
    "Panic attack": "Psychiatry",
    "Bronchospasm / acute asthma exacerbation": "Emergency Medicine",
    "Acute COPD exacerbation / infection": "Pulmonology",
    "Bronchiolitis": "Pediatrics",
    "Pulmonary neoplasm": "Pulmonology",
    "Possible NSTEMI / STEMI": "Emergency Medicine",
    "Sarcoidosis": "Pulmonology",
    "Pancreatic neoplasm": "Gastroenterology",
    "Acute pulmonary edema": "Emergency Medicine",
    "Pericarditis": "Cardiology"
}

def get_department_for_condition(condition: str, age: int, available_departments: List[str] = None) -> str:
    """
    Given a predicted condition, patient age, and a list of available departments at the specific hospital,
    returns the most appropriate department.
    """
    # 1. Base Mapping
    base_dept = CONDITION_TO_DEPARTMENT.get(condition, "General Medicine / Internal Medicine")
    
    # 2. Age-based overrides
    # If the patient is < 18 and the condition is usually General Med or standard respiratory, route to Pediatrics.
    if age < 18 and base_dept in ["General Medicine / Internal Medicine"]:
        base_dept = "Pediatrics"
        
    # 3. Hospital Availability Fallback
    if available_departments:
        if base_dept not in available_departments:
            # Try to find a fallback
            fallback = DEPARTMENT_FALLBACKS.get(base_dept)
            if fallback and fallback in available_departments:
                return fallback
            else:
                # Ultimate fallback
                return "Emergency Medicine" if base_dept == "Emergency Medicine" else "General Medicine / Internal Medicine"
                
    return base_dept
