import os
import json
import asyncio
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..db.supabase_client import get_supabase
from .kg import get_kg
from .rag import get_rag_engine

# Minimal stub for HuggingFace NER since we aren't writing full ML pipelines here.
# In a real app, this would use d4data/biomedical-ner-all
def dummy_ner(text: str) -> List[str]:
    # Placeholder: mock extracting symptoms
    symptoms = []
    text_lower = text.lower()
    if "chest pain" in text_lower: symptoms.append("E_55")
    if "headache" in text_lower: symptoms.append("E_53")
    if "fever" in text_lower: symptoms.append("E_91")
    return symptoms

def evaluate_red_flags(symptoms: List[str], text: str) -> bool:
    # Standalone triggers
    text_lower = text.lower()
    if any(x in text_lower for x in ["loss of consciousness", "suicid", "kill myself", "can't breathe", "bleeding heavily"]):
        return True
    
    # Combination triggers mapped to dummy symptoms
    if "E_55" in symptoms and ("shortness of breath" in text_lower or "jaw pain" in text_lower):
        return True
    return False

# Department Synonyms
DEPARTMENT_SYNONYMS = {
    "skin": "Dermatology", "heart": "Cardiology", "child": "Pediatrics",
    "kids": "Pediatrics", "bone": "Orthopedics", "joint": "Orthopedics",
    "stomach": "Gastroenterology", "digestive": "Gastroenterology",
    "brain": "Neurology", "nerve": "Neurology", "mental": "Psychiatry",
    "lung": "Respiratory", "breathing": "Respiratory",
}
BOOKING_TRIGGER_PHRASES = ["book", "appointment", "schedule", "see a doctor", "see dr", "consult"]

class TriageState(TypedDict):
    session_id: str
    patient_id: str
    messages: List[str]
    present_symptoms: List[str]
    confidence: Optional[float]
    triage_level: Optional[int]
    department: Optional[str]
    payment_status: Optional[str]
    is_emergency: bool
    intent: Optional[str]
    requested_department_raw: Optional[str]
    requested_doctor_raw: Optional[str]
    selected_doctor_id: Optional[str]
    awaiting_department_choice: bool
    booking_intent: Optional[bool]
    available_slots: Optional[List[dict]]
    selected_slot_id: Optional[str]
    final_diagnosis: Optional[str]
    asked_symptoms: List[str]

# --- Nodes ---

def node_emergency_check(state: TriageState) -> TriageState:
    text = state["messages"][-1]
    symptoms = state.get("present_symptoms", []) + dummy_ner(text)
    
    matched = evaluate_red_flags(symptoms, text)
    if matched:
        state["is_emergency"] = True
        state["final_diagnosis"] = "Possible Medical Emergency"
        state["department"] = "Emergency Medicine"
        state["triage_level"] = 1
    return state

def node_emergency_response(state: TriageState) -> TriageState:
    supabase = get_supabase()
    
    # Update chat_session
    supabase.table("chat_session").update({
        "status": "completed",
        "is_emergency": True,
        "completed_at": datetime.utcnow().isoformat()
    }).eq("session_id", state["session_id"]).execute()
    
    # Write audit log
    supabase.table("audit_log").insert({
        "event": "emergency_flagged",
        "metadata": {"session_id": state["session_id"], "patient_id": state["patient_id"]}
    }).execute()
    
    state["messages"].append("EMERGENCY_TRIGGERED: Please seek immediate emergency medical care.")
    return state

def fuzzy_match_department(text: str, synonyms: dict, threshold: int = 85) -> str:
    try:
        from rapidfuzz import process, fuzz
        supabase = get_supabase()
        res = supabase.table("specialty").select("name").execute()
        specialties = [row["name"] for row in res.data]
        
        # Check synonyms
        for key, val in synonyms.items():
            if key in text.lower():
                return val
                
        if not specialties:
            return None
            
        match = process.extractOne(text, specialties, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            return match[0]
    except:
        pass
    return None

def fuzzy_match_doctor(text: str, threshold: int = 85):
    try:
        from rapidfuzz import process, fuzz
        supabase = get_supabase()
        res = supabase.table("doctor").select("id, name, specialty_id").execute()
        if not res.data: return None
        
        doctor_names = [r["name"] for r in res.data]
        match = process.extractOne(text, doctor_names, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            matched_doc = next(d for d in res.data if d["name"] == match[0])
            spec_res = supabase.table("specialty").select("name").eq("id", matched_doc["specialty_id"]).execute()
            if spec_res.data:
                return {"id": matched_doc["id"], "name": matched_doc["name"], "specialty_name": spec_res.data[0]["name"]}
    except:
        pass
    return None

def node_detect_intent(state: TriageState) -> TriageState:
    text = state["messages"][-1].lower()
    
    doc_match = fuzzy_match_doctor(text)
    if doc_match:
        state["intent"] = "direct_booking_doctor"
        state["selected_doctor_id"] = doc_match["id"]
        state["department"] = doc_match["specialty_name"]
        state["requested_doctor_raw"] = text
        return state
        
    dept_match = fuzzy_match_department(text, DEPARTMENT_SYNONYMS)
    if dept_match:
        state["intent"] = "direct_booking_department"
        state["department"] = dept_match
        state["requested_department_raw"] = text
        return state
        
    if any(phrase in text for phrase in BOOKING_TRIGGER_PHRASES):
        state["intent"] = "direct_booking_department"
        state["department"] = None
        state["requested_department_raw"] = text
        return state
        
    state["intent"] = "symptom_triage"
    return state

def node_prompt_department_choice(state: TriageState) -> TriageState:
    if not state.get("awaiting_department_choice"):
        state["awaiting_department_choice"] = True
        state["messages"].append("PROMPT_DEPARTMENT: Which department would you like to book an appointment with?")
    else:
        text = state["messages"][-1].lower()
        dept_match = fuzzy_match_department(text, DEPARTMENT_SYNONYMS)
        if dept_match:
            state["department"] = dept_match
            state["awaiting_department_choice"] = False
        else:
            state["messages"].append("PROMPT_DEPARTMENT_RETRY: I couldn't match that department. Please select from the available options.")
    return state

def node_fetch_slots_for_doctor(state: TriageState) -> TriageState:
    supabase = get_supabase()
    res = supabase.table("clinician_slot")\
        .select("id, start_time, doctor_id, doctor!inner(name, rating, avg_consult_min)")\
        .eq("doctor_id", state["selected_doctor_id"])\
        .eq("status", "open")\
        .order("start_time")\
        .limit(5).execute()
        
    if not res.data:
        state["messages"].append(f"This doctor has no open slots. Would you like to book another doctor in {state['department']}?")
        state["intent"] = "direct_booking_department"
        state["selected_doctor_id"] = None
    else:
        state["available_slots"] = res.data
        state["messages"].append("SLOTS_OFFERED")
    return state

def node_fetch_slots(state: TriageState) -> TriageState:
    supabase = get_supabase()
    res = supabase.table("clinician_slot")\
        .select("id, start_time, doctor_id, doctor!inner(name, rating, avg_consult_min, specialty!inner(name))")\
        .eq("status", "open")\
        .eq("doctor.specialty.name", state["department"])\
        .order("doctor.rating", desc=True)\
        .order("doctor.avg_consult_min", desc=False)\
        .limit(3).execute()
        
    if not res.data:
        state["messages"].append(f"No available slots in {state['department']} right now.")
        state["available_slots"] = []
    else:
        state["available_slots"] = res.data
        state["messages"].append("SLOTS_OFFERED")
    return state

def node_extract_symptoms(state: TriageState) -> TriageState:
    text = state["messages"][-1]
    new_symptoms = dummy_ner(text)
    curr = state.get("present_symptoms", [])
    state["present_symptoms"] = list(set(curr + new_symptoms))
    return state

def node_next_question(state: TriageState) -> TriageState:
    kg = get_kg()
    next_symptom = kg.rank_next_questions(state["present_symptoms"], state.get("asked_symptoms", []))
    
    if next_symptom:
        state["asked_symptoms"] = state.get("asked_symptoms", []) + [next_symptom]
        
        # Here we would normally rewrite this using ChatOllama and MedDialog RAG.
        # But we don't have Ollama running during the build stage.
        rag = get_rag_engine()
        rag_examples = rag.query_conversations("patient symptoms", k=2)
        
        state["messages"].append(f"QUESTION: Do you have symptom {next_symptom}?")
    else:
        # Fallback to classify if we run out of symptoms to ask
        state["messages"].append("SYSTEM_FALLBACK: Proceed to classification.")
    return state

def node_classify(state: TriageState) -> TriageState:
    # Dummy classification (we would use the XGBoost model here in a real run, 
    # but the environment might not have it loaded fully in the web process right away)
    import os, pickle, numpy as np
    
    # Try to load XGBoost if it exists, otherwise use mock
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../model"))
    xgb_path = os.path.join(model_dir, "xgb_model.json")
    
    if os.path.exists(xgb_path):
        import xgboost as xgb
        clf = xgb.XGBClassifier()
        clf.load_model(xgb_path)
        
        with open(os.path.join(model_dir, "mlb.pkl"), "rb") as f:
            mlb = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
            
        import scipy.sparse as sp
        evidence_matrix = mlb.transform([state["present_symptoms"]])
        # Fetch actual patient demographics for prediction
        supabase = get_supabase()
        patient_res = supabase.table("patient").select("age, gender").eq("id", state["patient_id"]).execute()
        
        if patient_res.data:
            age = patient_res.data[0].get("age", 30)
            gender_str = patient_res.data[0].get("gender", "male")
            # In DDXPlus, often F=0, M=1. We map accordingly.
            sex = 0 if gender_str.lower() == "female" else 1
        else:
            age, sex = 30, 1
            
        age_sex_matrix = sp.csr_matrix([[age, sex]]) 
        X = sp.hstack([age_sex_matrix, evidence_matrix])
        
        probs = clf.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
        pred_condition = le.inverse_transform([max_idx])[0]
        
        kg = get_kg()
        severity = kg.get_condition_severity(pred_condition)
        
        state["final_diagnosis"] = pred_condition
        state["confidence"] = confidence
        state["triage_level"] = severity
    else:
        state["final_diagnosis"] = "General Condition"
        state["confidence"] = 0.8
        state["triage_level"] = 3
        
    if state["confidence"] < 0.3:
        state["triage_level"] = min(state["triage_level"], 3)
        state["final_diagnosis"] = "Uncertain Diagnosis"
        state["department"] = "General Medicine / Internal Medicine"
    else:
        # Hardcode department to General Medicine for prototype
        state["department"] = "General Medicine / Internal Medicine"
        
    return state

def node_explain(state: TriageState) -> TriageState:
    rag = get_rag_engine()
    rag_chunks = rag.query_medquad(state["final_diagnosis"])
    
    explanation = f"DIAGNOSIS_EXPLANATION: Based on your symptoms, this might be {state['final_diagnosis']}."
    
    supabase = get_supabase()
    try:
        supabase.table("chat_session").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "final_diagnosis": state["final_diagnosis"],
            "department": state["department"],
            "triage_level": state["triage_level"],
            "confidence": state["confidence"],
            "triage_summary": explanation,
        }).eq("session_id", state["session_id"]).execute()
    except Exception as e:
        supabase.table("audit_log").insert({
            "event": "chat_session_persist_failed",
            "metadata": {"error": str(e), "session_id": state["session_id"]}
        }).execute()
        
    state["messages"].append(explanation)
    return state

def node_prompt_booking(state: TriageState) -> TriageState:
    state["messages"].append("PROMPT_BOOKING: Would you like to book an appointment with this department?")
    return state

def node_handle_booking(state: TriageState) -> TriageState:
    text = state["messages"][-1].lower()
    affirmative = ["yes", "yeah", "sure", "ok", "book"]
    if any(x in text for x in affirmative):
        state["booking_intent"] = True
    else:
        state["booking_intent"] = False
        state["messages"].append("Okay, please let me know if you need anything else.")
    return state

def node_confirm_slot(state: TriageState) -> TriageState:
    supabase = get_supabase()
    # Find slot to confirm - this would be based on user selection, picking the first for the stub
    # Assuming frontend sent selected_slot_id
    slot_id = state.get("selected_slot_id")
    if not slot_id and state.get("available_slots"):
        slot_id = state["available_slots"][0]["id"]
        
    if not slot_id:
        return state
        
    try:
        triage_level = state.get("triage_level") or 5
        confidence = state.get("confidence")
        
        res = supabase.rpc('book_slot', {
            'p_slot_id': slot_id,
            'p_patient_id': state["patient_id"],
            'p_chat_session_id': state["session_id"],
            'p_department': state["department"],
            'p_triage_level': triage_level,
            'p_confidence': confidence
        }).execute()
        
        # Direct booking chat session upsert
        if state.get("intent") != "symptom_triage":
            supabase.table("chat_session").update({
                "department": state["department"],
                "triage_level": triage_level,
                "confidence": confidence,
                "final_diagnosis": "Patient-requested direct booking",
                "status": "completed"
            }).eq("session_id", state["session_id"]).execute()
            
        state["payment_status"] = "pending"
        state["messages"].append(f"SLOT_CONFIRMED: {slot_id}")
    except Exception as e:
        if "SLOT_NOT_AVAILABLE" in str(e):
            state["messages"].append("That slot was just taken. Let's find another.")
            state["selected_slot_id"] = None
            if state["intent"] == "direct_booking_doctor":
                state["available_slots"] = None # force refetch
            else:
                state["available_slots"] = None # force refetch
                
    return state

async def node_process_payment(state: TriageState) -> TriageState:
    text = state["messages"][-1].lower()
    if "pay" in text or state.get("payment_status") == "pending":
        await asyncio.sleep(1.5) # simulate stripe
        supabase = get_supabase()
        
        # Find appointment
        res = supabase.table("appointment").select("id").eq("chat_session_id", state["session_id"]).execute()
        if res.data:
            appt_id = res.data[0]["id"]
            
            import uuid
            supabase.table("payment").insert({
                "appointment_id": appt_id,
                "stripe_intent": f"pi_{uuid.uuid4().hex[:12]}",
                "status": "succeeded",
                "amount_paisa": 150000 # 1500 INR
            }).execute()
            
            state["payment_status"] = "succeeded"
            state["messages"].append("PAYMENT_SUCCESS: Appointment confirmed.")
    return state

# --- Routing ---

def route_entry(state: TriageState):
    if state.get("is_emergency"): return "node_emergency_response"
    if not state.get("intent"): return "node_detect_intent"
    
    intent = state["intent"]
    if intent == "direct_booking_department" and not state.get("department"):
        return "node_prompt_department_choice"
    if intent == "direct_booking_doctor" and not state.get("available_slots"):
        return "node_fetch_slots_for_doctor"
    if intent == "direct_booking_department" and state.get("department") and not state.get("available_slots"):
        return "node_fetch_slots"
        
    if state.get("payment_status") == "pending": return "node_process_payment"
    if state.get("selected_slot_id") and state.get("payment_status") != "succeeded":
        return "node_confirm_slot"
        
    if state.get("available_slots") is not None and state.get("booking_intent") is None and not state.get("selected_slot_id"):
        return "node_handle_booking"
        
    if intent == "symptom_triage" and state.get("booking_intent") == True and not state.get("available_slots"):
        return "node_fetch_slots"
        
    if intent == "symptom_triage" and state.get("final_diagnosis") and state.get("booking_intent") is None and not state.get("available_slots"):
        return "node_prompt_booking"
        
    return "node_extract_symptoms"

def route_clinical_loop(state: TriageState):
    if len(state.get("asked_symptoms", [])) >= 5 or len(state.get("present_symptoms", [])) >= 3:
        return "node_classify"
    return "node_next_question"

# --- Graph Definition ---

def build_graph():
    builder = StateGraph(TriageState)
    
    # Add nodes
    builder.add_node("node_emergency_check", node_emergency_check)
    builder.add_node("node_emergency_response", node_emergency_response)
    builder.add_node("node_detect_intent", node_detect_intent)
    builder.add_node("node_prompt_department_choice", node_prompt_department_choice)
    builder.add_node("node_fetch_slots_for_doctor", node_fetch_slots_for_doctor)
    builder.add_node("node_fetch_slots", node_fetch_slots)
    builder.add_node("node_extract_symptoms", node_extract_symptoms)
    builder.add_node("node_next_question", node_next_question)
    builder.add_node("node_classify", node_classify)
    builder.add_node("node_explain", node_explain)
    builder.add_node("node_prompt_booking", node_prompt_booking)
    builder.add_node("node_handle_booking", node_handle_booking)
    builder.add_node("node_confirm_slot", node_confirm_slot)
    builder.add_node("node_process_payment", node_process_payment)
    
    # Setup Edges
    builder.set_entry_point("node_emergency_check")
    
    builder.add_conditional_edges(
        "node_emergency_check",
        route_entry,
        {
            "node_emergency_response": "node_emergency_response",
            "node_detect_intent": "node_detect_intent",
            "node_prompt_department_choice": "node_prompt_department_choice",
            "node_fetch_slots_for_doctor": "node_fetch_slots_for_doctor",
            "node_fetch_slots": "node_fetch_slots",
            "node_process_payment": "node_process_payment",
            "node_confirm_slot": "node_confirm_slot",
            "node_handle_booking": "node_handle_booking",
            "node_prompt_booking": "node_prompt_booking",
            "node_extract_symptoms": "node_extract_symptoms",
        }
    )
    
    builder.add_edge("node_detect_intent", END)
    builder.add_edge("node_prompt_department_choice", END)
    builder.add_edge("node_fetch_slots_for_doctor", END)
    builder.add_edge("node_fetch_slots", END)
    
    builder.add_conditional_edges(
        "node_extract_symptoms",
        route_clinical_loop,
        {
            "node_classify": "node_classify",
            "node_next_question": "node_next_question"
        }
    )
    
    builder.add_edge("node_next_question", END)
    builder.add_edge("node_classify", "node_explain")
    builder.add_edge("node_explain", "node_prompt_booking")
    builder.add_edge("node_prompt_booking", END)
    builder.add_edge("node_handle_booking", END)
    builder.add_edge("node_confirm_slot", "node_process_payment")
    builder.add_edge("node_process_payment", END)
    builder.add_edge("node_emergency_response", END)
    
    return builder

# Compile graph
graph_builder = build_graph().compile()
