import os
import json
import asyncio
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from ..db.supabase_client import get_supabase
from .kg import get_kg
from .rag import get_rag_engine
from .unified_retrieval import get_unified_retriever
from .ner_symptom_extractor import get_biomedical_ner

# Department Synonyms
DEPARTMENT_SYNONYMS = {
    "skin": "Dermatology", "heart": "Cardiology", "child": "Pediatrics",
    "kids": "Pediatrics", "bone": "Orthopedics", "joint": "Orthopedics",
    "stomach": "Gastroenterology", "digestive": "Gastroenterology",
    "brain": "Neurology", "nerve": "Neurology", "mental": "Psychiatry",
    "lung": "Respiratory", "breathing": "Respiratory",
}
BOOKING_TRIGGER_PHRASES = ["book", "appointment", "schedule", "see a doctor", "see dr", "consult"]

# Emergency detection - conservative rules
STANDALONE_EMERGENCY = {
    "loss of consciousness", "unconscious", "passed out",
    "cannot breathe", "can't breathe", "unable to breathe",
    "severe bleeding", "hemorrhage", "bleeding heavily",
    "suicidal thoughts", "want to kill myself", "self-harm intent"
}

COMBINATION_TRIGGERS = [
    ({"chest pain", "shortness of breath", "jaw pain", "arm pain", "radiating pain"}, 2),
    ({"facial droop", "slurred speech", "one-sided weakness"}, 2),
    ({"high fever", "stiff neck", "confusion"}, 2),
]

def evaluate_red_flags(symptoms: List[str], text: str) -> bool:
    text_lower = text.lower()
    if any(trigger in text_lower for trigger in STANDALONE_EMERGENCY):
        return True
    symptom_set = set(s.lower() for s in symptoms)
    for required, min_count in COMBINATION_TRIGGERS:
        if len(symptom_set & required) >= min_count:
            return True
    return False

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
    rag_chunks: Optional[List[str]]
    latencies: Optional[dict]

# --- Nodes ---

def node_emergency_check(state: TriageState) -> TriageState:
    text = state["messages"][-1]
    ner = get_biomedical_ner()
    new_symptoms = ner.extract_symptoms(text)
    symptoms = state.get("present_symptoms", []) + new_symptoms
    
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

def route_after_intent(state: TriageState):
    """Route after intent detection based on the detected intent"""
    intent = state.get("intent")
    
    if intent == "direct_booking_doctor":
        return "node_fetch_slots_for_doctor"
    elif intent == "direct_booking_department":
        if state.get("department"):
            return "node_fetch_slots"
        else:
            return "node_prompt_department_choice"
    elif intent == "symptom_triage":
        return "node_extract_symptoms"
    return END

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
    ner = get_biomedical_ner()
    new_symptoms = ner.extract_symptoms(text)
    curr = state.get("present_symptoms", [])
    state["present_symptoms"] = list(set(curr + new_symptoms))
    return state

def ask_ollama(system_prompt: str, user_prompt: str) -> str:
    import os
    import logging
    import requests
    from urllib.parse import urljoin
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        # Test connection first
        health_url = urljoin(ollama_url, "/api/tags")
        requests.get(health_url, timeout=5)
        logging.info(f"Connecting to Ollama at {ollama_url}...")
        
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
        
        chat = ChatOllama(model="llama3.2", base_url=ollama_url, temperature=0.7)
        res = chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return res.content.strip()
    except requests.exceptions.ConnectionError:
        logging.warning(f"Ollama not available at {ollama_url}. Using fallback response.")
        return None
    except Exception as e:
        logging.warning(f"Ollama error: {e}. Using fallback response.")
        return None

def node_next_question(state: TriageState) -> TriageState:
    kg = get_kg()
    next_questions = kg.rank_next_questions(state["present_symptoms"], state.get("asked_symptoms", []))
    
    if next_questions and len(next_questions) > 0:
        next_symptom_id, score = next_questions[0]  # Get top question
        state["asked_symptoms"] = state.get("asked_symptoms", []) + [next_symptom_id]
        
        # Get unified retriever for few-shot examples from Conversations
        retriever = get_unified_retriever()
        
        # Get present symptoms as context for semantic search
        present_symptoms_text = " ".join(state.get("present_symptoms", []))
        
        import time
        t0 = time.time()
        # Retrieve top 3 few-shot examples (pre-extracted doctor turns) from Conversations
        # based on KG-determined question and symptom context
        few_shot_examples = retriever.get_fewshot_examples(
            query=str(next_symptom_id),
            symptom=present_symptoms_text,
            num_examples=3
        )
        t_rag = int((time.time() - t0) * 1000)
        state["rag_chunks"] = few_shot_examples
        
        system_prompt = (
            "You are a friendly, professional AI medical assistant. "
            "Ask the user if they are experiencing a specific symptom. Keep it brief and conversational (1-2 sentences). "
            "Do not give medical advice. Just ask the question. "
            "Use the examples below as reference for how similar questions are phrased by medical professionals, "
            "but do NOT copy them directly - generate your own natural question based on the pattern."
        )
        
        user_prompt = f"The symptom to ask about is: {next_symptom_id}."
        
        # Add few-shot examples from actual doctor conversations
        if few_shot_examples:
            user_prompt += "\n\nHere are examples of how medical professionals ask similar questions:\n"
            for i, example in enumerate(few_shot_examples, 1):
                user_prompt += f"{i}. {example}\n"
            user_prompt += "\nGenerate a similar but unique question for this symptom:"
            
        t0_llm = time.time()
        ollama_response = ask_ollama(system_prompt, user_prompt)
        t_llm = int((time.time() - t0_llm) * 1000)
        
        state["latencies"] = {"RAG Retrieval": t_rag, "LLM Generation": t_llm}
        
        if ollama_response and ollama_response.strip():
            state["messages"].append(f"QUESTION: {ollama_response}")
        else:
            # Fallback question when Ollama is not available
            state["messages"].append(f"QUESTION: Do you have symptom {next_symptom_id}?")
    else:
        # Fallback to classify if we run out of symptoms to ask
        state["messages"].append("SYSTEM_FALLBACK: Proceeding to analysis.")
    return state

def node_classify(state: TriageState) -> TriageState:
    import os
    import pickle
    import numpy as np
    from collections import Counter
    
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
        
        # --- Department Prediction ---
        # 1. Try KG specialty from predicted condition
        kg = get_kg()
        department = None
        condition_id = None
        for cid, cinfo in kg.conditions.items():
            if cinfo.get("name", "").lower() == pred_condition.lower():
                condition_id = cid
                break
        
        if condition_id:
            department = kg.get_condition_specialty(condition_id)
        
        # 2. Fallback: Use symptom-to-department mapping
        if not department:
            from .symptom_to_dept import predict_department_from_symptoms
            department = predict_department_from_symptoms(state["present_symptoms"])
    else:
        state["final_diagnosis"] = "General Condition"
        state["confidence"] = 0.8
        state["triage_level"] = 3
        
        # Fallback department prediction
        from .symptom_to_dept import predict_department_from_symptoms
        department = predict_department_from_symptoms(state["present_symptoms"])
    
    # Confidence flooring
    confidence_floor = float(os.getenv("CONFIDENCE_FLOOR", "0.3"))
    if state["confidence"] < confidence_floor:
        state["triage_level"] = min(state["triage_level"], 3)
        state["final_diagnosis"] = "Uncertain Diagnosis"
        department = "General Medicine / Internal Medicine"
    
    state["department"] = department
    return state

def node_explain(state: TriageState) -> TriageState:
    import time
    t0 = time.time()
    rag = get_rag_engine()
    rag_chunks = rag.query_medquad(state["final_diagnosis"])
    t_rag = int((time.time() - t0) * 1000)
    state["rag_chunks"] = rag_chunks
    
    system_prompt = (
        "You are a friendly, professional AI medical assistant. "
        "Explain to the patient that based on their symptoms, they might have a specific condition. "
        "Keep it empathetic and reassuring (2-3 sentences). "
        "Always clarify that this is not a definitive medical diagnosis and they should consult the doctor."
    )
    user_prompt = f"The condition is: {state['final_diagnosis']}."
    
    if rag_chunks:
        user_prompt += f"\nHere is some medical context to help you explain it accurately: '{rag_chunks[0]}'"
        
    t0_llm = time.time()
    ollama_response = ask_ollama(system_prompt, user_prompt)
    t_llm = int((time.time() - t0_llm) * 1000)
    
    state["latencies"] = {"MedQuAD RAG": t_rag, "LLM Generation": t_llm}
    
    if ollama_response and ollama_response.strip():
        explanation = f"DIAGNOSIS_EXPLANATION: {ollama_response}"
    else:
        explanation = f"DIAGNOSIS_EXPLANATION: Based on your symptoms, this might be {state['final_diagnosis']}. Please consult with a healthcare professional for a proper diagnosis."
    
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
    
    builder.add_conditional_edges(
        "node_detect_intent",
        route_after_intent,
        {
            "node_fetch_slots_for_doctor": "node_fetch_slots_for_doctor",
            "node_prompt_department_choice": "node_prompt_department_choice",
            "node_fetch_slots": "node_fetch_slots",
            "node_extract_symptoms": "node_extract_symptoms",
        }
    )
    
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

# Compile graph with SQLite checkpointer for session persistence
checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")
graph_builder = build_graph().compile(checkpointer=checkpointer)
