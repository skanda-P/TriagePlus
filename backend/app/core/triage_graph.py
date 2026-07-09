import os
import json
import pickle
import sqlite3
import xgboost as xgb
from typing import Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from .knowledge_graph import KnowledgeGraph
from .departments import get_department_for_condition
from .db import get_supabase
from .prompts import EMERGENCY_SCREENING_PROMPT, SLOT_EXTRACTION_PROMPT, SYMPTOM_MAPPING_PROMPT
from langchain_ollama import ChatOllama
from pathlib import Path

# Paths to models and data
BASE_DIR = Path(__file__).parent.parent.parent.parent / "ai_engine" / "ml_training"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "DDXPlus"

# Initialize Knowledge Graph and ML Models lazily to avoid startup overhead if not needed immediately
_kg = None
_xgb_model = None
_mlb = None
_label_encoder = None

def get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(str(DATA_DIR))
    return _kg

def load_ml_models():
    global _xgb_model, _mlb, _label_encoder
    if _xgb_model is None:
        _xgb_model = xgb.XGBClassifier()
        _xgb_model.load_model(MODELS_DIR / "triage_xgb.json")
        with open(MODELS_DIR / "mlb.pkl", "rb") as f:
            _mlb = pickle.load(f)
        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
            _label_encoder = pickle.load(f)

# LLM for extraction and summarization
# Assuming Ollama is running locally with Llama3
llm = ChatOllama(model="llama3", temperature=0)

class TriageState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    patient_id: Optional[str]
    age: int
    gender: str
    present_symptoms: List[str]  # List of DDXPlus E_* codes
    absent_symptoms: List[str]   # List of DDXPlus E_* codes
    is_emergency: bool
    final_diagnosis: Optional[str]
    department: Optional[str]
    triage_summary: Optional[str]
    question_count: int

# --- NODES ---

def node_extract_symptoms(state: TriageState) -> Dict:
    """Uses LLM to map user message to DDXPlus evidence codes."""
    # In a real setup, we provide a prompt with common symptoms to map.
    # For now, we will simulate symptom extraction or do a basic extraction.
    kg = get_kg()
    last_msg = state["messages"][-1].content.lower()
    
    extracted_present = []
    extracted_absent = []
    
    # Very naive extraction logic - in prod, use LLM structured output mapping to kg.evidences
    # For hackathon: LLM prompt to return JSON of mapped symptoms
    prompt = SYMPTOM_MAPPING_PROMPT.format(patient_message=last_msg)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        import re
        json_str = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group())
        else:
            data = {"present": [], "absent": []}
            
        # We map string terms to E_ codes
        for term in data.get("present", []):
            for e_id, e_data in kg.evidences.items():
                if term.lower() in e_data.get("question_en", "").lower() or term.lower() in e_data.get("name", "").lower():
                    extracted_present.append(e_id)
                    break
    except Exception:
        pass
        
    return {
        "present_symptoms": state["present_symptoms"] + extracted_present,
        "absent_symptoms": state["absent_symptoms"] + extracted_absent
    }

def node_emergency_check(state: TriageState) -> Dict:
    """Checks if any present symptoms map to severe conditions (severity 1 or 2)."""
    kg = get_kg()
    is_emerg = False
    
    # 1. T1 Mitigation: Keyword safety net
    last_msg = state["messages"][-1].content.lower()
    emergency_keywords = ["chest pain", "heart attack", "stroke", "can't breathe", "breathing difficulty", "severe bleeding", "unconscious", "suicide", "kill myself"]
    if any(kw in last_msg for kw in emergency_keywords):
        is_emerg = True
        
    # 2. DDXPlus mapping check
    if not is_emerg:
        for cond_id, cond_data in kg.conditions.items():
            if cond_data.get("severity") in [1, 2]:
                cond_symps = cond_data.get("symptoms", {})
                if any(s in cond_symps for s in state["present_symptoms"]):
                    is_emerg = True
                    break
                    
    return {"is_emergency": is_emerg}

def node_decide_next(state: TriageState) -> str:
    if state.get("is_emergency"):
        return "explain"
    if state.get("question_count", 0) >= 5 or len(state["present_symptoms"]) >= 3:
        return "classify"
    return "next_question"

def node_next_question(state: TriageState) -> Dict:
    kg = get_kg()
    top_qs = kg.rank_next_questions(state["present_symptoms"], state["absent_symptoms"], top_k=1)
    if not top_qs:
        # Default question
        q = "Can you describe your symptoms in more detail?"
    else:
        q = kg.get_question_for_evidence(top_qs[0])
        
    return {
        "messages": [AIMessage(content=q)],
        "question_count": state.get("question_count", 0) + 1
    }

def node_classify(state: TriageState) -> Dict:
    load_ml_models()
    import scipy.sparse as sp
    
    # Prepare features: age, sex, and binarized symptoms
    # Sex mapping: M=1, F=0
    sex_val = 1 if state.get("gender", "M") == "M" else 0
    age_val = state.get("age", 30)
    
    # Binarize evidences exactly as trained
    symp_list = state["present_symptoms"]
    X_symp = _mlb.transform([symp_list])
    
    age_sex = [[age_val, sex_val]]
    X = sp.hstack([age_sex, X_symp]).tocsr()
    
    # Predict
    pred_idx = _xgb_model.predict(X)[0]
    pred_proba = _xgb_model.predict_proba(X)[0]
    confidence = float(max(pred_proba))
    
    condition = _label_encoder.inverse_transform([pred_idx])[0]
    dept = get_department_for_condition(condition, age_val, available_departments=None)
    
    # T2 Mitigation: Floor urgency score on low confidence
    # (Since we don't have a strict urgency scale anymore, we use the condition's severity)
    kg = get_kg()
    cond_data = kg.conditions.get(condition, {})
    # Severity 1-5 where 1 is highest urgency
    urgency_score = 6 - cond_data.get("severity", 5) # Map to 1-5 scale (higher = more urgent)
    
    if confidence < 0.3:
        urgency_score = max(urgency_score, 3) # Floor to at least 3 (medium urgency) if unsure
        condition = "Uncertain Diagnosis"
        dept = "General Medicine / Internal Medicine"
    
    return {
        "final_diagnosis": condition,
        "department": dept
    }

def node_explain(state: TriageState) -> Dict:
    if state.get("is_emergency"):
        msg = "🚨 This sounds like a medical emergency. Please visit the nearest Emergency Room or call emergency services immediately."
        summary = "Emergency detected based on severe symptoms."
        dept = "Emergency Medicine"
    else:
        msg = f"Based on your symptoms, the likely condition is **{state['final_diagnosis']}**. I recommend consulting the **{state['department']}** department."
        summary = f"Predicted {state['final_diagnosis']}, routed to {state['department']}"
        dept = state["department"]
        
    # Route to Supabase DB to store triage session
    supabase = get_supabase()
    if supabase and state.get("patient_id"):
        try:
            supabase.table("chat_session").insert({
                "patient_id": state["patient_id"],
                "summary": summary,
                "recommended_department_id": None, # Ideally we would look up the ID from the department table
                "status": "completed"
            }).execute()
        except Exception as e:
            print("Failed to save to Supabase:", e)
            
    return {
        "messages": [AIMessage(content=msg)],
        "triage_summary": summary,
        "department": dept
    }

# --- BUILD GRAPH ---

graph_builder = StateGraph(TriageState)
graph_builder.add_node("extract_symptoms", node_extract_symptoms)
graph_builder.add_node("emergency_check", node_emergency_check)
graph_builder.add_node("next_question", node_next_question)
graph_builder.add_node("classify", node_classify)
graph_builder.add_node("explain", node_explain)

graph_builder.add_edge(START, "extract_symptoms")
graph_builder.add_edge("extract_symptoms", "emergency_check")

graph_builder.add_conditional_edges(
    "emergency_check",
    node_decide_next,
    {
        "explain": "explain",
        "classify": "classify",
        "next_question": "next_question"
    }
)

graph_builder.add_edge("classify", "explain")
graph_builder.add_edge("explain", END)

# SQLite checkpointer
db_path = BASE_DIR.parent / "backend" / "triage_checkpoints.sqlite"
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

triage_app = graph_builder.compile(checkpointer=checkpointer)
