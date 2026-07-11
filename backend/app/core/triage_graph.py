import os
import json
import pickle
import sqlite3
import asyncio
import re
import logging
import xgboost as xgb # type: ignore
from typing import Annotated, Dict, List, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field # type: ignore

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # type: ignore
from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore

from .knowledge_graph import KnowledgeGraph # type: ignore
from .departments import get_department_for_condition # type: ignore
from .db import get_supabase # type: ignore
from .prompts import EMERGENCY_SCREENING_PROMPT, SLOT_EXTRACTION_PROMPT, SYMPTOM_MAPPING_PROMPT, RAG_NEXT_QUESTION_PROMPT, CLASSIFICATION_EXPLANATION_PROMPT # type: ignore
from langchain_ollama import ChatOllama # type: ignore
from pathlib import Path

logger = logging.getLogger("triage_graph")

# Paths to models and data
BASE_DIR = Path(__file__).parent.parent.parent.parent / "ai_engine" / "ml_training"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "DDXPlus"

# Initialize Knowledge Graph and ML Models lazily to avoid startup overhead if not needed immediately
_kg = None
_xgb_model = None
_mlb = None
_label_encoder = None
_medquad_index = None
_conversations_index = None
_embeddings_model = None
_ner_pipeline = None
_rag_load_attempted = False
_rag_health: Dict[str, Any] = {
    "embeddings_loaded": False,
    "medquad_loaded": False,
    "conversations_loaded": False,
    "issues": []
}
_ner_health = {
    "loaded": False,
    "error": None
}

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

def load_rag_models() -> Dict[str, Any]:
    global _medquad_index, _conversations_index, _embeddings_model, _rag_load_attempted, _rag_health
    if _rag_load_attempted:
        return dict(_rag_health)

    _rag_load_attempted = True
    _rag_health = {
        "embeddings_loaded": False,
        "medquad_loaded": False,
        "conversations_loaded": False,
        "issues": []
    }

    try:
        from langchain_community.vectorstores import FAISS # type: ignore
        from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
    except Exception as exc:
        _rag_health["issues"].append(f"RAG dependency import failed: {exc}")
        logger.error("Failed to import RAG dependencies", exc_info=True)
        return dict(_rag_health)

    try:
        _embeddings_model = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
        _rag_health["embeddings_loaded"] = True
    except Exception as exc:
        _rag_health["issues"].append(f"Embeddings model load failed: {exc}")
        logger.error("Failed to load embeddings model", exc_info=True)
        return dict(_rag_health)

    faiss_dir = BASE_DIR.parent / "faiss"
    if not faiss_dir.exists():
        _rag_health["issues"].append(f"FAISS directory missing: {faiss_dir}")
        logger.warning("FAISS directory is missing at %s", faiss_dir)
        return dict(_rag_health)

    medquad_dir = faiss_dir / "medquad"
    conversations_dir = faiss_dir / "conversations"

    if medquad_dir.exists():
        try:
            _medquad_index = FAISS.load_local(str(medquad_dir), _embeddings_model, allow_dangerous_deserialization=False)
            _rag_health["medquad_loaded"] = True
        except Exception as exc:
            _rag_health["issues"].append(f"MedQuAD index load failed: {exc}")
            logger.error("Failed to load MedQuAD FAISS index", exc_info=True)
    else:
        _rag_health["issues"].append(f"MedQuAD index missing: {medquad_dir}")

    if conversations_dir.exists():
        try:
            _conversations_index = FAISS.load_local(str(conversations_dir), _embeddings_model, allow_dangerous_deserialization=False)
            _rag_health["conversations_loaded"] = True
        except Exception as exc:
            _rag_health["issues"].append(f"MedDialog index load failed: {exc}")
            logger.error("Failed to load MedDialog FAISS index", exc_info=True)
    else:
        _rag_health["issues"].append(f"MedDialog index missing: {conversations_dir}")

    if not (_rag_health["medquad_loaded"] or _rag_health["conversations_loaded"]):
        logger.warning("RAG indices unavailable. issues=%s", _rag_health["issues"])

    return dict(_rag_health)

def load_ner_model() -> Dict[str, Any]:
    global _ner_pipeline, _ner_health
    if _ner_pipeline is not None:
        return dict(_ner_health)

    try:
        from transformers import pipeline # type: ignore
        logger.info("Loading biomedical NER model d4data/biomedical-ner-all")
        _ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        _ner_health = {"loaded": True, "error": None}
    except Exception as exc:
        _ner_health = {"loaded": False, "error": str(exc)}
        logger.error("Failed to load biomedical NER model", exc_info=True)
    return dict(_ner_health)

# LLM for extraction and summarization
# Assuming Ollama is running locally with Llama3.2
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

class TriageState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    patient_id: Optional[str]
    age: int
    gender: str
    present_symptoms: List[str]  # List of DDXPlus E_* codes
    absent_symptoms: List[str]   # List of DDXPlus E_* codes
    final_diagnosis: Optional[str]
    department: Optional[str]
    triage_summary: Optional[str]
    question_count: int
    confidence: float
    urgency: int
    booking_intent: Optional[bool]
    available_slots: Optional[List[Dict[str, Any]]]
    selected_slot_id: Optional[str]
    payment_status: Optional[str]
    rag_chunks: Optional[List[str]]
    latencies: Optional[Dict[str, float]]
    rag_status: Optional[Dict[str, Any]]
    model_health: Optional[Dict[str, Any]]

# --- NODES ---

async def node_extract_symptoms(state: TriageState) -> Dict:
    """Uses Medical NER to extract symptoms and maps them to DDXPlus evidence codes."""
    kg = get_kg()
    ner_health = load_ner_model()
    last_msg = state["messages"][-1].content.lower()
    
    extracted_present = []
    
    if _ner_pipeline is not None:
        try:
            entities = await asyncio.to_thread(_ner_pipeline, last_msg)
            # Filter for symptoms and diseases
            for ent in entities:
                if ent['entity_group'] in ['Sign_symptom', 'Disease_disorder', 'Detailed_description']:
                    term = ent['word'].lower()
                    # Map to E_ codes
                    for e_id, e_data in kg.evidences.items():
                        if term in e_data.get("question_en", "").lower() or term in e_data.get("name", "").lower():
                            if e_id not in state["present_symptoms"] and e_id not in extracted_present:
                                extracted_present.append(e_id)
                            break
        except Exception as exc:
            logger.error("NER extraction failed", exc_info=True)
            ner_health = {"loaded": False, "error": str(exc)}
        
    return {
        "present_symptoms": state["present_symptoms"] + extracted_present,
        "model_health": {"ner": ner_health}
    }



def node_decide_next(state: TriageState) -> str:
    if state.get("question_count", 0) >= 5 or len(state["present_symptoms"]) >= 3:
        return "classify"
    return "next_question"

async def node_next_question(state: TriageState) -> Dict:
    import time
    start_time = time.time()
    
    kg = get_kg()
    rag_health = load_rag_models()
    
    top_qs = kg.rank_next_questions(state["present_symptoms"], state["absent_symptoms"], top_k=1)
    if not top_qs:
        base_question = "Can you describe your symptoms in more detail?"
    else:
        base_question = kg.get_question_for_evidence(top_qs[0])
        
    faiss_start = time.time()
    # RAG Conversation Aiding
    rag_examples = ""
    rag_chunks = state.get("rag_chunks", []) or []
    if _conversations_index is not None and top_qs:
        # Search the conversation index using the target question or current symptoms
        search_query = base_question + " " + " ".join(state["present_symptoms"])
        docs = await asyncio.to_thread(_conversations_index.similarity_search, search_query, k=2)
        for doc in docs:
            chunk = doc.page_content
            rag_examples += f"- {chunk}\n"
            rag_chunks.append(f"[MedDialog]: {chunk}")
            
    if not rag_examples:
        rag_examples = "(No relevant conversation examples found - ask naturally)"
    faiss_end = time.time()

    prompt = RAG_NEXT_QUESTION_PROMPT.format(
        target_question=base_question,
        rag_conversation_examples=rag_examples
    )
    
    llm_start = time.time()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        q = response.content
    except Exception:
        q = base_question
    llm_end = time.time()
        
    latencies = state.get("latencies", {}) or {}
    latencies["next_question_faiss"] = round((faiss_end - faiss_start) * 1000, 2)
    latencies["next_question_llm"] = round((llm_end - llm_start) * 1000, 2)
    
    return {
        "messages": [AIMessage(content=q)],
        "question_count": state.get("question_count", 0) + 1,
        "rag_chunks": rag_chunks,
        "latencies": latencies,
        "rag_status": rag_health
    }

async def node_classify(state: TriageState) -> Dict:
    load_ml_models()
    import scipy.sparse as sp # type: ignore
    
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
        "department": dept,
        "confidence": round(confidence * 100, 1),
        "urgency": urgency_score
    }

async def node_explain(state: TriageState) -> Dict:
    import time
    latencies = state.get("latencies", {}) or {}
    rag_chunks = state.get("rag_chunks", []) or []
    
    rag_health = load_rag_models()
    dept = state["department"]
    final_diagnosis = state["final_diagnosis"]
    
    faiss_start = time.time()
    # RAG Clinical Facts
    rag_facts = ""
    if _medquad_index is not None and final_diagnosis:
        docs = await asyncio.to_thread(_medquad_index.similarity_search, final_diagnosis, k=3)
        # Cap the length so it doesn't blow up the context window
        for doc in docs:
            chunk = doc.page_content[:300]
            rag_facts += f"- {chunk}...\n"
            rag_chunks.append(f"[MedQuAD]: {chunk}...")
            
    if not rag_facts:
        rag_facts = "(No specific clinical facts found - rely on general knowledge)"
    faiss_end = time.time()

    symptom_text = ", ".join(state["present_symptoms"]) if state.get("present_symptoms") else "None reported"

    prompt = CLASSIFICATION_EXPLANATION_PROMPT.format(
        department=dept,
        confidence_pct=state.get("confidence", 0.0),
        urgency=state.get("urgency", 3),
        symptom_summary=f"Patient reports these symptoms: {symptom_text}",
        rag_block=rag_facts
    )
    
    llm_start = time.time()
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        msg = response.content
    except Exception:
        msg = f"Based on your symptoms, the likely condition is **{final_diagnosis}**. I recommend consulting the **{dept}** department."
    if not rag_health.get("medquad_loaded"):
        msg = f"{msg}\n\nNote: Clinical retrieval is currently in degraded mode (MedQuAD index unavailable)."
    llm_end = time.time()
        
    summary = f"Predicted {final_diagnosis}, routed to {dept}"
    latencies["explain_faiss"] = round((faiss_end - faiss_start) * 1000, 2)
    latencies["explain_llm"] = round((llm_end - llm_start) * 1000, 2)
        
    # Route to Supabase DB to store triage session
    supabase = get_supabase()
    if supabase and state.get("patient_id"):
        persisted = False
        try:
            await asyncio.to_thread(supabase.table("chat_session").insert({
                "patient_id": state["patient_id"],
                "summary": summary,
                "recommended_department_id": None, # Ideally we would look up the ID from the department table
                "status": "completed"
            }).execute)
            persisted = True
        except Exception as exc:
            logger.warning("chat_session write failed; falling back to audit_log", exc_info=True)
            try:
                await asyncio.to_thread(supabase.table("audit_log").insert({
                    "event": "triage_session_completed",
                    "metadata": {
                        "patient_id": state["patient_id"],
                        "summary": summary,
                        "department": dept
                    }
                }).execute)
                persisted = True
            except Exception:
                logger.error("Failed to persist triage summary in both chat_session and audit_log", exc_info=True)
        if not persisted:
            logger.error("Triage summary persistence failed for patient_id=%s", state["patient_id"])
            
    return {
        "messages": [AIMessage(content=msg)],
        "triage_summary": summary,
        "department": dept,
        "rag_chunks": rag_chunks,
        "latencies": latencies,
        "rag_status": load_rag_models()
    }

# --- BOOKING & PAYMENT NODES ---

async def node_prompt_booking(state: TriageState) -> Dict:
    dept = state.get("department", "the appropriate")
    msg = f"Would you like me to find an available doctor in the **{dept}** department for you? (Please reply 'Yes' or 'No')"
    return {"messages": [AIMessage(content=msg)]}

async def node_handle_booking(state: TriageState) -> Dict:
    last_msg = state["messages"][-1].content.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", last_msg)
    tokens = {t for t in normalized.split() if t}
    negative_tokens = {"no", "not", "dont", "don't", "later", "maybe", "cancel"}
    affirmative_tokens = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "book", "booking"}
    has_booking_phrase = "book appointment" in last_msg or "book a slot" in last_msg
    if not (tokens & negative_tokens) and ((tokens & affirmative_tokens) or has_booking_phrase):
        return {"booking_intent": True}
    return {"booking_intent": False}

def node_decide_booking(state: TriageState) -> str:
    if state.get("booking_intent"):
        return "fetch_slots"
    return END

async def node_fetch_slots(state: TriageState) -> Dict:
    supabase = get_supabase()
    slots_list = []
    msg = "I'm sorry, I couldn't find any available slots right now."
    if supabase:
        try:
            res = await asyncio.to_thread(supabase.table("clinician_slot").select("id, start_time, doctor(name, specialty(name))").eq("status", "open").limit(50).execute)
            if res.data:
                dept = state.get("department")
                filtered_slots = []
                for s in res.data:
                    doc = s.get("doctor")
                    if doc and doc.get("specialty") and doc["specialty"].get("name") == dept:
                        filtered_slots.append(s)
                slots_list = filtered_slots[:3]
                msg = "Here are some available slots:\n"
                for i, s in enumerate(slots_list):
                    dr_name = s['doctor']['name'] if s.get('doctor') else "Unknown Doctor"
                    time_str = s['start_time'][:16].replace("T", " at ")
                    msg += f"{i+1}. Dr. {dr_name} - {time_str}\n"
                msg += "\nPlease reply with the **number** of the slot you want."
        except Exception as e:
            print("Failed to fetch slots:", e)
    return {"available_slots": slots_list, "messages": [AIMessage(content=msg)]}

async def node_confirm_slot(state: TriageState) -> Dict:
    last_msg = state["messages"][-1].content.lower()
    slots = state.get("available_slots", [])
    selected_slot = None
    
    for i, s in enumerate(slots):
        if str(i+1) in last_msg:
            selected_slot = s
            break
            
    if not selected_slot:
        return {"messages": [AIMessage(content="I didn't catch that. Booking cancelled.")], "booking_intent": False}
        
    slot_id = selected_slot["id"]
    supabase = get_supabase()
    if supabase and state.get("patient_id"):
        try:
            await asyncio.to_thread(supabase.table("clinician_slot").update({"status": "held"}).eq("id", slot_id).execute)
            res = await asyncio.to_thread(supabase.table("appointment").insert({
                "patient_id": state["patient_id"],
                "slot_id": slot_id,
                "department": state.get("department", "Unknown"),
                "triage_level": state.get("urgency", 3),
                "confidence": state.get("confidence", 0.0),
                "status": "pending_slot"
            }).execute)
        except Exception as e:
            print("Failed to book slot:", e)
            
    msg = "Great! Your slot is held. For this consultation, the fee is $50. Please type **PAY** to confirm your appointment and process the payment."
    return {"selected_slot_id": slot_id, "payment_status": "pending", "messages": [AIMessage(content=msg)]}

async def node_process_payment(state: TriageState) -> Dict:
    import asyncio, uuid
    last_msg = state["messages"][-1].content.lower()
    if "pay" in last_msg:
        await asyncio.sleep(1.5)  # Simulate payment processing delay
        intent_id = f"pi_{uuid.uuid4().hex[:20]}"
        msg = f"✅ Payment successful! (Intent ID: {intent_id})\nYour appointment is now fully scheduled. Thank you for using TriagePlus."
        status = "succeeded"
        # In prod, we'd update appointment status to scheduled here
    else:
        msg = "Payment pending. Please type **PAY** to complete the payment and finalize the booking."
        status = "pending"
        
    return {"payment_status": status, "messages": [AIMessage(content=msg)]}

def node_decide_payment(state: TriageState) -> str:
    if state.get("payment_status") == "succeeded":
        return END
    return "process_payment"

def route_entry(state: TriageState) -> str:
    """Routes the conversation to the correct sub-flow based on the current state."""
    # If they are pending payment, keep them in the payment flow
    if state.get("payment_status") == "pending":
        return "process_payment"
        
    # If they are selecting a slot, confirm the slot
    if state.get("booking_intent") is True and not state.get("selected_slot_id") and state.get("available_slots"):
        return "confirm_slot"
        
    # If they haven't answered the booking prompt yet
    if state.get("booking_intent") is None and state.get("department"):
        return "handle_booking"
        
    # Otherwise, it's a standard triage chat
    return "extract_symptoms"

# --- BUILD GRAPH ---

graph_builder = StateGraph(TriageState)
graph_builder.add_node("extract_symptoms", node_extract_symptoms)

graph_builder.add_node("next_question", node_next_question)
graph_builder.add_node("classify", node_classify)
graph_builder.add_node("explain", node_explain)

graph_builder.add_node("prompt_booking", node_prompt_booking)
graph_builder.add_node("handle_booking", node_handle_booking)
graph_builder.add_node("fetch_slots", node_fetch_slots)
graph_builder.add_node("confirm_slot", node_confirm_slot)
graph_builder.add_node("process_payment", node_process_payment)

# Entry Point
graph_builder.set_conditional_entry_point(
    route_entry,
    {
        "extract_symptoms": "extract_symptoms",
        "handle_booking": "handle_booking",
        "confirm_slot": "confirm_slot",
        "process_payment": "process_payment",
        "fetch_slots": "fetch_slots"
    }
)

# Triage Flow
graph_builder.add_conditional_edges(
    "extract_symptoms",
    node_decide_next,
    {
        "explain": "explain",
        "classify": "classify",
        "next_question": "next_question"
    }
)
graph_builder.add_edge("next_question", END)
graph_builder.add_edge("classify", "explain")
graph_builder.add_edge("explain", "prompt_booking")
graph_builder.add_edge("prompt_booking", END)

# Booking Flow
graph_builder.add_conditional_edges(
    "handle_booking",
    node_decide_booking,
    {
        "fetch_slots": "fetch_slots",
        END: END
    }
)
graph_builder.add_edge("fetch_slots", END)
graph_builder.add_edge("confirm_slot", END)

# Payment Flow
graph_builder.add_conditional_edges(
    "process_payment",
    node_decide_payment,
    {
        "process_payment": END, # Pause and wait for next user message if pending
        END: END
    }
)

# SQLite checkpointer path exported for async usage in API
db_path = BASE_DIR.parent.parent / "backend" / "triage_checkpoints.sqlite"
