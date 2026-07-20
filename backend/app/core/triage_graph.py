import logging
import os
import threading
from datetime import datetime
from typing import Annotated, List, Optional, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END

from ..db.supabase_client import get_supabase
from .kg import get_kg
from .unified_retrieval import get_unified_retriever
from .ner_symptom_extractor import get_biomedical_ner

logger = logging.getLogger(__name__)


# Department Synonyms — kept identical to seed specialties in Supabase (see
# supabase/migrations/0001_init.sql line 234).
DEPARTMENT_SYNONYMS = {
    "skin": "Dermatology", "heart": "Cardiology", "child": "Pediatrics",
    "kids": "Pediatrics", "bone": "Orthopedics", "joint": "Orthopedics",
    "stomach": "Gastroenterology", "digestive": "Gastroenterology",
    "brain": "Neurology", "nerve": "Neurology", "mental": "Psychiatry",
    "lung": "Respiratory", "breathing": "Respiratory",
}
BOOKING_TRIGGER_PHRASES = ["book", "appointment", "schedule", "see a doctor", "see dr", "consult"]

# Affirmative tokens used to interpret booking/slot replies. Whole-token only
# (previously "ok" matched inside "oklahoma", "book" inside "facebook").
AFFIRMATIVE_TOKENS = {"yes", "yeah", "yup", "sure", "ok", "okay", "book", "please", "confirm"}
NEGATIVE_TOKENS = {"no", "nope", "nah", "cancel", "later", "stop", "false", "not"}

# Internal sentinel markers that the chat.py wire layer should NOT echo to the
# patient UI (they're stage markers, not patient messages).
INTERNAL_SENTINELS = {
    "SLOTS_OFFERED", "PROMPT_BOOKING", "PROMPT_DEPARTMENT", "PROMPT_DEPARTMENT_RETRY",
    "SYSTEM_FALLBACK", "SLOT_CONFIRMED", "PAYMENT_SUCCESS", "EMERGENCY_TRIGGERED",
    "DIAGNOSIS_EXPLANATION", "QUESTION",
}


def _is_sentinel_message(msg: str) -> bool:
    if not isinstance(msg, str):
        return False
    # Match `"SENTINEL: ..."` and bare `"SENTINEL"` shapes.
    stripped = msg.split(":", 1)[0].strip()
    return stripped in INTERNAL_SENTINELS


# ----- Emergency detection -----------------------------------------------------
# Conservative keyword + combo rules with negation handling.

STANDALONE_EMERGENCY = {
    "loss of consciousness", "unconscious", "passed out",
    "cannot breathe", "can't breathe", "unable to breathe",
    "severe bleeding", "hemorrhage", "bleeding heavily",
    "suicidal thoughts", "want to kill myself", "self-harm intent",
}

NEGATION_WORDS = {
    "no", "not", "never", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "haven't", "havent", "hasn't", "hasnt", "without", "absence", "of", "denies",
    "denied", "negative", "for",
}

COMBINATION_TRIGGERS = [
    ({"chest pain", "shortness of breath", "jaw pain", "arm pain", "radiating pain"}, 2),
    ({"facial droop", "slurred speech", "one-sided weakness"}, 2),
    ({"high fever", "stiff neck", "confusion"}, 2),
]


def _tokens(text: str) -> List[str]:
    """Whitespace-split tokens for whole-word matching."""
    return text.lower().split()


def _has_negation_within(window: str) -> bool:
    tokens = _tokens(window)
    return any(tok in NEGATION_WORDS for tok in tokens)


def evaluate_red_flags(text: str) -> bool:
    """Return True if the user's text contains a non-negated emergency trigger
    or any combination trigger. Does NOT consult the symptom list (kept for
    signature stability with callers); the NER-side evidence-extraction path
    handles clinical-condition emergencies via the KG severity lookups."""
    text_lower = text.lower()

    for trigger in STANDALONE_EMERGENCY:
        pos = text_lower.find(trigger)
        if pos == -1:
            continue
        if pos == 0:
            return True
        # Look back ~3 words before the trigger to detect negation.
        window_start = max(0, pos - 30)
        if not _has_negation_within(text_lower[window_start:pos]):
            return True

    for required, min_count in COMBINATION_TRIGGERS:
        matches = sum(1 for s in required if s in text_lower)
        if matches >= min_count:
            return True
    return False


# ----- LangGraph state --------------------------------------------------------

def _dedup(values: Optional[List[str]]) -> List[str]:
    """Deterministic dedup preserving first-seen order (replaces `set(...)`)."""
    if not values:
        return []
    seen = set()
    out = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _add_messages(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Reducer for the `messages` channel: append new system messages but drop
    duplicates so a node re-emitting the same content doesn't double-send."""
    merged = list(left or [])
    if right:
        for m in right:
            if m and m not in merged:
                merged.append(m)
    return merged


def _add_unique_strs(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Reducer for symptom / asked-symptom channels: deterministic dedup."""
    return _dedup((left or []) + (right or []))


def _add_latencies(
    left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    if not right:
        return list(left or [])
    if not left:
        return list(right)
    return list(left) + list(right)


class TriageState(TypedDict, total=False):
    session_id: str
    patient_id: str
    messages: Annotated[List[str], _add_messages]

    present_symptoms: Annotated[List[str], _add_unique_strs]
    asked_symptoms: Annotated[List[str], _add_unique_strs]
    absent_symptoms: Annotated[List[str], _add_unique_strs]

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

    rag_few_shot: Annotated[List[str], _add_messages]
    rag_medquad: Optional[List[Dict]]

    # Per-turn latencies list (each entry is {"node": ..., "t_rag": ..., "t_llm": ...}).
    latencies: Annotated[Optional[List[Dict[str, Any]]], _add_latencies]

    # A stage flag for explicit multi-turn progression. Replacing the implicit
    # "intent=None every turn" anti-pattern that was breaking booking-flow
    # continuation.
    stage: Optional[str]


STAGE_INTAKE = "intake"
STAGE_TRIAGE = "triage"
STAGE_AWAIT_DEPT = "await_dept"
STAGE_AWAIT_AFFIRM_BOOKING = "await_affirm_booking"
STAGE_AWAIT_SLOT_PICK = "await_slot_pick"
STAGE_AWAIT_PAYMENT = "await_payment"
STAGE_DONE = "done"


# Global KG name-to-id cache (populated lazily with a lock).
_kg_condition_name_to_id: Optional[Dict[str, str]] = None
_kg_cname_lock = threading.Lock()

_xgboost_cache: Optional[tuple] = None
_xgboost_lock = threading.Lock()


# --- Nodes ---

def node_emergency_check(state: TriageState) -> TriageState:
    """Entry node: detect emergency phrases and reset per-turn booleans.

    Unlike the prior version, this does NOT blanket-reset `intent` — that broke
    multi-turn booking continuation. We only reset the *per-turn* intent field
    if the user explicitly typed something new (non-empty `messages` delta).
    The checkpointer holds on to `stage` across turns.
    """
    text = state["messages"][-1]

    # Re-extract any new symptoms from the latest message (NER is lru_cached,
    # so re-running on the same text is cheap).
    ner = get_biomedical_ner()
    new_symptoms = ner.extract_symptoms(text)
    if new_symptoms:
        state["present_symptoms"] = list(
            _dedup((state.get("present_symptoms") or []) + new_symptoms)
        )

    matched = evaluate_red_flags(text)

    if matched:
        # Only promote to emergency the first time we detect it; if we
        # previously had a real diagnosis, preserve it so a clarification
        # message ("no I mean my friend passed out") can recover later.
        state["is_emergency"] = True
        state["final_diagnosis"] = "Possible Medical Emergency"
        state["department"] = "General Medicine / Internal Medicine"
        state["triage_level"] = 1
        state["stage"] = STAGE_DONE
    else:
        # If this isn't actually an emergency on a turn where the previous
        # state was an emergency-overwrite, clear the emergency fields so we
        # don't leak 'Possible Medical Emergency' as the diagnosis forever.
        if state.get("final_diagnosis") == "Possible Medical Emergency":
            state["final_diagnosis"] = None
            state["department"] = None
            state["triage_level"] = None
            state["is_emergency"] = False

    # DO NOT reset `intent`. The stage-field-based router now makes the
    # per-turn decision; intent persists along with stage across turns.
    return state


def node_emergency_response(state: TriageState) -> TriageState:
    """Issue the emergency message + audit log + complete the chat session."""
    supabase = get_supabase()

    supabase.table("chat_session").update({
        "status": "completed",
        "is_emergency": True,
        "completed_at": datetime.utcnow().isoformat()
    }).eq("session_id", state["session_id"]).execute()

    supabase.table("audit_log").insert({
        "event": "emergency_flagged",
        "metadata": {"session_id": state["session_id"], "patient_id": state["patient_id"]}
    }).execute()

    state["messages"] = ["EMERGENCY_TRIGGERED: Please seek immediate emergency medical care."]
    state["stage"] = STAGE_DONE
    return state


def fuzzy_match_department(text: str, synonyms: dict, threshold: int = 85) -> Optional[str]:
    """Try to match a department name from the user's text using synonyms and
    fuzzy matching against the Supabase `specialty` table."""
    try:
        from rapidfuzz import process, fuzz
        supabase = get_supabase()
        res = supabase.table("specialty").select("name").execute()
        specialties = [row["name"] for row in (res.data or []) if isinstance(row, dict) and "name" in row]

        # Check synonyms first (exact word match, not substring).
        text_tokens = set(text.lower().split())
        for key, val in synonyms.items():
            if key in text_tokens:
                return val

        if not specialties:
            return None

        match = process.extractOne(text, specialties, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            return match[0]
    except Exception as e:
        logger.warning(f"fuzzy_match_department failed: {e}")
    return None


def fuzzy_match_doctor(text: str) -> Optional[dict]:
    """Fuzzy-match a doctor name."""
    try:
        from rapidfuzz import process, fuzz
        supabase = get_supabase()
        res = supabase.table("doctor").select("id, name, specialty_id, specialty!inner(name)").execute()
        doctors = {row["name"]: {"id": row["id"], "specialty_name": row["specialty"]["name"]} for row in (res.data or []) if isinstance(row, dict) and row.get("name")}
        if not doctors:
            return None
        match = process.extractOne(text, list(doctors.keys()), scorer=fuzz.WRatio)
        if match and match[1] >= 80:
            return doctors[match[0]]
    except Exception as e:
        logger.warning(f"fuzzy_match_doctor failed: {e}")
    return None


def route_after_intent(state: TriageState):
    """Branch after `node_detect_intent`. Includes END in the mapping so a
    future unknown intent doesn't crash the graph."""
    intent = state.get("intent")

    if intent == "direct_booking_doctor":
        return "node_fetch_slots_for_doctor"
    if intent == "direct_booking_department":
        return "node_fetch_slots" if state.get("department") else "node_prompt_department_choice"
    if intent == "symptom_triage":
        return "node_extract_symptoms"
    return END


def node_detect_intent(state: TriageState) -> TriageState:
    text = (state["messages"][-1] or "").lower()

    doc_match = fuzzy_match_doctor(text)
    if doc_match:
        state["intent"] = "direct_booking_doctor"
        state["selected_doctor_id"] = doc_match["id"]
        state["department"] = doc_match["specialty_name"]
        state["requested_doctor_raw"] = text
        state["stage"] = STAGE_AWAIT_SLOT_PICK
        return state

    dept_match = fuzzy_match_department(text, DEPARTMENT_SYNONYMS)
    if dept_match:
        state["intent"] = "direct_booking_department"
        state["department"] = dept_match
        state["requested_department_raw"] = text
        state["stage"] = STAGE_AWAIT_SLOT_PICK
        return state

    text_tokens = set(text.split())
    if any(phrase in text for phrase in BOOKING_TRIGGER_PHRASES) or (
        text_tokens & {"book", "appointment", "schedule"}
    ):
        state["intent"] = "direct_booking_department"
        state["department"] = None
        state["requested_department_raw"] = text
        state["stage"] = STAGE_AWAIT_DEPT
        return state

    state["intent"] = "symptom_triage"
    state["stage"] = STAGE_TRIAGE
    return state


def node_prompt_department_choice(state: TriageState) -> TriageState:
    if not state.get("awaiting_department_choice"):
        state["awaiting_department_choice"] = True
        state["messages"] = ["PROMPT_DEPARTMENT: Which department would you like to book an appointment with?"]
        state["stage"] = STAGE_AWAIT_DEPT
    else:
        text = (state["messages"][-1] or "").lower()
        dept_match = fuzzy_match_department(text, DEPARTMENT_SYNONYMS)
        if dept_match:
            state["department"] = dept_match
            state["awaiting_department_choice"] = False
            state["stage"] = STAGE_AWAIT_SLOT_PICK
            # NOTE: A conditional edge below routes onwards to fetch_slots.
        else:
            state["messages"] = ["PROMPT_DEPARTMENT_RETRY: I couldn't match that department. Please select from the available options."]
            state["stage"] = STAGE_AWAIT_DEPT
    return state


def node_fetch_slots_for_doctor(state: TriageState) -> TriageState:
    supabase = get_supabase()
    res = supabase.table("clinician_slot") \
        .select("id, start_time, doctor_id, doctor!inner(name, rating, avg_consult_min)") \
        .eq("doctor_id", state["selected_doctor_id"]) \
        .eq("status", "open") \
        .order("start_time") \
        .limit(5).execute()

    if not res.data:
        state["messages"] = [
            f"This doctor has no open slots. Would you like to book another doctor in {state.get('department', 'that area')}?"
        ]
        state["intent"] = "direct_booking_department"
        state["selected_doctor_id"] = None
        state["available_slots"] = None
        state["stage"] = STAGE_AWAIT_DEPT
    else:
        state["available_slots"] = res.data
        state["messages"] = ["SLOTS_OFFERED"]
        state["stage"] = STAGE_AWAIT_SLOT_PICK
    return state


def node_fetch_slots(state: TriageState) -> TriageState:
    supabase = get_supabase()
    res = supabase.table("clinician_slot") \
        .select("id, start_time, doctor_id, doctor!inner(name, rating, avg_consult_min, specialty!inner(name))") \
        .eq("status", "open") \
        .eq("doctor.specialty.name", state["department"]) \
        .order("doctor.rating", desc=True) \
        .order("doctor.avg_consult_min", desc=False) \
        .limit(3).execute()

    if not res.data:
        state["messages"] = [f"No available slots in {state['department']} right now."]
        state["available_slots"] = []
    else:
        state["available_slots"] = res.data
        state["messages"] = ["SLOTS_OFFERED"]
    state["stage"] = STAGE_AWAIT_SLOT_PICK
    return state


def _base_evidence(s): 
    return s.split("_@_")[0] if isinstance(s, str) and "_@_" in s else s


def node_extract_symptoms(state: TriageState) -> TriageState:
    """Detect symptoms from the latest user message, intersect with KG base codes."""
    text = (state["messages"][-1] or "")
    ner = get_biomedical_ner()
    new_symptoms = ner.extract_symptoms(text)
    # Strip value suffixes (NER/_regex_fallback emit base codes already, but
    # this is a defensive measure against legacy callers).
    new_base = [_base_evidence(s) for s in new_symptoms]
    state["present_symptoms"] = list(_dedup((state.get("present_symptoms") or []) + new_base))
    return state


def ask_ollama(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Invoke Ollama. Returns None on failure (caller falls back)."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        import requests
        from urllib.parse import urljoin
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        # Cheap pre-flight check so we fail fast.
        requests.get(urljoin(ollama_url, "/api/tags"), timeout=5)

        chat = ChatOllama(model=model_name, base_url=ollama_url, temperature=0.7)
        res = chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return res.content.strip()
    except Exception as e:
        logger.warning(f"Ollama error: {e}. Using fallback response.")
        return None


def node_next_question(state: TriageState) -> TriageState:
    kg = get_kg()
    next_questions = kg.rank_next_questions(
        state.get("present_symptoms") or [],
        state.get("asked_symptoms") or [],
        state.get("absent_symptoms") or [],
    )

    if not next_questions:
        state["messages"] = ["SYSTEM_FALLBACK: Proceeding to analysis."]
        # Force the clinical loop to terminate
        state["asked_symptoms"] = (state.get("asked_symptoms") or []) + ["__exhausted__"]
        return state

    next_symptom_id, _ = next_questions[0]
    next_symptom_id = _base_evidence(next_symptom_id)
    state["asked_symptoms"] = (state.get("asked_symptoms") or []) + [next_symptom_id]

    # Fetch the human-readable text for the evidence so the LLM query phrase
    # is meaningful (previously we sent the bare E_XX code to the LLM, which
    # produced nonsense follow-up questions).
    evid = kg.get_evidence_info(next_symptom_id) or {}
    symptom_phrase = evid.get("question_en") or evid.get("name") or next_symptom_id

    retriever = get_unified_retriever()
    present_symptoms_text = " ".join(state.get("present_symptoms") or [])

    # Query Conversations with the symptom phrase (NL), NOT the bare code.
    few_shot_examples: List[str] = []
    rag_lat: int = 0
    llm_lat: int = 0
    try:
        import time
        t0 = time.time()
        few_shot_examples = retriever.get_fewshot_examples(
            query=symptom_phrase,
            symptom=present_symptoms_text,
            num_examples=3,
        )
        rag_lat = int((time.time() - t0) * 1000)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    state["rag_few_shot"] = few_shot_examples

    system_prompt = (
        "You are a friendly, professional AI medical assistant. "
        "Ask the user if they are experiencing a specific symptom. Keep it brief and conversational (1-2 sentences). "
        "Do not give medical advice. Just ask the question. "
        "Use the examples below as reference for how similar questions are phrased by medical professionals, "
        "but do NOT copy them directly - generate your own natural question based on the pattern."
    )

    user_prompt = f"The symptom to ask about is: {symptom_phrase}."

    if few_shot_examples:
        user_prompt += "\n\nHere are examples of how medical professionals ask similar questions:\n"
        for i, example in enumerate(few_shot_examples, 1):
            user_prompt += f"{i}. {example}\n"
        user_prompt += "\nGenerate a similar but unique question for this symptom:"

    try:
        import time
        t0 = time.time()
        ollama_response = ask_ollama(system_prompt, user_prompt)
        llm_lat = int((time.time() - t0) * 1000)
    except Exception:
        ollama_response = None

    state["latencies"] = [{"node": "node_next_question", "RAG Retrieval": rag_lat, "LLM Generation": llm_lat}]

    if ollama_response and ollama_response.strip():
        state["messages"] = [f"QUESTION: {ollama_response}"]
    else:
        # Fallback question uses the human-readable phrase rather than the code.
        state["messages"] = [f"QUESTION: Do you have {symptom_phrase}?"]

    return state


def _load_xgboost_artifacts():
    """Load XGBoost model + preprocessing artifacts (cached, thread-safe)."""
    global _xgboost_cache

    if _xgboost_cache is not None:
        return _xgboost_cache

    with _xgboost_lock:
        if _xgboost_cache is not None:
            return _xgboost_cache

        import os
        import json as _json
        import pickle
        import xgboost as xgb

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model"))
        xgb_path = os.path.join(model_dir, "xgb_model.json")
        manifest_path = os.path.join(model_dir, "training_manifest.json")

        if not os.path.exists(xgb_path):
            logger.warning(
                f"XGBoost model not found at {xgb_path}. Triage classification will "
                "fall back to KG/keyword-only department prediction; final_diagnosis will be 'Uncertain Diagnosis'."
            )
            # Cache the negative result so we don't re-stat the disk every turn.
            _xgboost_cache = (None, None, None)
            return _xgboost_cache

        try:
            mlb = None
            le = None
            with open(os.path.join(model_dir, "mlb.pkl"), "rb") as f:
                mlb = pickle.load(f)
            with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
                le = pickle.load(f)

            clf = xgb.XGBClassifier()
            clf.load_model(xgb_path)

            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = _json.load(f)
            # Restore the multi-class objective params so predict_proba routes
            # through the right path (sklearn wrapper expects num_class /
            # objective to match the trained booster).
            clf.set_params(
                objective=manifest.get("objective", "multi:softprob"),
                num_class=manifest.get("num_class", len(le.classes_)),
            )

            _xgboost_cache = (clf, mlb, le)
            logger.info("XGBoost model + MLB + LabelEncoder loaded")
        except Exception as e:
            logger.error(f"XGBoost load failed: {e}")
            _xgboost_cache = (None, None, None)

        return _xgboost_cache


def _get_patient_demographics(patient_id: str) -> tuple:
    supabase = get_supabase()
    patient_res = supabase.table("patient").select("age, gender").eq("id", patient_id).execute()
    if patient_res.data:
        age = patient_res.data[0].get("age", 30)
        gender_str = patient_res.data[0].get("gender", "male")
        sex = 0 if gender_str.lower() == "female" else 1
        return age, sex
    # Align with training-time fillna(0) so distribution shift is minimized.
    # (Previously used 30/1 which doesn't match the training-time imputation.)
    return 0, 0


def _get_kg_severity(pred_condition: str) -> int:
    kg = get_kg()
    severity = kg.get_condition_severity(pred_condition)
    if not isinstance(severity, int) or severity < 1 or severity > 5:
        return 3
    return severity


def _get_kg_department(pred_condition: str) -> Optional[str]:
    """Look up a department for a condition NAME from the KG, returning a
    canonical Supabase specialty (or None if the condition is unknown)."""
    global _kg_condition_name_to_id
    kg = get_kg()
    if _kg_condition_name_to_id is None or True:
        # Build once under lock. After build, the reference stays valid for the
        # process lifetime; the lock prevents a race during construction.
        with _kg_cname_lock:
            if _kg_condition_name_to_id is None:
                mapping: Dict[str, str] = {}
                for cid, cinfo in kg.conditions.items():
                    cond_name = (cinfo.get("condition_name") or cinfo.get("name") or "").strip().lower()
                    if cond_name:
                        mapping[cond_name] = cid
                _kg_condition_name_to_id = mapping
    condition_id = _kg_condition_name_to_id.get(pred_condition.strip().lower())
    if condition_id:
        return kg.get_condition_specialty(condition_id)
    return None


def node_classify(state: TriageState) -> TriageState:
    clf, mlb, le = _load_xgboost_artifacts()

    if clf is not None and mlb is not None and le is not None:
        import scipy.sparse as sp
        import numpy as np

        # Validate NER-extracted codes against the MLB vocabulary. Unknown
        # codes are dropped by MultiLabelBinarizer (with a UserWarning). If
        # *all* codes are unknown, the row will be all-zero and argmax would
        # produce the dataset prior — we short-circuit that case.
        valid_codes = [c for c in (state.get("present_symptoms") or []) if c in mlb.classes_]
        if not valid_codes:
            logger.warning(
                "None of present_symptoms are in the MLB vocabulary; treating as Uncertain (model prior only)."
            )
            state["final_diagnosis"] = "Uncertain Diagnosis"
            state["confidence"] = 0.0
            state["triage_level"] = _get_kg_severity("Unknown")
            from .symptom_to_dept import predict_department_from_symptoms
            department = predict_department_from_symptoms(state.get("present_symptoms") or [])
        else:
            evidence_matrix = mlb.transform([valid_codes])
            age, sex = _get_patient_demographics(state["patient_id"])
            age_sex_matrix = sp.csr_matrix([[age, sex]])
            X = sp.hstack([age_sex_matrix, evidence_matrix])

            probs = clf.predict_proba(X)[0]
            # Top-K for severity aggregation — the predicted condition may be
            # wrong, but USUALLY the highest-severity candidate is the right
            # urgency floor (e.g. "Pulmonary embolism" severity=2 vs "URI"
            # severity=4 — if either is plausible, the right triage level is 2).
            top_k_idx = np.argsort(probs)[::-1][:5]
            top_k_conditions = le.inverse_transform(top_k_idx)
            top_k_severities = [_get_kg_severity(c) for c in top_k_conditions]
            # Best severity among plausible (probability > epsilon) candidates.
            max_idx = int(top_k_idx[0])
            confidence = float(probs[max_idx])
            pred_condition = le.inverse_transform([max_idx])[0]

            severity = top_k_severities[0]
            # Low-confidence triage floor: if the model is unsure, escalate (use
            # the *most urgent* severity across the top-K plausible classes),
            # don't relax. This was the inverse (wrong) direction before.
            confidence_floor = float(os.getenv("CONFIDENCE_FLOOR", "0.3"))
            if confidence < confidence_floor:
                severity = min(s for s in top_k_severities if s)  # most urgent = lowest number
                state["final_diagnosis"] = "Uncertain Diagnosis"
                state["triage_level"] = severity
                state["confidence"] = confidence
                from .symptom_to_dept import predict_department_from_symptoms
                department = predict_department_from_symptoms(state.get("present_symptoms") or [])
            else:
                state["final_diagnosis"] = pred_condition
                state["confidence"] = confidence
                state["triage_level"] = severity
                department = _get_kg_department(pred_condition)
                if not department:
                    from .symptom_to_dept import predict_department_from_symptoms
                    department = predict_department_from_symptoms(state.get("present_symptoms") or [])
    else:
        # No trained model — don't fake `confidence=0.8`. Surface uncertainty
        # to the operator (audit log), use KG/keyword for dept, and set a low
        # explicit confidence so downstream code knows this was a guess.
        supabase = get_supabase()
        try:
            supabase.table("audit_log").insert({
                "event": "xgboost_unavailable_fallback_used",
                "metadata": {"session_id": state.get("session_id")}
            }).execute()
        except Exception:
            pass

        state["final_diagnosis"] = "Uncertain Diagnosis"
        state["confidence"] = 0.0
        state["triage_level"] = 3

        from .symptom_to_dept import predict_department_from_symptoms
        department = predict_department_from_symptoms(state.get("present_symptoms") or [])

    state["department"] = department
    state["stage"] = STAGE_AWAIT_AFFIRM_BOOKING
    return state


def node_explain(state: TriageState) -> TriageState:
    """Run RAG against MedQuAD to gather context, then ask Ollama to explain."""
    import time
    t0 = time.time()

    # Skip RAG when diagnosis is uncertain — passing "Uncertain Diagnosis" as
    # the query to MedQuAD would retrieve ~random chunks.
    rag_chunks: List[Dict] = []
    if state.get("final_diagnosis") and state["final_diagnosis"] != "Uncertain Diagnosis":
        try:
            retriever = get_unified_retriever()
            rag_chunks = retriever.retrieve_medquad(state["final_diagnosis"])
        except Exception as e:
            logger.warning(f"MedQuAD retrieval failed: {e}")
    t_rag = int((time.time() - t0) * 1000)
    state["rag_medquad"] = rag_chunks

    system_prompt = (
        "You are a friendly, professional AI medical assistant. "
        "Explain to the patient that based on their symptoms, they might have a specific condition. "
        "Keep it empathetic and reassuring (2-3 sentences). "
        "Always clarify that this is not a definitive medical diagnosis and they should consult the doctor."
    )
    diagnosis = state.get("final_diagnosis") or "your symptoms"
    user_prompt = f"The condition is: {diagnosis}."

    # Extract *text* from the chunks, with a token budget. Previously we passed
    # the raw dict repr to the LLM, which buried the actual medical content.
    if rag_chunks:
        ctx_parts = []
        budget_tokens = 1500
        running_tokens = 0
        for r in rag_chunks:
            chunk = r.get("chunk") or {}
            txt = chunk.get("answer_chunk") or chunk.get("full_answer") or ""
            if not txt:
                continue
            tokens_est = len(txt) // 4
            if running_tokens + tokens_est > budget_tokens:
                break
            ctx_parts.append(txt)
            running_tokens += tokens_est
        if ctx_parts:
            user_prompt += "\nHere is some medical context to help you explain it accurately:\n" + "\n---\n".join(
                ctx_parts
            )

    try:
        t0 = time.time()
        ollama_response = ask_ollama(system_prompt, user_prompt)
        t_llm = int((time.time() - t0) * 1000)
    except Exception:
        ollama_response = None
        t_llm = 0

    state["latencies"] = [{"node": "node_explain", "MedQuAD RAG": t_rag, "LLM Generation": t_llm}]

    if ollama_response and ollama_response.strip():
        explanation = f"DIAGNOSIS_EXPLANATION: {ollama_response}"
    else:
        explanation = (
            f"DIAGNOSIS_EXPLANATION: Based on your symptoms, you might have "
            f"{diagnosis}. Please consult with a healthcare professional for a proper diagnosis."
        )

    supabase = get_supabase()
    try:
        supabase.table("chat_session").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "final_diagnosis": state.get("final_diagnosis"),
            "department": state.get("department"),
            "triage_level": state.get("triage_level"),
            "confidence": state.get("confidence"),
            "triage_summary": explanation,
        }).eq("session_id", state["session_id"]).execute()
    except Exception as e:
        try:
            supabase.table("audit_log").insert({
                "event": "chat_session_persist_failed",
                "metadata": {"error": str(e), "session_id": state["session_id"]}
            }).execute()
        except Exception:
            pass

    state["messages"] = [explanation]
    state["stage"] = STAGE_AWAIT_AFFIRM_BOOKING
    return state


def node_prompt_booking(state: TriageState) -> TriageState:
    state["messages"] = ["PROMPT_BOOKING: Would you like to book an appointment with this department?"]
    state["stage"] = STAGE_AWAIT_AFFIRM_BOOKING
    return state


def node_handle_booking(state: TriageState) -> TriageState:
    text = (state["messages"][-1] or "").lower()
    text_tokens = set(text.split())
    if text_tokens & AFFIRMATIVE_TOKENS and not (text_tokens & NEGATIVE_TOKENS):
        state["booking_intent"] = True
        state["stage"] = STAGE_AWAIT_SLOT_PICK
    else:
        state["booking_intent"] = False
        state["messages"] = ["Okay, please let me know if you need anything else."]
        state["stage"] = STAGE_DONE
    return state


def node_parse_slot_selection(state: TriageState) -> TriageState:
    """NEW node: read a slot index/id from the user message and set
    `selected_slot_id`. Previously nothing ever populated `selected_slot_id`,
    which made `node_confirm_slot` unreachable."""
    text = (state["messages"][-1] or "")
    slots = state.get("available_slots") or []

    # Several supported grammars:
    #   "1", "2"            → 1-indexed pick from available_slots
    #   "<slot_id-uuid>"    → if slot id is in available_slots
    selected: Optional[str] = None
    stripped = text.strip()
    # 1) numeric pick
    if stripped.isdigit():
        idx = int(stripped)
        if 1 <= idx <= len(slots):
            selected = slots[idx - 1].get("id")
    # 2) negative answer — user declines
    if selected is None and any(tok in NEGATIVE_TOKENS for tok in stripped.lower().split()):
        state["messages"] = ["Okay, please let me know if you need anything else."]
        state["stage"] = STAGE_DONE
        return state
    # 3) fuzzy-match by raw slot text
    if selected is None:
        for slot in slots:
            slot_id = slot.get("id", "")
            if slot_id and slot_id in text:
                selected = slot_id
                break

    if selected:
        state["selected_slot_id"] = selected
        state["stage"] = STAGE_AWAIT_PAYMENT
    else:
        state["messages"] = ["PROMPT_SLOT_RETRY: I couldn't match that to a slot. Please reply with the slot number (e.g. 1, 2, 3)."]
        state["stage"] = STAGE_AWAIT_SLOT_PICK
    return state


def node_confirm_slot(state: TriageState) -> TriageState:
    supabase = get_supabase()
    slot_id = state.get("selected_slot_id")
    if not slot_id and state.get("available_slots"):
        slot_id = state["available_slots"][0].get("id")

    if not slot_id:
        state["messages"] = ["SLOT_CONFIRMATION_FAILED: No slot selected."]
        state["stage"] = STAGE_AWAIT_SLOT_PICK
        return state

    try:
        triage_level = state.get("triage_level") or 5
        confidence = state.get("confidence")

        supabase.rpc("book_slot", {
            "p_slot_id": slot_id,
            "p_patient_id": state["patient_id"],
            "p_chat_session_id": state["session_id"],
            "p_department": state.get("department"),
            "p_triage_level": triage_level,
            "p_confidence": confidence,
        }).execute()

        if state.get("intent") != "symptom_triage":
            supabase.table("chat_session").update({
                "department": state.get("department"),
                "triage_level": triage_level,
                "confidence": confidence,
                "final_diagnosis": "Patient-requested direct booking",
                "status": "completed"
            }).eq("session_id", state["session_id"]).execute()

        state["payment_status"] = "pending"
        state["messages"] = [f"SLOT_CONFIRMED: {slot_id}"]
        state["stage"] = STAGE_AWAIT_PAYMENT
    except Exception as e:
        msg = str(e)
        if "SLOT_NOT_AVAILABLE" in msg:
            state["messages"] = ["That slot was just taken. Let's find another."]
            state["selected_slot_id"] = None
            state["available_slots"] = None  # force refetch
            state["stage"] = STAGE_AWAIT_SLOT_PICK
        else:
            # Surface the failure visibly (previously was silently swallowed).
            logger.error(f"node_confirm_slot failed: {e}")
            state["messages"] = [f"SLOT_CONFIRMATION_FAILED: {msg}"]
            try:
                supabase.table("audit_log").insert({
                    "event": "slot_confirm_failed",
                    "metadata": {"error": msg, "session_id": state["session_id"], "slot_id": slot_id}
                }).execute()
            except Exception:
                pass
            state["stage"] = STAGE_AWAIT_SLOT_PICK
    return state


def node_process_payment(state: TriageState) -> TriageState:
    """Charge the patient. Idempotent: only acts when payment_status ==
    'pending', preventing duplicate payment rows on subsequent messages."""
    if state.get("payment_status") != "pending":
        # Payment already succeeded (or was never initiated). Don't fire again
        # just because the user's reply contained the substring "pay".
        return state

    try:
        import uuid
        # Simulated stripe delay
        # FIXME: replace with a real payment provider integration before prod.
        supabase = get_supabase()
        res = supabase.table("appointment").select("id").eq("chat_session_id", state["session_id"]).execute()
        if not res.data:
            state["messages"] = ["PAYMENT_FAILED: No appointment found to pay for."]
            state["stage"] = STAGE_DONE
            return state
        appt_id = res.data[0]["id"]

        # Guard against duplicate payment rows: check existence first.
        existing = supabase.table("payment") \
            .select("id") \
            .eq("appointment_id", appt_id) \
            .eq("status", "succeeded") \
            .limit(1).execute()
        if existing.data:
            state["payment_status"] = "succeeded"
            state["messages"] = ["PAYMENT_SUCCESS: Appointment already confirmed."]
            state["stage"] = STAGE_DONE
            return state

        supabase.table("payment").insert({
            "appointment_id": appt_id,
            "stripe_intent": f"pi_{uuid.uuid4().hex[:12]}",
            "status": "succeeded",
            "amount_paisa": 150000  # 1500 INR
        }).execute()

        state["payment_status"] = "succeeded"
        state["messages"] = ["PAYMENT_SUCCESS: Appointment confirmed."]
        state["stage"] = STAGE_DONE
    except Exception as e:
        logger.error(f"node_process_payment failed: {e}")
        state["messages"] = [f"PAYMENT_FAILED: {e}"]
        state["stage"] = STAGE_DONE
    return state


# --- Routing ------------------------------------------------------------------


def route_entry(state: TriageState):
    """Branch from entry node based on stage + per-turn state.

    Now uses `stage` as the primary decision key, with boolean state only used
    as a fallback for the stage-less discovery path. This fixes the bug where
    the prior code reset intent=None every turn and never reached the slot
    confirmation path.
    """
    if state.get("is_emergency"):
        return "node_emergency_response"

    stage = state.get("stage")
    if stage == STAGE_AWAIT_PAYMENT:
        return "node_process_payment"
    if stage == STAGE_AWAIT_SLOT_PICK:
        # Either we have slots + awaiting pick, or we need to fetch from DB.
        if state.get("available_slots") is None and state.get("department") and not state.get("selected_slot_id"):
            if state.get("selected_doctor_id"):
                return "node_fetch_slots_for_doctor"
            return "node_fetch_slots"
        return "node_parse_slot_selection"
    if stage == STAGE_AWAIT_DEPT:
        return "node_prompt_department_choice"
    if stage == STAGE_AWAIT_AFFIRM_BOOKING:
        # User just sent a reply after being prompted to book.
        if (state.get("available_slots") is None) and state.get("booking_intent") is None:
            return "node_handle_booking"
        if state.get("booking_intent") and state.get("available_slots") is None:
            return "node_fetch_slots"
        return "node_handle_booking"
    if stage == STAGE_TRIAGE:
        return "node_extract_symptoms"
    if stage == STAGE_DONE:
        return END

    # Fallback (intake / unknown stage) → intent detection
    if not state.get("intent"):
        return "node_detect_intent"

    # Pre-stage legacy branch — keep as a backstop.
    intent = state.get("intent")
    if intent == "direct_booking_department" and not state.get("department"):
        return "node_prompt_department_choice"
    if intent == "direct_booking_doctor" and not state.get("available_slots"):
        return "node_fetch_slots_for_doctor"
    if intent == "direct_booking_department" and state.get("department") and not state.get("available_slots"):
        return "node_fetch_slots"
    if state.get("payment_status") == "pending":
        return "node_process_payment"
    if state.get("available_slots") is not None and state.get("booking_intent") is None and not state.get("selected_slot_id"):
        return "node_handle_booking"
    if intent == "symptom_triage" and state.get("booking_intent") and not state.get("available_slots"):
        return "node_fetch_slots"
    if intent == "symptom_triage" and state.get("final_diagnosis") and state.get("booking_intent") is None and not state.get("available_slots"):
        return "node_prompt_booking"
    return "node_extract_symptoms"


def route_clinical_loop(state: TriageState):
    """Termination gate for the symptom-question loop.

    Exits to classify when:
      - asked >= 5 questions, OR
      - present symptoms >= 3, OR
      - we exhausted the candidate evidences (sentinel "__exhausted__"),
        so the loop doesn't ping-pong indefinitely.

    Otherwise, ask the next highest-IG question.
    """
    asked = state.get("asked_symptoms") or []
    present = state.get("present_symptoms") or []
    if "__exhausted__" in asked:
        return "node_classify"
    if len(asked) >= 5 or len(present) >= 3:
        return "node_classify"
    return "node_next_question"


def route_after_dept_choice(state: TriageState):
    """Conditional edge off `node_prompt_department_choice`: forward the booking
    flow when a department was just matched."""
    if state.get("awaiting_department_choice"):
        return END  # stay waiting
    return "node_fetch_slots"


def route_after_fetch_slots_for_doctor(state: TriageState):
    """If `node_fetch_slots_for_doctor` failed over to dept-booking mode, route
    to `node_fetch_slots` rather than END so the fallback completes in turn."""
    if state.get("intent") == "direct_booking_department" and state.get("department") and state.get("available_slots") is None:
        return "node_fetch_slots"
    return END


# --- Graph Definition ---------------------------------------------------------

def build_graph():
    builder = StateGraph(TriageState)

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
    builder.add_node("node_parse_slot_selection", node_parse_slot_selection)  # NEW
    builder.add_node("node_confirm_slot", node_confirm_slot)
    builder.add_node("node_process_payment", node_process_payment)

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
            "node_parse_slot_selection": "node_parse_slot_selection",
            "node_extract_symptoms": "node_extract_symptoms",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "node_detect_intent",
        route_after_intent,
        {
            "node_fetch_slots_for_doctor": "node_fetch_slots_for_doctor",
            "node_prompt_department_choice": "node_prompt_department_choice",
            "node_fetch_slots": "node_fetch_slots",
            "node_extract_symptoms": "node_extract_symptoms",
            END: END,
        },
    )

    # Department-choice forwarding now drives booking onward instead of
    # ending the turn here (previously the flow dead-ended after a successful match).
    builder.add_conditional_edges(
        "node_prompt_department_choice",
        route_after_dept_choice,
        {
            "node_fetch_slots": "node_fetch_slots",
            END: END,
        },
    )

    # Likewise forward the booking flow on the doctor-slot failure path.
    builder.add_conditional_edges(
        "node_fetch_slots_for_doctor",
        route_after_fetch_slots_for_doctor,
        {
            "node_fetch_slots": "node_fetch_slots",
            END: END,
        },
    )

    # After slots are offered, the next turn parses the user's slot pick.
    # In the same turn (since node_fetch_slots pushes SLOTS_OFFERED), we end
    # and wait for the user's reply.
    builder.add_edge("node_fetch_slots", END)

    builder.add_conditional_edges(
        "node_extract_symptoms",
        route_clinical_loop,
        {
            "node_classify": "node_classify",
            "node_next_question": "node_next_question",
        },
    )

    builder.add_edge("node_next_question", END)
    builder.add_edge("node_classify", "node_explain")
    builder.add_edge("node_explain", "node_prompt_booking")

    # After asking "would you like to book?" → wait for user reply (END).
    # The next user message comes back via route_entry with stage=await_affirm_booking.
    builder.add_edge("node_prompt_booking", END)
    builder.add_edge("node_handle_booking", END)

    # Slot parsing routes to confirm_slot; on failure loops back to await_slot_pick.
    builder.add_edge("node_parse_slot_selection", END)
    builder.add_edge("node_confirm_slot", "node_process_payment")
    builder.add_edge("node_process_payment", END)
    builder.add_edge("node_emergency_response", END)

    return builder
