# local_inference.py (formerly gemini_inference.py)
"""
Local LLM interactive and final inference module for TriagePlus using Ollama.
Loads SentenceTransformer and FAISS globally to eliminate latency.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import time
import ollama  # Replaced google.generativeai with ollama

# Set KMP flag before any heavy imports to prevent OpenMP deadlocks on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("local_inference")

DEPARTMENTS = [
    "Cardiology", "Dermatology", "Emergency Medicine", "Endocrinology",
    "Gastroenterology", "General Medicine", "Gynecology", "Hematology",
    "Neurology", "Oncology", "Ophthalmology", "Orthopedics",
    "Pediatrics", "Psychiatry", "Pulmonology", "Rheumatology", "Urology",
]

# ── Convergence controls ────────────────────────────────────────────────────
# Convergence is enforced here in code, not just requested in the prompt.
# The LLM may still choose "complete" early, but it can no longer loop
# forever: once REQUIRED_SLOTS are filled, or MAX_INTERACTIVE_TURNS is hit,
# the backend forces "complete" itself regardless of what the model returns.
REQUIRED_SLOTS = ["chief_complaint", "onset", "severity"]
MAX_INTERACTIVE_TURNS = 6

_embedder = None
_index_a = None
_index_b = None
_meta_a = None
_meta_b = None

# Removed _get_gemini_client() entirely as Ollama runs as a background service

def _get_rag_components():
    global _embedder, _index_a, _index_b, _meta_a, _meta_b
    if _embedder is None:
        logger.info("Loading SentenceTransformer and FAISS indices into memory...")
        from sentence_transformers import SentenceTransformer
        import faiss
        
        # Load embedder
        try:
            _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
        except Exception:
            logger.warning("Local files not found, attempting to download...")
            _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        faiss_dir = Path(__file__).parent.parent / "faiss"
        
        # Load FAISS Index A (Conversations)
        _index_a = faiss.read_index(str(faiss_dir / "index_a.faiss"))
        with open(faiss_dir / "index_a_meta.json", "r", encoding="utf-8") as f:
            _meta_a = json.load(f)
            
        # Load FAISS Index B (Medical Knowledge)
        _index_b = faiss.read_index(str(faiss_dir / "index_b.faiss"))
        with open(faiss_dir / "index_b_meta.json", "r", encoding="utf-8") as f:
            _meta_b = json.load(f)
            
        logger.info("RAG components loaded successfully.")
    return _embedder, _index_a, _index_b, _meta_a, _meta_b

def _search_faiss(query: str, index, meta, top_k=3):
    embedder, _, _, _, _ = _get_rag_components()
    
    t0 = time.time()
    query_emb = embedder.encode([query])[0]
    t_embed = (time.time() - t0) * 1000
    
    t1 = time.time()
    query_emb_np = np.array([query_emb], dtype=np.float32)
    D, I = index.search(query_emb_np, top_k)
    t_faiss = (time.time() - t1) * 1000
    
    results = []
    text_blob = ""
    max_score = -1.0
    
    for i, idx in enumerate(I[0]):
        if idx != -1 and idx < len(meta):
            score = float(D[0][i])
            if score > max_score:
                max_score = score
            results.append(meta[int(idx)])
            text_blob += f"- {meta[int(idx)].get('text', '')}\n"
            
    return results, text_blob, max_score, t_embed, t_faiss

def _empty_slots() -> dict:
    return {"chief_complaint": None, "onset": None, "severity": None, "associated_symptoms": []}

def _slots_satisfied(slots: dict) -> bool:
    return all(slots.get(k) for k in REQUIRED_SLOTS)

def _merge_slots(known: dict, model_slots: dict | None) -> dict:
    """Slots only ever get filled in, never erased — a flaky extraction on a
    later turn can't un-ask a question that was already answered earlier."""
    merged = dict(known)
    model_slots = model_slots or {}
    for k in REQUIRED_SLOTS:
        v = model_slots.get(k)
        if v:
            merged[k] = v
    extra = model_slots.get("associated_symptoms") or []
    if isinstance(extra, list):
        existing = merged.get("associated_symptoms") or []
        merged["associated_symptoms"] = list(dict.fromkeys(existing + extra))
    return merged

def _fallback_summary(slots: dict, chat_history: list) -> str:
    """Used if we're forcing completion (turn cap or LLM failure) and the
    model didn't hand back a usable 'summary' field itself."""
    parts = []
    if slots.get("chief_complaint"):
        parts.append(f"Chief complaint: {slots['chief_complaint']}")
    if slots.get("onset"):
        parts.append(f"Onset: {slots['onset']}")
    if slots.get("severity"):
        parts.append(f"Severity: {slots['severity']}")
    if slots.get("associated_symptoms"):
        parts.append(f"Associated symptoms: {', '.join(slots['associated_symptoms'])}")

    still_missing = [s for s in REQUIRED_SLOTS if not slots.get(s)]
    if parts:
        summary = ". ".join(parts) + "."
        if still_missing:
            # Told explicitly, not just implied by absence — this also helps
            # infer_department_final() itself stay conservative downstream.
            summary += f" (Not confirmed: {', '.join(still_missing)}.)"
        return summary
    return " ".join(m["content"] for m in chat_history if m["role"] == "user")

async def check_emergency_llm(text: str) -> bool:
    """Uses LLM to detect if the patient message is a medical emergency."""
    prompt = f"""
Analyze the following patient message. Is it a life-threatening medical emergency (e.g. heart attack, stroke, severe bleeding, not breathing, unconscious)?
Ignore negative statements like "I do not have chest pain".
Reply ONLY with a valid JSON object: {{"is_emergency": true/false}}.

Patient Message: "{text}"
"""
    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.0, 'num_predict': 128, 'keep_alive': '5m'}
        )
        raw = response['message']['content'].strip()
        data = json.loads(raw, strict=False)
        return bool(data.get("is_emergency", False))
    except Exception as e:
        logger.error(f"Emergency LLM check failed: {e}")
        return False

async def infer_department_interactive(chat_history: list, session_id: str = "default",
                                  turn_count: int = 0, known_slots: dict | None = None,
                                  patient_info: dict | None = None):
    logger.info(f"Running interactive inference for session {session_id} (turn {turn_count})...")
    embedder, index_a, _, meta_a, _ = _get_rag_components()

    known_slots = known_slots or _empty_slots()
    missing = [s for s in REQUIRED_SLOTS if not known_slots.get(s)]
    force_complete = turn_count >= MAX_INTERACTIVE_TURNS

    # RAG Amnesia fix
    chief_complaint = known_slots.get("chief_complaint") or (chat_history[0]["content"] if chat_history else "")
    latest_msg = chat_history[-1]["content"] if chat_history else ""
    query = f"Patient symptom: {chief_complaint}. Latest update: {latest_msg}"
    top_a, cases_text, _, t_embed, t_faiss = _search_faiss(query, index_a, meta_a, top_k=3)
    
    formatted_history = ""
    for msg in chat_history:
        role = "Doctor" if msg["role"] == "assistant" else "Patient"
        formatted_history += f"{role}: {msg['content']}\n"

    # Step 1: Extraction
    patient_context = f"Age: {patient_info.get('age', 'Unknown')}, Gender: {patient_info.get('gender', 'Unknown')}" if patient_info else "Unknown"

    extraction_prompt = f"""
You are an expert medical triage assistant extracting symptoms.
PATIENT CONTEXT: {patient_context}
ALREADY KNOWN: {json.dumps(known_slots, indent=2)}
LATEST MESSAGE: {latest_msg}

Extract the symptoms into this JSON format:
- "slots": {{"chief_complaint": string or null, "onset": string or null, "severity": string or null, "associated_symptoms": [string]}}

Do NOT blank out any values from ALREADY KNOWN. Merge any new details from LATEST MESSAGE.
Output strict JSON only.
"""
    t_llm = 0
    try:
        t0 = time.time()
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': extraction_prompt}],
            format='json',
            options={'temperature': 0.1, 'num_predict': 512, 'keep_alive': '5m'}
        )
        t_llm += (time.time() - t0) * 1000
        raw = response['message']['content'].strip()
        data = json.loads(raw, strict=False)
        merged_slots = _merge_slots(known_slots, data.get("slots"))
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        merged_slots = known_slots

    # Determine action using Python logic
    new_missing = [s for s in REQUIRED_SLOTS if not merged_slots.get(s)]
    if force_complete or not new_missing:
        action = "complete"
    else:
        action = "ask"
    
    latencies = {
        "embed": round(t_embed, 2),
        "faiss": round(t_faiss, 2),
        "llm": 0, # Will update
        "total": 0
    }

    if action == "complete":
        summary = _fallback_summary(merged_slots, chat_history)
        latencies["llm"] = round(t_llm, 2)
        latencies["total"] = round(t_embed + t_faiss + t_llm, 2)
        yield {
            "type": "result",
            "data": {
                "action": "complete",
                "summary": summary,
                "slots": merged_slots,
                "top_k_a": top_a,
                "prompt": extraction_prompt,
                "raw_response": "Extraction only",
                "latencies": latencies
            }
        }
        return

    # Step 2: Generation (Streaming)
    instruction = ""
    if "chief_complaint" in new_missing:
        instruction = "The patient has not provided a clear chief complaint. Ask them politely to describe what brings them in."
    elif "onset" in new_missing:
        instruction = "The patient has not provided the onset of their symptoms. Ask them politely when the symptoms started."
    elif "severity" in new_missing:
        instruction = "The patient has not provided severity. Ask them politely to rate their pain or discomfort from 1-10."
    else:
        instruction = "Ask them politely for any more details about their symptoms."

    generation_prompt = f"""
You are an expert medical triage assistant.
Current conversation:
{formatted_history}

INSTRUCTION: {instruction}
Respond directly to the patient as the Doctor in 1-2 short sentences. Do not include any internal thoughts, formatting, or JSON.
"""

    yield {"type": "stream_start"}
    full_reply = ""
    t0 = time.time()
    try:
        response_stream = await client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': generation_prompt}],
            options={'temperature': 0.3, 'num_predict': 256, 'keep_alive': '5m'},
            stream=True
        )
        async for chunk in response_stream:
            text = chunk['message']['content']
            full_reply += text
            if text:
                yield {"type": "stream_chunk", "content": text}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        full_reply = "Can you tell me more about that?"
        yield {"type": "stream_chunk", "content": full_reply}
        
    t_llm += (time.time() - t0) * 1000
    latencies["llm"] = round(t_llm, 2)
    latencies["total"] = round(t_embed + t_faiss + t_llm, 2)

    yield {
        "type": "result",
        "data": {
            "action": "ask",
            "reply": full_reply.strip(),
            "slots": merged_slots,
            "top_k_a": top_a,
            "prompt": generation_prompt,
            "raw_response": full_reply,
            "latencies": latencies
        }
    }

def infer_department_final(summary: str, session_id: str = "default", patient_info: dict | None = None):
    """
    Called when LLM decides the conversation is "complete".
    Uses Index B (knowledge base) to make final department and urgency prediction.
    """
    logger.info(f"Running final triage inference for session {session_id}...")
    embedder, _, index_b, _, meta_b = _get_rag_components()
    
    top_b, knowledge_text, max_score, t_embed, t_faiss = _search_faiss(summary, index_b, meta_b, top_k=3)
    
    patient_context = f"Age: {patient_info.get('age', 'Unknown')}, Gender: {patient_info.get('gender', 'Unknown')}" if patient_info else "Unknown"

    prompt = f"""
You are an expert AI triage assistant.
Analyze the patient's symptom summary and assign them to the most appropriate hospital department.
Also, provide an urgency score from 1 to 10 (1 = non-urgent, 10 = life-threatening emergency).

Medical knowledge reference:
{knowledge_text}

Patient context: {patient_context}
Patient summary: {summary}

You must output ONLY a valid JSON object with these fields:
- "department": one of {', '.join(DEPARTMENTS)}
- "confidence": a float between 0.0 and 1.0 representing your certainty.
- "urgency_score": an integer from 1 to 10.
- "reasoning": a brief explanation of why you chose this department and urgency.
"""
    
    for attempt in range(1, 4):
        try:
            t0_llm = time.time()
            
            # Swapped Gemini call for Ollama call
            response = ollama.chat(
                model='llama3.2',
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={
                    'temperature': 0.0, # Kept at 0.0 for strict analytical output
                    'num_predict': 512,
                    'keep_alive': '5m'
                }
            )
            
            t_llm = (time.time() - t0_llm) * 1000
            
            raw = (response['message']['content'] or "").strip()
            
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            data = json.loads(raw, strict=False)
            dept = data.get("department", "General Medicine")
            conf = float(data.get("confidence", 0.0))
            urgency = int(data.get("urgency_score", 1))
            
            if conf < 0.6 or max_score < 0.3:
                logger.warning(f"Low confidence ({conf}) or poor RAG match ({max_score}). Falling back to General Medicine.")
                dept = "General Medicine"
                
            if dept not in DEPARTMENTS:
                dept = "General Medicine"
                
            diag_data = {
                "type": "diagnostic",
                "session_id": session_id,
                "query": summary,
                "top_k_a": [],
                "top_k_b": top_b,
                "prompt": prompt,
                "raw_response": raw,
                "department": dept,
                "confidence": conf,
                "urgency_score": urgency,
                "latencies": {
                    "embed": round(t_embed, 2),
                    "faiss": round(t_faiss, 2),
                    "llm": round(t_llm, 2),
                    "total": round(t_embed + t_faiss + t_llm, 2)
                }
            }
            
            return dept, conf, urgency, diag_data
        except Exception as e:
            logger.error(f"Final inference attempt {attempt} failed: {e}")
            
    fallback_diag = {
        "type": "diagnostic",
        "session_id": session_id,
        "query": summary,
        "top_k_a": [],
        "top_k_b": top_b,
        "prompt": prompt,
        "raw_response": "FAILED",
        "department": "General Medicine",
        "confidence": 0.0,
        "urgency_score": 1,
        "latencies": {"embed": 0, "faiss": 0, "llm": 0, "total": 0}
    }
    return "General Medicine", 0.0, 1, fallback_diag
