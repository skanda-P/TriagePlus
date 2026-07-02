# gemini_inference.py
"""
Gemini 2.5-flash interactive and final inference module for TriagePlus.
Loads SentenceTransformer and FAISS globally to eliminate latency.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import time

# Set KMP flag before any heavy imports to prevent OpenMP deadlocks on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("gemini_inference")

DEPARTMENTS = [
    "Cardiology", "Dermatology", "Emergency Medicine", "Endocrinology",
    "Gastroenterology", "General Medicine", "Gynecology", "Hematology",
    "Neurology", "Oncology", "Ophthalmology", "Orthopedics",
    "Pediatrics", "Psychiatry", "Pulmonology", "Rheumatology", "Urology",
]

_embedder = None
_index_a = None
_index_b = None
_meta_a = None
_meta_b = None
_client = None

def _get_gemini_client():
    global _client
    if _client is None:
        import google.generativeai as genai
        from dotenv import load_dotenv
        if not os.environ.get("GEMINI_API_KEY"):
            env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
            load_dotenv(env_path)
            
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        _client = genai.GenerativeModel(model_name="gemini-2.5-flash")
    return _client

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

def infer_department_interactive(chat_history: list, session_id: str = "default"):
    """
    Called on every chat message in GEMINI_CONVERSATION state.
    Uses Index A (conversations) to guide follow-up questions.
    """
    logger.info(f"Running interactive inference for session {session_id}...")
    embedder, index_a, _, meta_a, _ = _get_rag_components()
    
    # Use the last few messages for FAISS search to get contextual few-shot examples
    last_msgs = " ".join([m["content"] for m in chat_history[-3:]])
    top_a, cases_text, _, t_embed, t_faiss = _search_faiss(last_msgs, index_a, meta_a, top_k=3)
    
    # Format chat history
    formatted_history = ""
    for msg in chat_history:
        role = "Doctor" if msg["role"] == "assistant" else "Patient"
        formatted_history += f"{role}: {msg['content']}\n"
        
    prompt = f"""
You are an expert medical triage assistant (Doctor). 
Your goal is to converse with the Patient to gather enough information about their symptoms to recommend a hospital department.

Here are examples of how a Doctor typically asks follow-up questions for similar cases:
{cases_text}

Here is the current conversation:
{formatted_history}

You must output ONLY a valid JSON object with the following fields:
- "action": either "ask" (if you need more information to confidently triage) or "complete" (if you have gathered sufficient symptoms to make a triage decision).
- "reply": your response to the patient (only if action is "ask").
- "summary": a detailed summary of all the patient's symptoms (only if action is "complete").

Do not ask too many questions. Once you know the primary issue and basic severity, choose "complete".
CRITICAL INSTRUCTION: DO NOT repeat the exact same questions. Acknowledge what the patient said, and ask a specific, new follow-up question based on their last message. If they are unable or unwilling to provide more details, choose "complete".

JSON format:
"""
    client = _get_gemini_client()
    import google.generativeai as genai
    for attempt in range(1, 4):
        try:
            t0_llm = time.time()
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            t_llm = (time.time() - t0_llm) * 1000
            
            raw = (response.text or "").strip()
            print("=== PROMPT ===", prompt)
            print("=== RAW RESPONSE ===", raw)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            data = json.loads(raw, strict=False)
            data["top_k_a"] = top_a
            data["prompt"] = prompt
            data["raw_response"] = raw
            data["latencies"] = {
                "embed": round(t_embed, 2),
                "faiss": round(t_faiss, 2),
                "llm": round(t_llm, 2),
                "total": round(t_embed + t_faiss + t_llm, 2)
            }
            return data
        except Exception as e:
            logger.error(f"Interactive inference attempt {attempt} failed: {e}")
            
    raise Exception("Gemini API failed after 3 attempts.")

def infer_department_final(summary: str, session_id: str = "default"):
    """
    Called when Gemini decides the conversation is "complete".
    Uses Index B (knowledge base) to make final department and urgency prediction.
    """
    logger.info(f"Running final triage inference for session {session_id}...")
    embedder, _, index_b, _, meta_b = _get_rag_components()
    
    top_b, knowledge_text, max_score, t_embed, t_faiss = _search_faiss(summary, index_b, meta_b, top_k=3)
    
    prompt = f"""
You are an expert AI triage assistant.
Analyze the patient's symptom summary and assign them to the most appropriate hospital department.
Also, provide an urgency score from 1 to 10 (1 = non-urgent, 10 = life-threatening emergency).

Medical knowledge reference:
{knowledge_text}

Patient summary: {summary}

You must output ONLY a valid JSON object with these fields:
- 'department': one of {', '.join(DEPARTMENTS)}
- 'confidence': a float between 0.0 and 1.0 representing your certainty.
- 'urgency_score': an integer from 1 to 10.
- 'reasoning': a brief explanation of why you chose this department and urgency.

JSON format:
"""
    client = _get_gemini_client()
    import google.generativeai as genai
    for attempt in range(1, 4):
        try:
            t0_llm = time.time()
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            t_llm = (time.time() - t0_llm) * 1000
            
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            data = json.loads(raw, strict=False)
            dept = data.get("department", "General Medicine")
            conf = float(data.get("confidence", 0.0))
            urgency = int(data.get("urgency_score", 1))
            
            # Fallback Logic: Low confidence OR very poor RAG match
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
