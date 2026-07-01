import os
import sys
import json
import argparse

# Set KMP flag before any heavy imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure root path is in sys.path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

_embedder = None
_index_a = None
_index_b = None
_meta_a = None
_meta_b = None
_client = None

DEPARTMENTS = [
    "Cardiology", "Dermatology", "Emergency Medicine", "Endocrinology",
    "Gastroenterology", "General Medicine", "Gynecology", "Hematology",
    "Neurology", "Oncology", "Ophthalmology", "Orthopedics",
    "Pediatrics", "Psychiatry", "Pulmonology", "Rheumatology", "Urology",
]

def _get_gemini_client():
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
            load_dotenv(env_path)
            
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        _client = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=f"You are a medical triage AI. Given a patient's conversation summary or symptom description, classify the most appropriate hospital department from the list below. Return ONLY a JSON object with two fields: {{'department': '<one of the departments>', 'confidence': <float 0.0-1.0>}}. Do NOT output any conversational text.\n\nDepartments: {', '.join(DEPARTMENTS)}",
        )
    return _client

def _get_rag_components():
    global _embedder, _index_a, _index_b, _meta_a, _meta_b
    if _embedder is None:
        _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        faiss_dir = Path(__file__).parent.parent / "faiss"
        _index_a = faiss.read_index(str(faiss_dir / "index_a.faiss"))
        _index_b = faiss.read_index(str(faiss_dir / "index_b.faiss"))
        with open(faiss_dir / "index_a_meta.json", "r", encoding="utf-8") as f:
            _meta_a = json.load(f)
        with open(faiss_dir / "index_b_meta.json", "r", encoding="utf-8") as f:
            _meta_b = json.load(f)
    return _embedder, _index_a, _index_b, _meta_a, _meta_b

def run_isolated_inference(text):
    embedder, index_a, index_b, meta_a, meta_b = _get_rag_components()
    query_emb = embedder.encode([text])[0]
    query_emb_np = np.array([query_emb], dtype=np.float32)
    
    D_a, I_a = index_a.search(query_emb_np, 3)
    D_b, I_b = index_b.search(query_emb_np, 3)
    
    top_a = []
    cases_text = ""
    for idx in I_a[0]:
        if idx != -1 and idx < len(meta_a):
            top_a.append(meta_a[int(idx)])
            cases_text += f"- {meta_a[int(idx)].get('text', '')}\n"
            
    top_b = []
    knowledge_text = ""
    for idx in I_b[0]:
        if idx != -1 and idx < len(meta_b):
            top_b.append(meta_b[int(idx)])
            knowledge_text += f"- {meta_b[int(idx)].get('text', '')}\n"
            
    prompt = f"""
You are an expert AI triage assistant.
Analyze the user's symptoms and output ONLY a JSON object with 'department' and 'confidence'.

Reference cases: {cases_text}
Medical knowledge: {knowledge_text}

User symptoms: {text}
"""
    client = _get_gemini_client()
    for attempt in range(1, 4):
        try:
            response = client.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            raw = (response.text or "").strip()
            print(f"RAW GEMINI RESPONSE: {repr(raw)}", file=sys.stderr)
            if not raw:
                raise ValueError("Empty response")
                
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
                
            print(f"CLEANED RAW RESPONSE: {repr(raw)}", file=sys.stderr)
            data = json.loads(raw)
            if "department" in data and "confidence" in data:
                data["top_k_a"] = top_a
                data["top_k_b"] = top_b
                data["prompt"] = prompt
                return data
        except Exception as e:
            import traceback
            print(f"Gemini call attempt {attempt} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return {"department": "General Medicine", "confidence": 0.0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    
    try:
        result = run_isolated_inference(args.text)
        print(json.dumps(result))
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"department": "General Medicine", "confidence": 0.0}))
