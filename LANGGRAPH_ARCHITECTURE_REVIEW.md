# TriagePlus LangGraph Architecture Review

**Status:** AUDIT IN PROGRESS  
**Reviewer:** v0  
**Date:** 2025-07-16

---

## Overview

I've analyzed the actual implementation (`triage_graph.py`, `chat.py`, `main.py`, and supporting modules) against the **product requirements** from `03_AI_Engine_Build_Prompt.md`, `02_Backend_Architecture_Build_Prompt.md`, and the README.

This document lists:
1. **CRITICAL ISSUES** — blocking bugs or spec violations
2. **QUESTIONS FOR YOU** — architectural choices that need clarification
3. **COMPLIANCE GAPS** — features specified but not yet implemented
4. **WORKING CORRECTLY** — what's good

---

## CRITICAL ISSUES (Must Fix)

### 1. **Missing XGBoost Classifier (`node_classify`)**
**Spec Location:** AI Engine §6, Backend §7  
**Current Code:** Line 15 is a stub `dummy_ner()`, but `node_classify` itself is **completely missing**.

**Problem:**  
- The spec says `node_classify` must:
  - Take age, gender, and symptom vector
  - Call `_xgb_model.predict_proba()` 
  - Assign `triage_level` from Knowledge Graph severity lookup
  - Apply confidence flooring (if confidence < 0.3 → cap at triage 3)
  - Set `final_diagnosis` and `department` on low confidence
- Your graph has **no XGBoost model loading, no predict call, no severity mapping**.

**Impact:** The entire clinical classification workflow fails silently. Patients get no diagnosis.

**Question for You:**  
- Have you trained the XGBoost model? Is `backend/model/xgboost_model.pkl` present?
- Should I assume the model exists and write the node, or do you need to handle that first?

---

### 2. **Routing Logic Not Fully Implemented (`route_entry`)**
**Spec Location:** AI Engine §2 (Routes 1–11)  
**Current Code:** Graph structure exists but routing edges are incomplete.

**Spec Routes Missing:**
| # | Condition | Route to | Status |
|---|---|---|---|
| 1 | `is_emergency == True` | `node_emergency_response` | ✓ Present |
| 2 | `intent is None` | `node_detect_intent` | ? Unclear |
| 3 | `intent == "direct_booking_department"` and `department is None` | `node_prompt_department_choice` | ✗ Not found |
| 4 | `intent == "direct_booking_doctor"` and `available_slots is None` | `node_fetch_slots_for_doctor` | ✗ Not found |
| 5 | `intent == "direct_booking_department"` and `department is not None` and `available_slots is None` | `node_fetch_slots` | ✗ Not found |
| 6 | `payment_status == "pending"` | `node_process_payment` | ✗ Not found |
| 7 | `selected_slot_id is not None` | `node_confirm_slot` | ✗ Not found |
| 8 | `available_slots is not None` and `booking_intent is None` | `node_handle_booking` | ✗ Not found |
| 9–10 | Symptom-triage paths | Various | ✗ Not found |
| 11 | Default | `node_extract_symptoms` | ? Unclear |

**Impact:** Only emergency detection and some basic flow works. The full state machine graph is incomplete.

**Questions:**
- Are these nodes implemented elsewhere in the file (beyond line 100)?
- Should I check if they exist and just need to be wired into the graph edges?

---

### 3. **Fuzzy Matching Not Production-Ready**
**Spec Location:** AI Engine §4.1  
**Current Code:** Line 99–100 sketch a function but it's incomplete.

**Spec Requirements:**
- Use `rapidfuzz.process.extractOne` with threshold 85
- Query `doctor.name` and `specialty.name` from Supabase
- Reuse NER pipeline (already using `d4data/biomedical-ner-all`)

**Current Implementation:** Stub function, unclear if:
- `rapidfuzz` is installed
- Database queries are happening
- Threshold is actually 85 or different

**Question:**
- Is fuzzy matching fully implemented? If so, where do `fuzzy_match_doctor()` and `fuzzy_match_department()` live?

---

### 4. **RAG Graceful Degradation Not Tested**
**Spec Location:** Backend §3, AI Engine §9  
**Current Code:** `rag.py` has try/except for CUDA, but downstream nodes don't handle `None` returns.

**Problem:**
- Spec says: RAG failures should set `_rag_health[index]` (specific per index) and degrade gracefully
- When `ask_ollama()` returns `None`, nodes fall back to hardcoded text ✓
- But if FAISS indices fail to load, does the graph recover or crash?

**Questions:**
- Do you have a `_rag_health` dictionary tracking index-specific failures?
- What happens if both MedQuAD and MedDialog indices are missing at startup?

---

## COMPLIANCE GAPS (Specified but Not Yet Implemented)

### 5. **Knowledge Graph (DDXPlus) Loader**
**Spec Location:** AI Engine §12, Backend §7  
**Status:** File `backend/app/core/kg.py` exists but I haven't reviewed its contents.

**Spec Says:**
- Must parse `backend/data/ddxplus/release_conditions.json` and `release_evidences.json`
- Build a NetworkX graph with severity (1–5 ESI) attached to each condition
- Provide `get_kg()` singleton
- `node_next_question` calls `kg.rank_next_questions(present_symptoms, asked_symptoms)` by information gain

**Question:**
- Is the KG fully functional, or is it a stub?

---

### 6. **FAISS Index Build Script**
**Spec Location:** AI Engine §9  
**Status:** Should exist at `backend/scripts/build_faiss_indices.py`.

**Spec Says:**
- MedQuAD: split answers with `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`
- MedDialog: split by `\n\n` for conversational examples
- Assert `index.ntotal > 1000` after each build
- Indices saved to `backend/faiss/medquad/` and `backend/faiss/conversations/`

**Questions:**
- Are the FAISS indices pre-built and present, or do they need to be generated?
- Are the source datasets (`backend/data/medquad/`, `backend/data/meddialog/`) present and in the right format?

---

### 7. **Intake FSM (`sessions.db` + state machine)**
**Spec Location:** Backend §4.1  
**Current Code:** File `backend/app/intake_fsm.py` is imported but I haven't reviewed it.

**Spec Says:**
- Flow: `NAME_ENTRY → AGE_ENTRY → GENDER_ENTRY → PHONE_ENTRY → INITIAL_SYMPTOM`
- On `PHONE_ENTRY` complete: resolve or create patient, create `chat_session`
- Return `patient_id` to be injected into first `TriageState`

**Questions:**
- Is the FSM fully implemented?
- Does it actually create the `chat_session` row before passing to LangGraph?

---

### 8. **Booking & Payment Nodes**
**Spec Location:** AI Engine §8, Backend §7  
**Current Code:** `node_confirm_slot`, `node_process_payment` not yet found.

**Spec Says:**
- `node_confirm_slot`: calls `supabase.rpc('book_slot', {...})`
- `node_process_payment`: simulates sleep(1.5), sets `status='succeeded'`
- Both write to `chat_session` with final state

**Questions:**
- Are these implemented?
- Is the `book_slot` RPC fully wired into the database?

---

### 9. **Doctor Portal REST API**
**Spec Location:** Backend §6.2  
**Status:** Router exists at `backend/app/routers/doctor.py`.

**Spec Routes Required:**
- `GET /api/v1/doctor/me`
- `GET /api/v1/doctor/dashboard`
- `GET /api/v1/doctor/queue`
- `GET /api/v1/doctor/appointments?date=`
- `PATCH /api/v1/doctor/appointments/{id}`
- `DELETE /api/v1/doctor/appointments/{id}`
- etc.

**Questions:**
- Are all these routes implemented?
- Do they authenticate via JWT and check doctor role correctly?

---

## QUESTIONS FOR YOU (Clarifications)

### 10. **Ollama Integration — Fallback Behavior**
**Spec Location:** AI Engine §4.1, §5, §6.1  
**Current Code:** Line 226–245 of `triage_graph.py` shows `ask_ollama()` returns `None` if Ollama is unavailable.

**Your Decision Needed:**
- When Ollama is down:
  - Do you want the system to degrade to **hardcoded fallback text** (current approach)?
  - Or should it **fail fast and alert** the user?
  - Or should it **retry with backoff**?

**Impact:** Affects what users see when the LLM is unavailable.

---

### 11. **Triage Level Severity Mapping**
**Spec Location:** AI Engine §6  
**Spec Says:** Pull severity (1–5 ESI) from Knowledge Graph for the predicted condition.

**Questions:**
- Does the Knowledge Graph already have severity attached?
- How should the mapping happen? Example: condition "Pneumonia" → `triage_level = 3`?

---

### 12. **Confidence Flooring Threshold**
**Spec Says:** If confidence < 0.3 → cap `triage_level` at 3 and route to General Medicine.

**Questions:**
- Is 0.3 the right threshold for your use case, or should it be tuned?
- Should this be a configurable environment variable?

---

### 13. **Emergency Rule Set**
**Spec Location:** AI Engine §3  
**Current Code:** Line 24–33 shows dummy rules.

**Spec Says:** Requires clinical review before deployment.

**Questions:**
- Have you reviewed the emergency triggers with a clinician?
- Are the rules in your implementation complete, or placeholders?

---

### 14. **Session Persistence Strategy**
**Spec Location:** Backend §4.2  
**Spec Says:** Use `AsyncSqliteSaver` with `PRAGMA journal_mode=WAL`.

**Questions:**
- Is this configured in `main.py`?
- Are checkpoints being saved between graph invocations?

---

### 15. **CORS & Allowed Origins**
**Spec Location:** Backend §3 (uses `CORS_ALLOWED_ORIGINS` env var)  
**Current Code:** `main.py` line 35 reads it correctly.

**Verify:**
- Is `CORS_ALLOWED_ORIGINS` set in your `.env` to include the frontend URL?

---

## WORKING CORRECTLY

### ✓ Emergency Detection Structure
- Node runs first on every turn
- Sets `is_emergency = True` and routes to response node
- Writes audit log

### ✓ Intake FSM Integration
- Imported and called in `chat.py`
- Completes before LangGraph invocation

### ✓ WebSocket Event Broadcasting
- Diagnostic clients receive updates
- Chat clients receive messages, emergency alerts, metadata
- Ping task keeps connections alive

### ✓ Ollama Fallback Pattern
- Returns `None` on connection failure
- Nodes have hardcoded fallback text

### ✓ FAISS Graceful Degradation
- Try/except on model load with CPU fallback

---

## SUMMARY: What Needs Your Decision

**Before I proceed with fixes, answer these:**

1. **Is the XGBoost model trained and available?** (Yes/No)
2. **Where are the remaining nodes?** (Listed in gap #2 — are they elsewhere in the file or truly missing?)
3. **Are FAISS indices pre-built?** (Yes/No — if no, do we build them first?)
4. **Ollama fallback strategy:** Hardcoded text, fail-fast, or retry? (Choose one)
5. **Has the emergency rule set been clinically reviewed?** (Yes/No/In progress)
6. **Should fuzzy matching threshold be 85 or tunable?** (Current value in code?)

---

## Next Steps (Pending Your Answers)

1. Clarify the questions above.
2. I'll fix critical issues (missing nodes, routing, classifier).
3. I'll implement gaps (remaining booking nodes, payment flow).
4. I'll write a test to verify the full graph traversal end-to-end.

