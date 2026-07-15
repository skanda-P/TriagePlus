# TriagePlus — AI Engine Build Prompt (LangGraph, ML, RAG)

**Project:** TriagePlus · IIT Dharwad Summer of Innovation · Team "Hardly Human"
**Depends on:** `01_Database_Schema_Build_Prompt.md` (table/RPC names), `02_Backend_Architecture_Build_Prompt.md` (session identity contract, public endpoints)

## 1. Overview

The AI engine isolates LLMs from making clinical decisions. An XGBoost model handles pathology classification, a deterministic NetworkX Knowledge Graph drives the clinical interviewing logic, and HuggingFace/FAISS handle entity extraction and RAG-augmented explanations. LLM usage (`ChatOllama`, llama3.2) is restricted to natural-language rewriting — never to diagnosis, triage-level, routing, or intent decisions. Every classification-like decision (symptom-driven diagnosis, department/doctor intent resolution) is made by deterministic or ML logic and only *phrased* by the LLM.

The engine supports two conversational paths, forking right after intake:
- **Symptom triage** — the existing NER → Knowledge Graph → XGBoost clinical loop, for patients who describe symptoms.
- **Direct booking** — for patients who already know they want a specific department or a specific doctor, mirroring how a typical hospital website chatbot lets you skip straight to scheduling.

Two scale conventions are binding and used consistently through every node in this document:
- **`triage_level`**: integer 1–5, ESI convention. `1` = most critical (Resuscitation), `5` = non-urgent. This is the only severity field in the system — there is no separate "urgency" concept. Direct bookings (no symptom assessment took place) are stored as `triage_level = 5` with a code comment marking them self-scheduled, not AI-triaged.
- **`confidence`**: float in `[0, 1]`, taken straight from `predict_proba()`. Never scaled to a percentage in this layer. Direct bookings store `confidence = null` — there was no classifier run to produce one.

## 2. LangGraph State Machine — Routing

`TriageState` (TypedDict) fields relevant to routing:

| Field | Type | Meaning |
|---|---|---|
| `is_emergency` | bool | Set by `node_emergency_check` |
| `intent` | `"symptom_triage"` \| `"direct_booking_department"` \| `"direct_booking_doctor"` \| `None` | Set once by `node_detect_intent` |
| `requested_department_raw` | str \| None | Raw text the patient used to name a department, kept for logging/debugging |
| `requested_doctor_raw` | str \| None | Raw text the patient used to name a doctor |
| `selected_doctor_id` | uuid \| None | Resolved doctor, only set on the `direct_booking_doctor` path |
| `awaiting_department_choice` | bool | True while `node_prompt_department_choice` is waiting on a reply |
| `department` | str \| None | Resolved specialty name — set either by the classifier or by intent detection |
| `payment_status` | str \| None | |
| `booking_intent` | bool \| None | Patient's yes/no answer to "do you want to book?" (symptom-triage path only) |
| `available_slots` | list \| None | |
| `selected_slot_id` | uuid \| None | |
| `final_diagnosis` | str \| None | |

`route_entry` evaluates the following **in order**, first match wins:

| # | Condition | Route to |
|---|---|---|
| 1 | `is_emergency == True` | `node_emergency_response` (terminal) |
| 2 | `intent is None` | `node_detect_intent` |
| 3 | `intent == "direct_booking_department"` and `department is None` | `node_prompt_department_choice` |
| 4 | `intent == "direct_booking_doctor"` and `available_slots is None` | `node_fetch_slots_for_doctor` |
| 5 | `intent == "direct_booking_department"` and `department is not None` and `available_slots is None` | `node_fetch_slots` |
| 6 | `payment_status == "pending"` | `node_process_payment` |
| 7 | `selected_slot_id is not None` and not yet confirmed | `node_confirm_slot` |
| 8 | `available_slots is not None` and `booking_intent is None` and `selected_slot_id is None` | `node_handle_booking` |
| 9 | `intent == "symptom_triage"` and `booking_intent == True` and `available_slots is None` | `node_fetch_slots` |
| 10 | `intent == "symptom_triage"` and `final_diagnosis is not None` and `booking_intent is None` and `available_slots is None` | `node_prompt_booking` |
| 11 | *default* | `node_extract_symptoms` — only reachable once `intent == "symptom_triage"` |

`node_emergency_check` always runs first, on every turn, regardless of which path the conversation is on — a patient mid-way through a direct doctor booking who mentions chest pain must still be caught.

## 3. Emergency Detection

> The rule set below is a starting structure for a prototype, not a finished clinical protocol. Because this gates a patient-safety-critical path, have it reviewed by your project mentor or a clinical advisor before relying on it beyond a demo.

`node_emergency_check` runs on every turn, before intent detection or symptom extraction. It requires either a standalone high-specificity signal or a combination of correlated signals — a single generic symptom keyword should never be sufficient on its own:

- **Standalone triggers** (one signal is enough — these are unambiguous): loss of consciousness, stated suicidal or self-harm intent, severe uncontrolled bleeding, inability to breathe.
- **Combination triggers** (require ≥2 co-occurring signals): chest pain **and** (shortness of breath **or** radiating arm/jaw pain); sudden facial droop **and** slurred speech **and/or** one-sided weakness (stroke pattern); high fever **and** stiff neck **and** confusion.

Implementation: maintain `RED_FLAG_COMBINATIONS` as a small, explicit, testable list of `(required_entities, min_matches)` tuples, matched against the same `d4data/biomedical-ner-all` NER output already used in `node_extract_symptoms` — reuse that pipeline rather than building a second one.

```python
def node_emergency_check(state: TriageState) -> TriageState:
    matched = evaluate_red_flags(state["present_symptoms"], state["messages"][-1])
    if matched:
        state["is_emergency"] = True
        state["final_diagnosis"] = "Possible Medical Emergency"
        state["department"] = "Emergency Medicine"
    return state
```

`node_emergency_response` (terminal): sets `chat_session.is_emergency = True`, `status = 'completed'`, writes an `audit_log` event (`event: "emergency_flagged"`), and returns a message instructing the patient to seek immediate in-person or emergency care. It does not continue into intent detection, symptom extraction, classification, or booking. The backend's WebSocket layer turns this into the `{"type": "emergency"}` message.

## 4. Intent Detection & Direct Booking

This is the fork that lets a patient skip straight to scheduling — "I'd like to see a cardiologist" or "book me with Dr. Mehta" — without being forced through symptom collection, the same way most hospital website chatbots work.

### 4.1 `node_detect_intent`

Runs exactly once per session, on the first substantive message after intake (i.e. only when `intent is None`). Resolution is **deterministic and DB-backed first**, with the LLM used only as a fallback interpreter whose output is always re-validated against the database — never trust a raw LLM guess for routing.

```python
DEPARTMENT_SYNONYMS = {
    "skin": "Dermatology", "heart": "Cardiology", "child": "Pediatrics",
    "kids": "Pediatrics", "bone": "Orthopedics", "joint": "Orthopedics",
    "stomach": "Gastroenterology", "digestive": "Gastroenterology",
    "brain": "Neurology", "nerve": "Neurology", "mental": "Psychiatry",
    "lung": "Respiratory", "breathing": "Respiratory",
    # Extend this dict as real usage surfaces new phrasing. Keep it in one place,
    # reviewable in a single diff — do not scatter synonym matching across nodes.
}
BOOKING_TRIGGER_PHRASES = ["book", "appointment", "schedule", "see a doctor", "see dr", "consult"]

def node_detect_intent(state: TriageState) -> TriageState:
    text = state["messages"][-1].lower()

    # 1. Doctor name match — fuzzy against doctor.name (rapidfuzz, threshold 85)
    doctor_match = fuzzy_match_doctor(text, threshold=85)  # queries `doctor` table
    if doctor_match:
        state["intent"] = "direct_booking_doctor"
        state["selected_doctor_id"] = doctor_match.id
        state["department"] = doctor_match.specialty_name
        state["requested_doctor_raw"] = text
        return state

    # 2. Department match — fuzzy against specialty.name plus DEPARTMENT_SYNONYMS keys
    dept_match = fuzzy_match_department(text, synonyms=DEPARTMENT_SYNONYMS, threshold=85)
    if dept_match:
        state["intent"] = "direct_booking_department"
        state["department"] = dept_match  # canonical specialty.name
        state["requested_department_raw"] = text
        return state

    # 3. Booking language present but no department/doctor identified
    if any(phrase in text for phrase in BOOKING_TRIGGER_PHRASES):
        state["intent"] = "direct_booking_department"
        state["department"] = None
        state["requested_department_raw"] = text
        return state

    # 4. Fallback: ask the LLM to extract a department guess, then re-validate
    #    against the specialty table via the same fuzzy_match_department() call
    #    above. If the LLM's guess doesn't clear the threshold either, fall through.
    llm_guess = llm_extract_department_guess(text)  # ChatOllama, constrained JSON output
    dept_match = fuzzy_match_department(llm_guess, synonyms=DEPARTMENT_SYNONYMS, threshold=85) if llm_guess else None
    if dept_match:
        state["intent"] = "direct_booking_department"
        state["department"] = dept_match
        state["requested_department_raw"] = text
        return state

    # 5. Default: treat as symptom description
    state["intent"] = "symptom_triage"
    return state
```

Add `rapidfuzz` as a dependency for `fuzzy_match_doctor` / `fuzzy_match_department` (`rapidfuzz.process.extractOne`). Query `doctor.name` and `specialty.name` fresh from Supabase (or a short-lived in-memory cache refreshed every few minutes) rather than hardcoding names in the matcher.

### 4.2 `node_prompt_department_choice`

Reached when `intent == "direct_booking_department"` but `department is None` — the patient said something booking-shaped ("I'd like to book an appointment") without naming a department.

- **First entry** (`awaiting_department_choice` not yet `True`): emit a message listing available departments (the frontend renders these as tappable chips sourced from `GET /api/v1/specialties`, see the frontend prompt §6), set `awaiting_department_choice = True`, end the turn.
- **Subsequent entry** (a new patient message has arrived while `awaiting_department_choice == True`): re-run `fuzzy_match_department` against the new message. On a match, set `department` and `awaiting_department_choice = False` (route 3 in §2 no longer applies, so the graph proceeds to `node_fetch_slots` next turn). On no match, re-ask, mirroring the same retry pattern `node_handle_booking` already uses for parsing a follow-up reply.

### 4.3 `node_fetch_slots_for_doctor`

Used only on the `direct_booking_doctor` path, where the doctor is already known — no ranking needed since the patient already chose. Query `clinician_slot` where `doctor_id = selected_doctor_id` and `status = 'open'`, ordered by `start_time ASC`, limited to the next 5. If zero slots are open, tell the patient this doctor currently has no open slots and offer to fall back to `node_fetch_slots` for the same department instead (set `intent = "direct_booking_department"`, keep `department` as-is, clear `selected_doctor_id`).

## 5. The Clinical Loop (symptom-triage path)

Only reached when `intent == "symptom_triage"`.

1. **`node_extract_symptoms`**: HuggingFace NER (`d4data/biomedical-ner-all`), filters `Sign_symptom`, `Disease_disorder`, `Detailed_description` entities, matches against DDXPlus evidence strings, appends to `present_symptoms`.
2. **`node_decide_next` (edge)**: if `question_count >= 5` or `len(present_symptoms) >= 3` → `classify`; else → `next_question`.
3. **`node_next_question`**: the Knowledge Graph selects the best un-asked symptom by information gain; retrieves 2 conversational examples from the MedDialog FAISS index (§9.2); `ChatOllama` rewrites the KG's question into a natural, empathetic tone, using the examples as style references.
4. **`node_classify`** (§6).
5. **`node_explain`**: queries MedQuAD FAISS (§9.1) using the final diagnosis; `ChatOllama` generates a safe, patient-facing explanation, injecting FAISS facts for clinical grounding. Writes a preliminary outcome to `chat_session` (§7), falling back to `audit_log` on failure. After `node_explain`, the graph proceeds to `node_prompt_booking` (route 10 in §2) exactly like the direct-booking paths converge on booking — from here on, both paths share the same booking machinery (§8).

## 6. `node_classify` — XGBoost Classifier

- **Feature vector:** numerical age, binary sex (`M=1, F=0`), and a `MultiLabelBinarizer` array of `present_symptoms`.
- **Prediction:** the sparse CSR feature matrix goes to `_xgb_model.predict_proba()`, predicting 1 of 49 DDXPlus conditions. The highest class probability becomes `confidence` (0–1 float, stored unmodified).
- **Severity lookup:** pull the predicted condition's base severity (1–5, ESI convention — DDXPlus already encodes it this way) directly from the Knowledge Graph and assign it directly to `triage_level`.
- **Confidence flooring (safety net):** if the model is uncertain, bias toward caution rather than away from it:
  ```python
  if confidence < 0.3:
      state["triage_level"] = min(state["triage_level"], 3)   # never let an uncertain case read as mild
      state["final_diagnosis"] = "Uncertain Diagnosis"
      state["department"] = "General Medicine / Internal Medicine"
  ```
  This caps `triage_level` at 3 ("Urgent") or more severe under uncertainty, and routes the patient to General Medicine rather than guessing a specialty.

## 7. `node_explain` Persistence

```python
await supabase.table("chat_session").update({
    "status": "completed",
    "completed_at": datetime.utcnow().isoformat(),
    "final_diagnosis": state["final_diagnosis"],
    "department": state["department"],
    "triage_level": state["triage_level"],
    "confidence": state["confidence"],
    "triage_summary": explanation_text,
}).eq("session_id", state["session_id"]).execute()
```
On exception, write to `audit_log` instead (`event: "chat_session_persist_failed"`, full state as `metadata`) so the clinical record isn't lost even if the write fails. This write only happens on the symptom-triage path (there is no explanation step on the direct-booking paths); see §8 for how `chat_session` gets its final values on both paths.

## 8. Booking & Payment Flow (shared by both paths)

Once either path reaches this point, `department` (and possibly `selected_doctor_id`) is set, and the remaining nodes don't need to know which path got them there.

1. **`node_prompt_booking`** *(symptom-triage path only — the direct-booking paths skip straight to fetching slots, since booking was the whole point)*: asks whether the patient wants to book a slot for the recommended department.
2. **`node_handle_booking`** *(symptom-triage path only)*: regex + keyword sets (`affirmative_tokens` / `negative_tokens`) parse booking intent from the reply.
3. **`node_fetch_slots`**: queries `clinician_slot` where `status = 'open'` and the doctor's specialty matches `department`, **ordered by `doctor.rating DESC, doctor.avg_consult_min ASC`**, limited to the top 3 doctors' slots. Used by the symptom-triage path and the `direct_booking_department` path alike — it doesn't care how `department` was set.
4. **`node_confirm_slot`**: calls `supabase.rpc('book_slot', {p_slot_id, p_patient_id, p_chat_session_id, p_department, p_triage_level, p_confidence})`, where:
   ```python
   p_triage_level = state.get("triage_level") or 5   # 5 = self-scheduled / not AI-triaged
   p_confidence = state.get("confidence")             # stays None for direct bookings
   ```
   On `SLOT_NOT_AVAILABLE`, tell the patient the slot was just taken and route back to `node_fetch_slots` (or `node_fetch_slots_for_doctor` on the doctor-specific path). On success, **also upsert `chat_session`** with `department`, `triage_level`, `confidence`, and — for the direct-booking paths only, since `node_explain` never ran — `final_diagnosis = "Patient-requested direct booking"` and `status = 'completed'`. This is the single point where every path's `chat_session` record ends up complete, regardless of how it got there. Sets `payment_status = "pending"` on success.
5. **`node_process_payment`**: parses for "pay", simulates `asyncio.sleep(1.5)`, generates a fake `stripe_intent`, writes `payment.amount_paisa` (integer), sets `status = "succeeded"`, `END`.

## 9. RAG Implementation (FAISS)

Both indices are built **offline**, once, by a standalone script — never at request time. The server only ever loads prebuilt index files at startup. This section is deliberately explicit about chunking so nothing is left for the coding agent to guess.

**Shared setup:**
- **Embedding model:** `NeuML/pubmedbert-base-embeddings` via `langchain_community.embeddings.HuggingFaceEmbeddings`. Output dimension: 768.
- **Vector store:** `langchain_community.vectorstores.FAISS`, backed by a flat `IndexFlatL2` (langchain's default when using `FAISS.from_documents` / `FAISS.from_texts` without a custom index factory). This is adequate at the corpus sizes below; only switch to `IndexIVFFlat` if a corpus grows past roughly 500k vectors and latency becomes a measured problem — don't pre-optimize for that.
- **Build script:** `backend/scripts/build_faiss_indices.py`, run manually via `python -m scripts.build_faiss_indices`. It must assert `index.ntotal > 1000` after building each index and fail loudly if not — a near-empty index from a broken parser should never pass silently into production.
- **On-disk layout:** each index saves via `FAISS.save_local(path)`, which langchain writes as two files: `index.faiss` (vectors) and `index.pkl` (docstore + metadata). Raw source data lives under `backend/data/medquad/`, `backend/data/meddialog/`, and `backend/data/symptom2disease/`, untouched by the build script (it reads from there, writes to `backend/faiss/...`).
- **Startup load:** `FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)` — both loads wrapped in try/except; a failure populates `_rag_health["issues"]` for that index specifically (not a global flag) and the graph degrades gracefully, appending a "degraded mode" disclaimer to LLM output rather than crashing.

### 9.1 Index 1 — MedQuAD (factual grounding, used only in `node_explain`)

- **Source format:** the standard MedQuAD release, organized as subfolders per source institution (e.g. `1_CancerGov_QA/`, `2_GARD_QA/`, ...), each containing XML files with a `<Document>` root, a `<Focus>` element, and one or more `<QAPairs><QAPair>` blocks, each holding `<Question>` and `<Answer>`.
- **Parsing:** use `xml.etree.ElementTree`. For every `QAPair`, extract `(focus, question, answer, qtype, source_file)`. **Drop any pair with a missing or empty `<Answer>`** — this is a known data-quality issue in MedQuAD (a meaningful fraction of entries have no answer text) and must be filtered, not passed through.
- **Cleaning:** strip HTML tags/entities if present, collapse repeated whitespace/newlines, trim.
- **Chunking:** split each `answer` (not the question) with `langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "])`. This produces multiple `Document` chunks per QA pair for long answers, and a single chunk for short ones.
- **Filtering:** discard any resulting chunk under 30 characters (residual noise from splitting).
- **Embedding target:** the embedding is computed on the **chunk text itself** (the answer content), since `node_explain` queries with diagnosis-like text that matches factual content better than it matches question phrasing.
- **Metadata stored per chunk** (not embedded, just attached): `{"focus": ..., "question": ..., "source": ..., "qtype": ..., "chunk_index": ...}`.
- **Build call:** `FAISS.from_documents(documents, embedding=pubmedbert_embeddings)`, then `save_local("faiss/medquad")`.
- **Query construction at retrieval time** (`node_explain`): `query = f"{final_diagnosis}: {', '.join(present_symptoms_display_names)}"`, `k=3`. Use `similarity_search_with_score` (not plain `similarity_search`) and drop any result with L2 distance above an empirically-tuned threshold (start at `1.2`, adjust after inspecting the score distribution on real queries during development). If every candidate is filtered out, set `rag_chunks = []` and still generate the explanation from Knowledge Graph facts alone, appending the degraded-mode disclaimer — never silently inject a weak/irrelevant match just to have something to show.

### 9.2 Index 2 — MedDialog (conversational style examples, used only in `node_next_question`)

- **Source format varies by release** (Kaggle vs. GitHub versions of MedDialog differ in raw structure) — the build script's first step must be a normalization pass that converts whatever raw format is used into one canonical structure: a list of conversations, each a list of `{"speaker": "patient" | "doctor", "text": ...}` turns in order. Write this normalization as an isolated adapter function so raw-format quirks never leak into the chunking logic below.
- **Chunking unit — NOT character-based.** The indexed unit is one *(patient turn → doctor follow-up turn)* exchange pair, not a fixed-size text split. Slide a window over each conversation's turn list and emit one exchange record for every consecutive `(patient, doctor)` turn pair where the speakers actually alternate as expected.
- **Filtering:** drop any turn under 3 words (greetings/noise like "ok", "thanks"). As a lightweight quality gate, keep only exchange pairs where the doctor's follow-up turn contains a `?` or starts with an interrogative word (what/why/when/where/how/do/does/did/is/are) — the goal of this index is retrieving example *questions*, and this heuristic is intentionally simple rather than a second NLP pass.
- **Embedding target — read this carefully, it's the part most likely to be built wrong by default:** the vector embedded is the **patient turn's text**, but the content actually returned and injected into the LLM prompt is the **doctor's follow-up turn**. `langchain`'s `FAISS.from_documents` embeds whatever is in `page_content`, so do **not** call it directly on the doctor text. Instead build it explicitly:
  ```python
  texts = [exchange.patient_turn for exchange in exchanges]
  metadatas = [
      {"doctor_followup": exchange.doctor_turn, "conversation_id": exchange.conv_id}
      for exchange in exchanges
  ]
  index = FAISS.from_texts(texts=texts, embedding=pubmedbert_embeddings, metadatas=metadatas)
  ```
  At retrieval time, read `result.metadata["doctor_followup"]` — that's the string handed to `ChatOllama` as a style reference, never `result.page_content` (which is the patient turn used only for the similarity match).
- **Build call:** as above, then `save_local("faiss/conversations")`.
- **Query construction at retrieval time** (`node_next_question`): `query = " ".join(state["messages"][-2:])` (the last one or two patient messages, raw text), `k=2`. No strict distance filter is required here since these are style references rather than factual claims, but still skip results beyond a generous distance ceiling (e.g. `2.0`) to avoid injecting completely unrelated conversational tone.

## 10. Knowledge Graph Engine (NetworkX)

Loaded lazily via `get_kg()` to parse DDXPlus JSON schemas. `rank_next_questions` filters candidate pathologies by iterating through `present_symptoms`, evaluates remaining un-asked symptoms, and returns the highest-information-gain question (the one that most cleanly splits the remaining candidate list).

## 11. Acceptance Tests

- A curated set of red-flag transcripts (standalone and combination triggers from §3) all set `is_emergency = True` within the same turn and never reach `node_extract_symptoms` or the booking flow on a subsequent turn, on either path.
- A curated set of mild single-symptom transcripts ("I have a headache") never trigger emergency.
- `node_detect_intent` test fixtures: "I want to book with Dr. Mehta" → `direct_booking_doctor`, `selected_doctor_id` set; "I need to see a skin doctor" → `direct_booking_department`, `department == "Dermatology"` (via synonym match); "I'd like an appointment" → `direct_booking_department`, `department is None`, prompts for a choice; "I have a fever and cough" → `symptom_triage`.
- A doctor-name fuzzy match below the 85 threshold (e.g. a name that shares only a couple of letters) does not falsely resolve — falls through to department or symptom-triage matching instead.
- A known high-severity DDXPlus test fixture produces `triage_level <= 2`; a known mild one produces `triage_level >= 4`.
- Forcing `confidence < 0.3` in a test fixture asserts `triage_level` is capped at 3 regardless of the raw KG severity for that condition.
- No test fixture ever produces a `confidence` value outside `[0, 1]`; every direct-booking test fixture ends with `confidence is None` and `triage_level == 5`.
- Full symptom-triage integration test: intake → clinical loop → classify → explain (asserts `chat_session.status == 'completed'`) → prompt_booking → fetch_slots (asserts ordering by rating) → confirm_slot → payment (asserts `appointment` row exists with matching `chat_session_id`).
- Full direct-booking-by-department integration test: intake → "book with a cardiologist" → fetch_slots → confirm_slot → payment (asserts `chat_session.final_diagnosis == "Patient-requested direct booking"` and `appointment.triage_level == 5`).
- Full direct-booking-by-doctor integration test: intake → "book with Dr. X" → fetch_slots_for_doctor (asserts only that doctor's slots returned) → confirm_slot → payment.
- MedQuAD index build: after running the build script, `index.ntotal > 1000`; a sample query against a known focus term returns that entry's chunks within the top 3 results.
- MedDialog index build: same `ntotal` sanity check; a spot-check retrieval returns a `doctor_followup` string in the metadata, not a raw patient statement, and that string contains a `?`.
- Killing the FAISS index files before startup: graph still completes end-to-end on both paths, response includes the degraded-mode disclaimer, server doesn't crash.
