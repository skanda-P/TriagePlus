# TriagePlus — Codebase Issues, Fixes, and Improvement Instructions

This document consolidates every issue found across the RAG pipeline, LLM orchestration, and system design — including the two newly reported symptoms: (1) the LLM picking up irrelevant/unrelated information from retrieved conversation chunks, and (2) wrong department assignment for common, non-specific complaints (fever, headache, etc.) that should route to General Medicine.

---

## Section 1 — Issues Found and Suggested Fixes

### 1.1 RAG contamination: LLM absorbs unrelated details from retrieved conversation chunks *(newly reported)*

**Where:** `RAG/ml_training/gemini_inference.py`, `infer_department_interactive()`, lines ~285–295.

**Root cause (three compounding problems):**

1. **No framing around the retrieved text.** `rag_context` is inserted into the system prompt as:
   ```
   Relevant past cases for guidance:
   {cases_text}
   ```
   There is no instruction telling the model that this text describes a *different, unrelated patient* — nothing says "do not copy facts, diagnoses, medications, or specifics from these examples into your reply about the current patient." A local 3B model like `llama3.2` has no reliable way to distinguish "stylistic example" from "information about this conversation" unless it's told explicitly. It will readily blend details from the retrieved case into its response to the real patient.

2. **No per-chunk relevance filtering.** `_search_faiss()` only checks `max_score` (the single best hit) against a threshold (`< 0.35` → discard everything). If the top hit clears the bar but hits #2 and #3 don't, all three are still concatenated into `cases_text` verbatim — so weakly-relevant or outright irrelevant chunks routinely ride along with one genuinely relevant one.

3. **No content curation or length cap on what's injected.** Chunks are raw windowed dialogue (up to ~6 lines each), inserted unmodified. Nothing strips diagnosis mentions, medication names, or specialty-specific details that don't apply to the current patient. Combined with Index A's specialty imbalance (see 1.3), a fever/general patient can end up with Respiratory-flavored "guidance" text injected verbatim — plausible-sounding, specific, and completely irrelevant to what the patient actually said.

**Suggested fix:**
- Add an explicit non-transfer instruction directly around `rag_context`, e.g.: *"The cases below are unrelated prior patients, shown only to illustrate question phrasing and style. Do not state, imply, or ask about any specific fact, diagnosis, medication, or detail from them unless the current patient has said it themselves."*
- Filter retrieved chunks **individually** against the relevance threshold, not just the max score — drop any chunk whose own score falls below the cutoff even if another chunk in the same top-k clears it.
- Cap injected text length (e.g., hard cap ~300–400 characters total across all chunks) and consider stripping the case's own diagnosis/department mentions from the injected text at index-build time, since guidance should be about *how to ask*, not *what the answer is*.
- Longer-term: this is compounded by Index A's imbalance (1.3) and by the fact that "guidance" quality is only as good as the corpus — see Section 2 for the concrete rebalancing and chunking plan.

---

### 1.2 Wrong department for common, non-specific complaints (fever, headache, etc.) *(newly reported)*

**Where:** `RAG/ml_training/gemini_inference.py`, `infer_department_final()`, lines ~355–371.

**Root cause (three compounding problems):**

1. **Index B (MedQuAD) has a structural bias toward specific/rare conditions.** MedQuAD is built from NIH/NLM disease-topic pages — it has articles on Migraine, Meningitis, Dengue, Influenza, etc., but nothing genuinely written to say "isolated fever/headache with no other findings is usually benign and belongs in General Medicine." A retrieval query for "fever" or "headache" will surface disease-specific articles (because that's all the corpus contains), and that specific, confident-sounding "Medical knowledge reference" text gets fed straight into the classification prompt — actively pulling the LLM toward a specialty (e.g., Neurology for headache) rather than the correct conservative default.

2. **The classification prompt has no explicit default-to-General-Medicine instruction.** The prompt says "assign to the most appropriate department" but never states that ambiguous, non-specific, or red-flag-free presentations should default to General Medicine. Combined with #1, the LLM is being handed specific disease content and asked to pick *some* specialty — General Medicine, which isn't "about" any disease in particular, is structurally disadvantaged in that framing.

3. **The existing confidence/score fallback doesn't catch this failure mode.** `infer_department_final` does fall back to General Medicine when `confidence < 0.6` or `max_score < 0.3` — but a plausible-sounding, well-scoring MedQuAD hit about (say) Migraine, combined with a self-reported high LLM confidence, sails right past that check even though the actual patient only reported "headache, mild, since this morning, no other symptoms."

4. **Index A's imbalance compounds this upstream, during the interactive phase.** Because Index A is 96% Respiratory/Musculoskeletal (see 1.3), a fever patient's interactive-turn retrieval is likely to surface Respiratory "guidance" cases, steering the generation model's follow-up questions toward respiratory-specific territory (cough, breathing difficulty) and populating `associated_symptoms` accordingly — which then feeds a skewed summary into the final classification step.

**Suggested fix:**
- Add an explicit instruction to the final classification prompt: *"If the symptoms are common, non-specific, and show no red-flag or specialty-defining features (e.g., isolated fever, mild headache, general fatigue, common cold symptoms), assign General Medicine rather than a specific specialty."* Pair this with 2–3 concrete in-prompt examples (few-shot) showing exactly this reasoning.
- Add a lightweight **rules-based pre-check** ahead of the LLM call: a small keyword/pattern table mapping common, low-specificity presenting complaints (fever alone, headache alone, fatigue, mild cold symptoms, sore throat) to General Medicine by default, only overridden if the patient's summary contains a specialty-defining feature (e.g., "worst headache of my life," "fever + neck stiffness," "chest pain"). This gives you a deterministic, auditable safety net that doesn't depend on the LLM inferring the right default on its own — see Section 2 for the concrete design.
- Ensure Index B actually contains general/benign-presentation content, not just disease-specific pages — pull in general-practice-oriented sources (e.g., general symptom-triage guidance) so the retrieval step has something to point toward *besides* a specific disease.
- Fix Index A's imbalance (1.3) so the interactive phase doesn't pre-bias the conversation toward Respiratory before the final classification even runs.

---

### 1.3 Most downloaded datasets are never actually embedded

**Where:** `RAG/ml_training/setup_and_train.py`, `build_faiss_indexes()`.

**Issue:** `meddialog_en_train.jsonl`/`en_medical_dialog.json` (~550MB, real doctor-patient dialogue) and `Symptom2Disease.csv` are downloaded and preprocessed but never read by the indexing function. Only `conversations/*.txt` (Index A) and `medquad.jsonl` (Index B) are actually embedded, despite the module docstring claiming Index A includes "conversation + MedDialog."

**Fix:** See the full chunking/embedding redesign in Section 2 — MedDialog and Symptom2Disease need to be brought into Index A and Index B respectively, with dataset-appropriate chunking (not just dumped in raw).

---

### 1.4 Index A is severely class-imbalanced

**Where:** `RAG/faiss/index_a_meta.json` (built from `conversations/*.txt`).

**Measured distribution:**
```
Total chunks: 6,624
  Respiratory:      5,091 (76.8%)
  Musculoskeletal:  1,256 (19.0%)
  Gastroenterology:   121
  Cardiology:         116
  Dermatology:         21
  General:             19
```
Only 6 of the app's 17 defined departments have any representation, and two specialties account for 96% of the index.

**Fix:** Cap chunks per specialty at index-build time (e.g., reservoir-sample to a configurable max per folder) so no specialty can numerically dominate retrieval regardless of query content. Add genuine General Medicine content (fever, headache, fatigue, common cold — exactly the complaints currently misrouted per 1.2) so the interactive phase has real exemplars to draw on for these cases instead of none.

---

### 1.5 Index B (MedQuAD) embeddings are silently truncated

**Where:** `RAG/ml_training/setup_and_train.py`, MedQuAD embedding step; `all-MiniLM-L6-v2` has a 256-token limit.

**Measured:**
```
Total entries: 16,412
  Mean length: 1,355 characters
  Max length:  29,090 characters
  >1,000 chars: 7,694 (47%)
  >2,000 chars: 2,676 (16%)
```
Nearly half of all entries are silently truncated before embedding (no chunking exists — `question+answer` is embedded as one blob), while the *full* untruncated text is still injected into the LLM prompt at retrieval time.

**Fix:** Split question from answer; chunk the answer using sentence-aware or semantic chunking. Full method detailed in Section 2 and in the prior chunking-plan document.

---

### 1.6 Blocking, CPU-bound call inside async request handling

**Where:** `RAG/ml_training/gemini_inference.py`, `_search_faiss()`, called from `infer_department_interactive()` (an `async def`).

**Issue:** `embedder.encode(...)` is a synchronous, CPU-bound call, not wrapped in `asyncio.to_thread`. On the single-worker uvicorn process this app runs (`run.py`), this call holds the event loop, freezing every other concurrent WebSocket session — including the ping/keepalive loop — for its duration. `infer_department_final` correctly wraps its blocking work in `asyncio.to_thread`; the interactive path, which runs on every single turn, does not.

**Fix:** Wrap `_search_faiss()` (or at minimum `embedder.encode`) in `asyncio.to_thread` everywhere it's invoked from async code.

---

### 1.7 Emergency detection fails open

**Where:** `RAG/ml_training/gemini_inference.py`, `check_emergency_llm()`.

**Issue:**
```python
except Exception as e:
    logger.error(f"Emergency LLM check failed: {e}")
    return False
```
Any failure (Ollama down, malformed JSON, timeout) silently resolves to "not an emergency," with no independent safety net and no signal to the user.

**Fix:** Add an independent keyword/regex check (chest pain, can't breathe, unconscious, severe bleeding, etc.) that runs regardless of LLM availability and ORs with the LLM result, so a model failure can't silently look identical to a confirmed non-emergency.

---

### 1.8 Three sequential LLM calls per turn, no timeouts, and wasted work on non-symptom turns

**Where:** `RAG/ml_training/gemini_inference.py` / `backend/app/api/v1/chat.py`.

**Issue:** Every patient message triggers `check_emergency_llm` → extraction call → generation call, all sequential against the same local Ollama instance, none with a timeout. `check_emergency_llm` also runs on intake fields like name, age, and phone number — not just symptom-related turns — adding unnecessary latency to every step of the conversation.

**Fix:** Add `asyncio.wait_for` timeouts around all three calls with sane fallbacks on timeout. Gate `check_emergency_llm` to only run once the conversation has reached symptom-related states (`INITIAL_SYMPTOM`/`GEMINI_CONVERSATION`), not intake fields.

---

### 1.9 Session state is in-memory, unpersisted, and doesn't resync on reconnect

**Where:** `backend/app/api/v1/chat.py`, `_sessions: dict[str, dict]`; `frontend/src/hooks/useSession.ts`.

**Issue:** A server restart wipes every in-progress session; there's no TTL/eviction (unbounded memory growth); it breaks if the app is ever scaled beyond one worker. Separately, `useSession.ts` persists `session_id` in `sessionStorage` so a page refresh reconnects to the same backend session, but the frontend's message list is not persisted — so on reconnect past `NAME_ENTRY`, the user sees a blank chat while the backend still expects an answer to a question it already silently asked.

**Fix:** Move session state to a persistent store (SQLite for a single-instance deploy, Redis if scaling to multiple workers). Add a reconnect path that replays `state["history"]` to the client when an existing session is resumed.

---

### 1.10 Documentation and dependency drift

**Where:** `README.md`, `requirements.txt`.

**Issue:** `README.md` still describes the app as "Powered by Gemini 2.5 Flash" with `GEMINI_API_KEY` setup instructions; the live path is 100% Ollama/`llama3.2`. `requirements.txt` still lists `google-generativeai`, used only by dead legacy scripts (`gemini_inference.py.bak`, `run_inference.py`).

**Fix:** Update README and requirements to reflect the actual Ollama-based stack; delete or clearly quarantine the dead Gemini scripts.

---

## Section 2 — Exact Instructions to Improve the Codebase and Final Product

Work through these in order — each one is scoped to be independently implementable and testable.

### Step 1: Fix RAG contamination in the interactive prompt
**File:** `RAG/ml_training/gemini_inference.py`, inside `infer_department_interactive()`.

1. In `_search_faiss()`, change the loop that builds `text_blob` so each chunk is checked against the relevance threshold **individually**, not just the max:
   ```python
   MIN_CHUNK_SCORE = 0.35
   for i, idx in enumerate(I[0]):
       if idx != -1 and idx < len(meta):
           score = float(D[0][i])
           if score < MIN_CHUNK_SCORE:
               continue  # drop weak matches individually, don't let a strong #1 drag in a weak #2/#3
           if score > max_score:
               max_score = score
           results.append(meta[int(idx)])
           text_blob += f"- {meta[int(idx)].get('text', '')[:250]}\n"  # per-chunk length cap
   ```
2. Replace the `rag_context` block in `system_prompt` with an explicitly framed version:
   ```python
   rag_context = (
       f"\nUNRELATED PAST CASES (for question-phrasing style only — "
       f"do NOT state, imply, or ask about any specific fact, diagnosis, or detail "
       f"from these unless the current patient has said it themselves):\n{cases_text}"
   ) if cases_text else ""
   ```
3. Add a matching negative instruction to the `instruction` string:
   ```python
   instruction = (
       "Do NOT ask about any field in already_collected. "
       "Ask about exactly one field from still_needed next, or politely ask for any other details if still_needed is empty. "
       "Do not copy or reference specific details from the unrelated past cases below — they are style examples only."
   )
   ```

### Step 2: Fix General Medicine misclassification for common complaints
**File:** `RAG/ml_training/gemini_inference.py`, inside `infer_department_final()`.

1. Add a deterministic pre-check **before** the LLM call — a small keyword table for low-specificity complaints that should default to General Medicine unless a red-flag/specialty-defining term is also present:
   ```python
   GENERAL_MEDICINE_DEFAULTS = {
       "fever", "headache", "fatigue", "tiredness", "sore throat",
       "common cold", "mild cough", "body ache", "general weakness",
   }
   SPECIALTY_OVERRIDE_TERMS = {
       "worst headache", "neck stiffness", "chest pain", "vision loss",
       "seizure", "severe", "unconscious", "difficulty breathing",
       "blood in", "pregnant", "rash spreading",
   }

   def _rule_based_department(summary: str) -> str | None:
       s = summary.lower()
       if any(term in s for term in SPECIALTY_OVERRIDE_TERMS):
           return None  # let the LLM decide, this needs real triage judgment
       if any(term in s for term in GENERAL_MEDICINE_DEFAULTS):
           return "General Medicine"
       return None
   ```
2. Call it before the LLM request and short-circuit if it returns a department:
   ```python
   rule_dept = _rule_based_department(summary)
   if rule_dept:
       # still run the LLM for urgency scoring/reasoning, but keep the department authoritative
       ...
   ```
3. Regardless of the rule-based check, add an explicit default-to-General-Medicine instruction plus few-shot examples directly in the prompt sent to the LLM:
   ```
   If the symptoms are common, non-specific, and show no red-flag or specialty-defining
   features (e.g., isolated fever, mild headache, general fatigue, common cold symptoms),
   assign "General Medicine" rather than a specific specialty.

   Example: "Fever for 2 days, mild fatigue, no other symptoms." → General Medicine.
   Example: "Headache since this morning, no nausea, no vision changes, no neck stiffness." → General Medicine.
   Example: "Worst headache of my life, sudden onset, blurred vision." → Neurology.
   ```
4. Keep the existing `conf < 0.6 or max_score < 0.3` fallback as a second safety net — it's correct, just insufficient alone.

### Step 3: Rebalance Index A and enrich its General Medicine coverage
**File:** `RAG/ml_training/setup_and_train.py`, `build_faiss_indexes()`.

1. Add a per-specialty cap when walking `conversations/*.txt` (reservoir sampling or simple truncation after shuffling), e.g. max 150 chunks per specialty folder.
2. Add or expand a `General/` folder of synthetic conversations specifically covering fever, headache, fatigue, and common cold presentations, following the same `D:`/`P:` script format as the existing folders, so the interactive phase has real General Medicine exemplars instead of 19 chunks' worth.

### Step 4: Bring MedDialog and Symptom2Disease into the indices
**File:** `RAG/ml_training/setup_and_train.py`, new/expanded `build_faiss_indexes()`.

Full dataset-specific chunking design (already scoped in the prior architecture review — repeated here for completeness):

- **MedDialog → Index A:**
  - Embed each consultation's `description` field alone (short, query-shaped); store the first ~8 turns of `utterances` as the returned payload (parent-child / small-to-big retrieval).
  - Additionally, sliding-window `utterances` (3-turn window, 1-turn overlap) as a second record type, tagged separately, for mid-conversation follow-up style guidance.
  - Exact-dedup on normalized `description` text; stratified-subsample to ~3,000–5,000 total entries so MedDialog doesn't drown out the synthetic corpus.
- **MedQuAD → Index B:**
  - Embed `question` alone as one record type.
  - Chunk `answer` with sentence-aware splitting (~180 tokens, ~20% overlap) as a baseline; semantic (embedding-similarity-based) chunking as a stronger follow-up, since MedQuAD answers have implicit topic sub-sections.
  - Cap what gets concatenated into the final prompt (e.g., top-3 chunks, ~800 character hard cap).
- **Symptom2Disease.csv → Index B:**
  - Embed `text` directly (already short enough, no chunking needed); tag with `source: "symptom2disease"` and `disease_label` so it can be weighted differently from MedQuAD's more encyclopedic content later.
- Every new record, regardless of source, uses the unified metadata schema (`text`, `embed_text`, `source`, `specialty`, `extra`) so retrieval provenance stays debuggable.

### Step 5: Fix the blocking call
**File:** `RAG/ml_training/gemini_inference.py`, `_search_faiss()` call sites.

Wrap the call inside `infer_department_interactive()`:
```python
top_a, cases_text, max_score, t_embed, t_faiss = await asyncio.to_thread(
    _search_faiss, query, index_a, meta_a, top_k=3
)
```
(Requires `import asyncio` at the top of the file if not already present.)

### Step 6: Add an emergency-detection safety net
**File:** `RAG/ml_training/gemini_inference.py`, `check_emergency_llm()`.

Add a keyword check that runs independently of the LLM and ORs with its result:
```python
EMERGENCY_KEYWORDS = [
    "can't breathe", "cannot breathe", "chest pain", "unconscious",
    "severe bleeding", "not breathing", "stroke", "heart attack",
]

def _keyword_emergency_check(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)

async def check_emergency_llm(text: str) -> bool:
    if _keyword_emergency_check(text):
        return True
    try:
        ...  # existing LLM check
    except Exception as e:
        logger.error(f"Emergency LLM check failed: {e}")
        return False  # keyword check above already covers the fail-open gap for obvious cases
```

### Step 7: Add timeouts and gate the emergency check to relevant states
**Files:** `RAG/ml_training/gemini_inference.py`, `backend/app/api/v1/chat.py`.

1. Wrap all three Ollama calls (`check_emergency_llm`, extraction, generation) in `asyncio.wait_for(..., timeout=<N seconds>)` with a graceful fallback message on timeout.
2. In `chat.py`, only call `check_emergency_llm` when `fsm_state` is `INITIAL_SYMPTOM` or `GEMINI_CONVERSATION` — skip it during `NAME_ENTRY`/`AGE_ENTRY`/`GENDER_ENTRY`/`PHONE_ENTRY`.

### Step 8: Persist session state and fix reconnect
**Files:** `backend/app/api/v1/chat.py`, `frontend/src/hooks/*`.

1. Replace `_sessions: dict[str, dict]` with a SQLite-backed store (or Redis if you expect multi-worker scaling) keyed by `session_id`.
2. On WebSocket connect, if `session_id` already has stored state, send the client its full `history` before waiting for the next message, so a page refresh doesn't leave the user staring at a blank chat mid-conversation.

### Step 9: Reconcile documentation
**Files:** `README.md`, `requirements.txt`.

1. Update `README.md` to describe the actual Ollama/`llama3.2` stack and remove the `GEMINI_API_KEY` setup instructions.
2. Remove `google-generativeai` from `requirements.txt`, or explicitly mark `gemini_inference.py.bak` and `run_inference.py` as archived/unused if you want to keep them for reference.

### Validation checklist after implementing the above
- Re-run the distribution check from Section 1.4 against the rebuilt `index_a_meta.json` — confirm no specialty exceeds a configured share (e.g. no more than ~25%) and General Medicine has non-trivial representation.
- Re-run the token-length audit against the rebuilt `index_b_meta.json` — confirm no `embed_text` exceeds the model's token limit.
- Manually test the exact reported failure cases (fever-only, headache-only) end-to-end and confirm they now resolve to General Medicine with correct reasoning in the `diagnostic` payload.
- Manually test a couple of interactive turns and inspect the `prompt` field in the `result` payload to confirm injected `cases_text` is short, filtered, and doesn't leak into the model's actual reply to the patient.
