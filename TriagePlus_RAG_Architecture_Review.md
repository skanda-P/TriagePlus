# TriagePlus — RAG System Architecture Review & Embedding/Chunking Plan

**Scope:** `RAG/` pipeline, `backend/app/api/v1/chat.py`, `RAG/ml_training/gemini_inference.py`, `RAG/ml_training/setup_and_train.py`, and the datasets feeding FAISS Index A and Index B.

**Repo reviewed:** `github.com/skanda-P/TriagePlus` (main branch, current snapshot).

---

## 1. Executive summary

The live implementation is a simpler stack than the aspirational architecture (no scispaCy/SapBERT/BiomedBERT in the running code — just `all-MiniLM-L6-v2` embeddings + FAISS + local `llama3.2` via Ollama). Within that simpler stack, the most important findings are:

1. **Most of the downloaded datasets are never actually embedded.** MedDialog (~550MB across two files) and `Symptom2Disease.csv` are downloaded/preprocessed but excluded from `build_faiss_indexes()`. Only the synthetic `conversations.zip` and MedQuAD reach FAISS.
2. **Index A (conversation cases) is severely class-imbalanced** — 2 of 6 represented specialties account for ~96% of all chunks, and only 6 of the app's 17 defined departments have any representation at all.
3. **Index B (MedQuAD) embeddings are being silently truncated** — no chunking exists, so `all-MiniLM-L6-v2`'s 256-token limit cuts off the majority of longer answers before they're embedded, while the *full* untruncated text still gets injected into the LLM prompt at retrieval time.
4. **A blocking, CPU-bound call sits inside async request-handling code**, which under concurrent load freezes every other active chat session on the server.
5. **Emergency detection fails open** — any exception in the LLM-based emergency classifier silently resolves to "not an emergency," with no independent safety net.
6. **Session state is in-memory, unpersisted, and doesn't resync on reconnect** — a page refresh mid-triage produces a blank chat window while the backend still expects the user to answer a question it already silently asked.
7. Documentation (`README.md`, `requirements.txt`, module docstrings) has drifted from the actual implementation (still references Gemini 2.5 Flash and `google-generativeai`, both replaced by Ollama/llama3.2).

The remainder of this document details each finding and lays out a concrete chunking/embedding redesign for Index A and Index B.

---

## 2. Current architecture as implemented

```
Patient message (WebSocket)
        │
        ▼
check_emergency_llm()  ── LLM call (llama3.2, JSON) ── runs on EVERY message, every FSM state
        │
        ▼
[FSM: NAME_ENTRY → AGE_ENTRY → GENDER_ENTRY → PHONE_ENTRY → INITIAL_SYMPTOM/GEMINI_CONVERSATION → RECOMMENDING → BOOKING]
        │
        ▼ (during INITIAL_SYMPTOM / GEMINI_CONVERSATION)
Extraction LLM call (llama3.2, JSON)  → merges into running "slots" dict (never erases, only fills)
        │
        ▼
_search_faiss(query, Index A)  → "past cases" context for the system prompt
        │
        ▼
Generation LLM call (llama3.2, streamed)  → next question to patient
        │
        ▼ (once REQUIRED_SLOTS satisfied or MAX_INTERACTIVE_TURNS hit)
infer_department_final()
        │
        ▼
_search_faiss(summary, Index B)  → "medical knowledge" context
        │
        ▼
Final LLM call (llama3.2, JSON)  → department + confidence + urgency_score
```

**Session state:** `_sessions: dict[str, dict]` in `chat.py`, in-process memory only, keyed by client-generated UUID.

**Indices:**
- **Index A** (`index_a.faiss` + `index_a_meta.json`): built from `conversations/*.txt` only — despite the `setup_and_train.py` docstring claiming "conversation + MedDialog."
- **Index B** (`index_b.faiss` + `index_b_meta.json`): built from `medquad.jsonl` only (MedlinePlus XML branch exists in code but is never populated — the file doesn't exist, so the code silently no-ops).

Both indices are `faiss.IndexFlatIP` (exact, brute-force cosine similarity via inner product on normalized vectors) — fine at current size, but linear-time, so it does not scale gracefully if the corpus grows by 10–100x, which the chunking plan below implies it will.

---

## 3. Findings — data & embeddings

### 3.1 Datasets downloaded vs. datasets actually embedded

| Dataset | File(s) | Size | Downloaded/processed? | Embedded into FAISS today? |
|---|---|---|---|---|
| Synthetic scripted conversations | `conversations.zip` → `conversations/*.txt` | small | Yes | **Yes** — Index A |
| MedDialog (English) | `en_medical_dialog.json`, `meddialog_en_train.jsonl` | ~550 MB combined | Yes | **No** |
| MedQuAD | `medquad.csv`, `medquad.jsonl` | ~22 MB | Yes | **Yes** — Index B |
| Symptom2Disease | `Symptom2Disease.csv` | ~224 KB | Yes (only for a separate, currently-disabled eval-sample collector) | **No** |
| MedlinePlus XML | `medlineplus.xml` | n/a | No — download disabled in code | No (branch is dead) |

**Root cause:** `build_faiss_indexes()` in `setup_and_train.py` only walks `conversations/*.txt` for Index A and `medquad.jsonl` (+ the never-present MedlinePlus XML) for Index B. `meddialog_en_train.jsonl` is downloaded and converted to JSONL by `download_hf_datasets()` but never read by the indexing function. `Symptom2Disease.csv` is read only inside `collect_real_samples()`, which feeds an offline prediction-evaluation script (`run_department_inference`) that is commented out in `__main__`.

**Implication:** the module docstring ("Builds FAISS Index A (conversation + MedDialog)") does not match the code. Whether or not excluding MedDialog was an intentional quality decision, it isn't documented as one — it reads as an oversight, and it means ~550MB of the highest-volume, most realistic doctor-patient dialogue data in the project is currently inert.

### 3.2 Index A — severe class imbalance

Measured directly from `index_a_meta.json`:

```
Index A total chunks: 6,624
  Respiratory:       5,091  (76.8%)
  Musculoskeletal:   1,256  (19.0%)
  Gastroenterology:    121
  Cardiology:          116
  Dermatology:          21
  General:              19
```

- Only **6 of the 17** departments defined in `DEPARTMENTS` (`gemini_inference.py`) have any representation in Index A at all.
- Two specialties account for **96%** of the index.
- Nearest-neighbor search over this corpus will surface Respiratory/Musculoskeletal chunks for a large share of queries regardless of what the patient actually describes, because they dominate numerically — not because they're semantically closer. This context is injected directly into the interactive-turn system prompt as "Relevant past cases for guidance," so the bias propagates into the live conversation, not just into logging/analytics.
- This is a **corpus-construction problem**, not an embedding-model problem — no change of embedding model fixes it.

### 3.3 Index B — no chunking, silent truncation

Measured directly from `index_b_meta.json`:

```
Index B total entries: 16,412
  Mean length: 1,355 characters
  Max length:  29,090 characters
  Entries > 1,000 chars: 7,694  (47%)
  Entries > 2,000 chars: 2,676  (16%)
```

- `all-MiniLM-L6-v2` has a hard limit of 256 word-piece tokens (~1,000–1,200 characters). Anything longer is silently truncated by the tokenizer *before* embedding — no error, no warning.
- Current build embeds `f"{question}\n{answer}"` as a single blob per row, with no chunking. For the 47% of entries over 1,000 characters, the embedding reflects only the opening portion of the answer (and possibly just the question, for the longest ones).
- At retrieval time, the *full, untruncated* text (up to 29,090 characters — several thousand tokens) is concatenated into `knowledge_text` and injected into the final-triage LLM prompt. This creates two independent problems from one root cause:
  - **Retrieval quality**: content deep in a long answer can never be matched, because it was never actually embedded.
  - **Latency/prompt bloat**: a single retrieved "chunk" can be tens of thousands of characters, inflating the local LLM's context and processing time for what's supposed to be a fast final classification step.

### 3.4 Duplicated/legacy data artifacts
- `medquad.csv` and `medquad.jsonl` are both present (22MB apiece) — the JSONL is derived from the CSV by `download_hf_datasets()`, so this is redundant storage, not a second dataset.
- `RAG/ml_training/gemini_inference.py.bak` and `RAG/ml_training/run_inference.py` are dead legacy scripts still importing the deprecated `google.generativeai` SDK (superseded by `google-genai` upstream, and by Ollama in this project's live path). `recover_faiss.py` contains a hardcoded personal Windows path (`D:\BTech\hackathons\...`) — a one-off recovery utility committed as-is, not portable.

---

## 4. Findings — LLM / RAG orchestration

### 4.1 Blocking call inside async request handling
`_search_faiss()` calls `embedder.encode(...)` synchronously. It is invoked directly inside `infer_department_interactive`, which is `async def`, **without** wrapping the call in `asyncio.to_thread`. Since `run.py` runs a single-worker uvicorn process, this CPU-bound call holds the event loop — meaning one patient's embedding computation freezes every other concurrent WebSocket connection on the server, including the keepalive ping loop.

By contrast, `infer_department_final` (the end-of-conversation call) *does* correctly wrap its blocking work in `asyncio.to_thread`. The interactive path — which runs on every single turn, far more frequently than the final path — is the one that's wrong.

### 4.2 Three sequential LLM round-trips per user turn, no timeouts
Per patient message during `INITIAL_SYMPTOM`/`GEMINI_CONVERSATION`:
1. `check_emergency_llm` — full LLM call, JSON-formatted.
2. Extraction call — second LLM call, JSON-formatted.
3. Generation call — third LLM call, streamed.

All three hit the same local Ollama instance serially; Ollama serializes generation on typical local hardware, so concurrent users queue behind each other's three-call chains. **None of the three calls has a timeout** (no `asyncio.wait_for` or equivalent) — a hung Ollama call hangs the WebSocket indefinitely.

`check_emergency_llm` runs on **every** message regardless of FSM state — including `NAME_ENTRY`, `AGE_ENTRY`, `GENDER_ENTRY`, `PHONE_ENTRY` — spending a full LLM round-trip to check whether someone's typed age or phone number constitutes a medical emergency.

### 4.3 Emergency detection fails open
```python
except Exception as e:
    logger.error(f"Emergency LLM check failed: {e}")
    return False
```
Any exception (Ollama down, malformed JSON, timeout) silently resolves to "not an emergency," and the conversation proceeds normally with no indication to the user or any fallback check. For a triage product, a failure in the safety classifier should never be indistinguishable from a confirmed negative result. There is no independent (e.g., keyword/regex) safety net that runs regardless of LLM availability.

### 4.4 Confidence/urgency fallback logic (working as intended — noted for completeness)
`infer_department_final` does correctly downgrade to "General Medicine" when `confidence < 0.6` **or** `max_score < 0.3` (combining LLM self-reported confidence with retrieval quality) — this is a reasonable conservative design and is called out here as something that does *not* need to change.

---

## 5. Findings — system design / reliability

### 5.1 In-memory, unpersisted session state
`_sessions: dict[str, dict] = {}` in `chat.py` is the entire persistence layer for active triage conversations.
- A server restart or redeploy wipes every in-progress session.
- No TTL or eviction — abandoned sessions accumulate for the lifetime of the process (unbounded memory growth under real traffic).
- Breaks immediately if the app is ever scaled to more than one uvicorn worker or process (e.g., Render's standard scaling path), since the dict isn't shared across processes.

### 5.2 No state resync on WebSocket reconnect
`useSession.ts` persists `session_id` in `sessionStorage`, so a page refresh reconnects to the *same* backend session — but the frontend's chat message list lives only in an in-memory zustand store with no persistence. If `fsm_state` has already advanced past `NAME_ENTRY`, `patient_ws` sends nothing on reconnect (the "send welcome" branch only fires for a brand-new session). Result: the user sees a blank chat window while the backend still expects them to answer a question it already silently asked and will never repeat.

### 5.3 Documentation/dependency drift
- `README.md` still describes the app as "Powered by Gemini 2.5 Flash" and instructs users to set `GEMINI_API_KEY` — the live code path is 100% Ollama/`llama3.2`.
- `requirements.txt` still lists `google-generativeai`, used only by the two dead legacy scripts noted in §3.4.

---

## 6. Proposed embedding & chunking redesign

### 6.1 Design principles

1. **Respect the embedding model's real token limit.** `all-MiniLM-L6-v2` truncates at 256 word-piece tokens (~1,000–1,200 characters). Target chunk sizes at 70–80% of that ceiling (≈180–200 tokens), since medical terminology (long Latin/Greek-derived terms) tokenizes denser than everyday English of the same character length.
2. **Separate "what gets embedded" from "what gets returned" wherever they differ in length** (small-to-big / parent-child retrieval). Embed a short, clean, query-shaped piece of text; store a longer, more informative piece of text as the payload returned to the LLM once that chunk is retrieved. This is the core fix to the MedQuAD truncation problem and the natural fit for MedDialog's `description` → `utterances` structure.
3. **Every chunk, in either index, uses one unified metadata schema**, so retrieval provenance is debuggable and future source-weighted reranking doesn't require re-embedding:
   ```json
   {
     "text": "<content actually shown to the LLM>",
     "embed_text": "<content actually embedded, if different from text>",
     "source": "synthetic_conv | meddialog_desc | meddialog_turnwin | medquad_question | medquad_answer_chunk | symptom2disease",
     "specialty": "<string or null>",
     "extra": { "...source-specific fields: focus_area, dialogue_id, disease_label, etc." }
   }
   ```
4. **Size mismatch between sources must be actively managed, not left to chance.** The synthetic conversation corpus is ~6.6K chunks; MedDialog alone is 257K consultations. Embedding it raw would replace today's imbalance (Respiratory dominating) with a worse one (MedDialog drowning out everything else). Dedup and stratified subsampling are part of the chunking plan, not a follow-up task.
5. **Per-user directive for this redesign:** `conversations/*.txt` + MedDialog → **Index A**. MedQuAD + Symptom2Disease (+ MedlinePlus if ever enabled) → **Index B**.

### 6.2 Index A — conversational / case-pattern retrieval

#### A1. Synthetic scripted conversations (`conversations/*.txt`)
- Keep the existing sliding-window scheme (window = 6 lines / ~3 turns, step = 4, one-turn overlap) — the source data is small and clean, this isn't broken.
- Tag every resulting chunk `"source": "synthetic_conv"` under the unified schema.
- **Fix the imbalance at build time**: cap chunks per specialty (e.g., reservoir-sample down to a configurable max, such as 150 chunks per specialty) so no single folder (today: Respiratory at 5,091) can numerically dominate nearest-neighbor results regardless of query content.

#### A2. MedDialog (`meddialog_en_train.jsonl`) — dual-granularity chunking
Schema per consultation (HF `UCSD26/medical_dialog`, `processed.en` config):
```json
{"description": "<short first-person patient blurb>", "utterances": ["patient: ...", "doctor: ...", ...]}
```

**(a) Opening-complaint exemplars — embed `description`, store a windowed excerpt of `utterances` as payload.**
`description` is structurally almost identical to what a real patient types at `INITIAL_SYMPTOM` — short, first-person, symptom-first. That makes it the highest-fidelity query-side match target available in the whole knowledge base. Embed it alone (short, no truncation risk); return the first ~3 exchanges of `utterances` as context so the generation LLM sees a realistic model of how to follow up:
```python
{
  "text": "\n".join(utterances[:8]),   # parent: shown to the LLM
  "embed_text": description,           # child: what gets embedded
  "source": "meddialog_desc",
  "specialty": None,                   # MedDialog has no specialty label — leave explicit, don't fabricate one
}
```

**(b) Mid-conversation follow-up style exemplars.**
Apply the same sliding-window technique used for the synthetic data (3-turn window, 1-turn overlap) directly to `utterances`, tagged `"source": "meddialog_turnwin"`. This matters because retrieval against Index A happens on every interactive turn, not just the first — later turns need "what does a good follow-up question look like" exemplars too, not just opening-complaint matches.

**(c) Dedup and subsample before embedding.**
- Exact-dedup on normalized `description` text (lowercase, strip punctuation) to remove verbatim forum repeats.
- Stratified random subsampling to cap total MedDialog contribution at roughly 3,000–5,000 entries (stratified by `description` length bucket, so short/uninformative entries aren't over-represented in what survives). This keeps Index A's total size in the same order of magnitude as the synthetic corpus, and keeps the brute-force `IndexFlatIP` search fast.

### 6.3 Index B — medical knowledge retrieval

#### B1. MedQuAD — split question from answer; chunk the answer
Schema: `question, answer, source, focus_area` (CSV) — `focus_area` and `source` are usable metadata for future filtering/boosting, not currently used.

Replace the current single blob (`f"{question}\n{answer}"`) with two record types:

**Question — its own entry, embedded and returned as-is:**
```python
{"text": question, "embed_text": question, "source": "medquad_question",
 "extra": {"focus_area": focus_area, "answer_ref": answer_id}}
```

**Answer — chunked, each chunk its own entry, inheriting the parent's metadata.** Two tiers:

- **Baseline (implement first): sentence-aware recursive chunking.** Split on sentence boundaries only (never mid-sentence), target ~180 tokens per chunk, ~20% overlap between consecutive chunks. This alone eliminates the current truncation bug — 47% of today's entries are silently cut off before embedding.

- **Better fit for this dataset: semantic/topic-boundary chunking.** MedQuAD answers are lightly-edited NIH/NLM web pages with implicit sub-headings (visible directly in the sampled data — "How Glaucoma Develops" runs straight into "Open-angle Glaucoma" with no chunk boundary in the raw text). Semantic chunking embeds each sentence, then cuts a new chunk wherever the cosine similarity between consecutive sentences drops below a threshold — finding where the topic actually shifts, rather than slicing at an arbitrary token count. This is a one-time offline cost since embedding is happening anyway:
  ```python
  def semantic_chunks(sentences, embedder, threshold=0.62, max_tokens=220):
      sent_embs = embedder.encode(sentences, normalize_embeddings=True)
      chunks, current = [], [sentences[0]]
      for i in range(1, len(sentences)):
          sim = float(sent_embs[i] @ sent_embs[i-1])
          current_len = sum(len(s.split()) for s in current)
          if sim < threshold or current_len > max_tokens:
              chunks.append(" ".join(current))
              current = [sentences[i]]
          else:
              current.append(sentences[i])
      if current:
          chunks.append(" ".join(current))
      return chunks
  ```

- **Stretch goal (research-backed, higher cost): proposition-based indexing.** Per Chen et al., *"Dense X Retrieval: What Retrieval Granularity Should We Use?"* (2023) — decompose each answer into atomic factual propositions ("Glaucoma risk increases significantly after age 60," "Open-angle glaucoma is the most common form") via an offline LLM batch pass, embed each proposition individually. Their benchmarks show proposition-level retrieval outperforming sentence- and passage-level chunking on QA tasks. This is a real batch job (one LLM call per answer, run once at index-build time, never in the live request path) — recommended as a follow-up once the two simpler methods above are working, not as the first implementation.

**Retrieval-time companion constraint (not chunking, but chunking is undermined without it):** cap what actually gets concatenated into `knowledge_text` at query time — e.g., top-3 chunks, hard cap ~800 characters total — so a good chunking job can't be undone by injecting unbounded retrieved text into the final-triage prompt.

#### B2. Symptom2Disease.csv — no chunking needed, only metadata correction
Schema: `label` (disease), `text` (short first-person symptom paragraph — sampled rows run well under the token limit, no truncation risk). Embed `text` directly:
```python
{"text": text, "embed_text": text, "source": "symptom2disease", "extra": {"disease_label": label}}
```
Tag `"source": "symptom2disease"` distinctly from `"source": "medquad_*"`. These are structurally different kinds of evidence — Symptom2Disease is direct symptom-to-diagnosis pattern matching (strong signal for department/urgency classification), MedQuAD is explanatory encyclopedia text (better suited to populating the `reasoning` field than to driving hard classification). Tagging now enables source-weighted reranking later without re-embedding.

#### B3. MedlinePlus XML (currently disabled/absent — for if it's ever enabled)
Structured (`<summary>`, `<also-called>`, `<see-reference>`) — chunk `summary` the same way as MedQuAD answers. Treat `also-called` as a free synonym table: worth a small query-expansion step so a patient's lay term ("high blood pressure") matches content indexed under the clinical term ("hypertension") — MedlinePlus provides that alias mapping directly.

### 6.4 Embedding model — a decision to make before re-embedding, not after
`all-MiniLM-L6-v2` is fast and adequate, but it's a general sentence-similarity model, not a retrieval-tuned one. Since this redesign introduces genuinely asymmetric embedding (short questions vs. long answer chunks, short descriptions vs. longer turn windows), a retrieval-tuned model — `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` — is a natural fit: both are trained with explicit `"query: "` / `"passage: "` prefixes for exactly this asymmetric case, are similar in size/speed to MiniLM, and typically outperform it on retrieval benchmarks. Not mandatory — everything above works with MiniLM — but worth deciding now, since switching later means re-embedding everything a second time.

### 6.5 Rebuilt indexing pipeline — shape

```
Index A sources → normalize to unified schema → dedup → stratified cap → embed → IndexFlatIP
  - synthetic_conv:       windowed D:/P: turns, per-specialty cap
  - meddialog_desc:       description (embedded) + first-8-turns (payload)
  - meddialog_turnwin:    windowed utterances

Index B sources → normalize to unified schema → embed → IndexFlatIP
  - medquad_question:     question alone
  - medquad_answer_chunk: semantic-chunked answer, inherits question/focus_area metadata
  - symptom2disease:      text as-is, tagged with disease_label
```

### 6.6 Validation plan — how to confirm the embeddings are actually correct this time
- **Token-length audit**: run the tokenizer over every `embed_text` post-chunking; assert max length is under the model's limit. This single check would have caught the current MedQuAD truncation bug immediately.
- **Retrieval spot-checks**: for a set of known symptom queries, print top-5 hits per source type and manually verify topical relevance before trusting the index in the live prompt.
- **Distribution check**: log per-specialty and per-source counts after every build (the same query used in §3.2/§3.3 of this document) and fail the build if any single source or specialty exceeds a configured share of the index — turns today's silent imbalance into a build-time error going forward.

---

## 7. Prioritized fix list (across both reviews)

1. Wrap `embedder.encode()` calls in `asyncio.to_thread` wherever invoked from async code — the actual latency/concurrency bug, highest impact for lowest effort.
2. Add an independent keyword/regex fallback to emergency detection so an LLM failure can't silently look like a confirmed non-emergency.
3. Implement MedQuAD question/answer chunking (§6.3, baseline tier) — fixes both retrieval quality and prompt bloat.
4. Rebalance Index A's synthetic corpus (per-specialty cap) and decide explicitly which departments are supported until more data exists for the rest.
5. Bring in MedDialog and Symptom2Disease per the plan in §6.2/§6.3 — currently-downloaded data that's doing nothing.
6. Persist session state (SQLite/Redis instead of the in-memory dict) and add a reconnect-resync path on the frontend.
7. Reconcile `README.md` / `requirements.txt` with the actual Ollama-based implementation; remove or clearly mark the dead legacy Gemini scripts.
