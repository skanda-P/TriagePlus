# MediGuide — Implementation Plan v2 (Locked Decisions)

This supersedes the v1 review. Three decisions are now locked and they reshape the architecture more than they might look like on the surface:

| Decision | Locked answer | Architectural consequence |
|---|---|---|
| Orchestrator LLM | Llama 3.2, 1B/3B, local CPU | LLM is now the *weakest link* in the system — it must be scoped to narrow, low-risk, cache-friendly tasks. Everything that was "LLM judgment" in v1 needs to become deterministic code or classical ML. |
| Diagnosis/routing | Hybrid: classifier decides, LLM explains via RAG | Confirms and locks in the v1 Node 5/6 split — good, no redesign needed there, just tightened. |
| Own conversation dataset | Raw, unlabeled | Can't be used to train the classifier directly. Repurposed as a RAG/phrasing corpus and a source for red-flag pattern mining — not wasted, just used differently than labeled data would be. |

Everything below is designed around these three facts, not around the generic "cloud LLM, infinite scale" assumptions in v1.

---

## Part 1 — Why a 1B/3B Local Model Changes the Design

This needs to be said plainly because it's the thing most likely to bite you late: **a 1B/3B model on CPU is not a reliable reasoner.** It's fine at short, templated, low-ambiguity generation. It is not fine at multi-step judgment, long-context tracking, or producing well-formed JSON zero-shot. Treat it accordingly:

**What Llama 3.2 1B/3B is allowed to do:**
- Paraphrase a slot-fill question into natural language, from a template + few-shot examples.
- Generate a short (100–150 token) patient-facing explanation from *retrieved facts you hand it* — not from its own knowledge.
- Act as a *secondary* fallback entity extractor when the primary (regex + biomedical NER) comes back empty.

**What it is explicitly NOT allowed to do:**
- Decide whether the conversation is "complete." (Deterministic: check the requirement checklist.)
- Decide whether something is a red flag. (Deterministic: keyword/pattern match, must run even if the LLM is slow or down.)
- Decide the diagnosis or department. (XGBoost classifier's job, per your hybrid decision.)
- Hold long conversation history in its prompt context. (Pass it the current state summary, not the raw transcript.)

**Practical mitigations for CPU latency:**
1. **Use Ollama's structured output mode** (`format` parameter with a JSON schema) for any call where you need parseable output — this is the difference between "usually works" and "have to write a regex parser for broken JSON." Don't rely on prompting alone to get valid JSON from a 1B/3B model; constrain the decoding.
2. **Cap generation length hard** (`num_predict: 80–150` depending on the node) — small models tend to ramble when unconstrained, and every extra token is CPU time you don't have to spare.
3. **Cache phrasing aggressively.** The next-question phrasing for a given `(symptom_category, missing_slot)` pair repeats constantly across conversations — this is a tiny space (maybe 50–100 unique combinations). Generate each once, cache in Redis keyed by `(symptom_category, missing_slot)`, and only fall back to a live LLM call for combinations not yet cached. This turns "call the LLM every turn" into "call the LLM once per unique question type, ever." This is the single biggest lever you have for making a local CPU model feel responsive.
4. **Don't cache the explanation node the same way** — it's genuinely case-specific (depends on which facts were retrieved for this patient), so it has to run live. But it only runs **once per completed conversation**, not once per turn, so the cost is bounded regardless.
5. **Model choice per node:** use 3B for the explanation node (it's called rarely, quality matters more there since it's patient-facing medical communication) and 1B is acceptable for phrasing (called more often, but mostly cache hits anyway, and phrasing quality bar is lower than explanation quality bar). If your machine struggles with 3B latency, fall back to 1B everywhere and lean harder on the template layer under it.
6. **Scalability claim, corrected:** v1 claimed "infinite horizontal scaling" — that assumed a cloud LLM API. With a local Ollama process, your throughput ceiling is one machine's CPU. For a Summer of Innovation demo this is a non-issue (you're not serving concurrent production traffic), but don't carry the "infinite scaling" language into your report — it's now inaccurate and an evaluator who knows the stack will notice. State it honestly: single-node inference, horizontally scalable *if* you later move the LLM to a hosted endpoint, not before.

---

## Part 2 — The Three RAG Arms (as requested)

You asked for a strong RAG system across three jobs. Here's each one, concretely.

### Arm A — Knowledge (clinical facts)
**Purpose:** ground any patient-facing factual statement in a real source, never in the LLM's own (unreliable, at this size) medical knowledge.
**Source:** MedQuad.
**Index:** FAISS Index 1.
**Chunking:** one chunk per Q&A pair (MedQuad is already naturally chunked this way — don't re-chunk it).
**Embedding model:** `bge-small-en-v1.5`, run locally. At 1B/3B-local scale you're already CPU-constrained; don't add an embedding API call into the hot path. `bge-small` is ~130MB, fast on CPU, and good enough for this retrieval task — you don't need a large embedding model for a domain-tagged corpus this size.
**Metadata stored per chunk:** `{question, answer, specialty_tag}`. The `specialty_tag` is what lets the explanation node retrieve facts relevant to *this patient's predicted department specifically*, not just topically similar facts. Build this tag with keyword matching against your department taxonomy first (fast, free), and only fall back to an LLM-assisted tag for chunks the keyword pass can't confidently label — cache these tags at index-build time, this is a one-time offline cost, not a runtime one.
**Retrieval at runtime:** top-3 chunks filtered by `specialty_tag == predicted_department`, then re-ranked by cosine similarity to the patient's symptom summary.

### Arm B — Helping ask the right questions (Next-Best-Question)
This is two components working together, and given your dataset situation, you have **two viable paths** depending on whether you get DDXPlus:

**B1. The requirement graph (what to ask, and in what order)**
- *If you download DDXPlus:* build the graph from its evidence/pathology structure — it already encodes which symptoms require which follow-up evidence, which is exactly the "Symptom → Requires Info" structure you need. This is the higher-quality path.
- *If you don't:* derive the graph from **MedDialog + your own transcripts**, using an LLM-assisted extraction pass (Gemini or any capable model *you already have access to for offline batch processing* — this doesn't have to run on your local Llama, since it's a one-time offline data-prep job, not a runtime dependency) to pull out `(symptom, requires_info, question_asked)` triples from real doctor dialogues, followed by the same validation gate from v1: whitelist the relation types, sample 5–10% for manual review before trusting the graph in production.
- **Recommendation:** get DDXPlus if you can. Symptom2Disease is single-turn (symptom text → disease label) and doesn't encode *sequential* questioning logic at all — without DDXPlus, your requirement graph is entirely dependent on extracting structure from unstructured dialogue, which is more error-prone. This was flagged as the highest-risk item in v1 for a reason; DDXPlus directly de-risks it.
- **Store in:** NetworkX (unchanged from v1 — still the right call at this scale, still don't reach for Neo4j).

**B2. The phrasing (how to ask it, empathetically)**
- **Sources:** MedDialog *and* your own doctor-patient transcripts — this is exactly where your unlabeled data earns its keep. You don't need labels to use a transcript as a retrieval template; you just need the question turns.
- **Processing:** split your transcripts into turn pairs, keep only the doctor's question-asking turns, tag each with the symptom category it's eliciting info about (keyword-match against your canonical symptom vocabulary — same lightweight tagging approach as Arm A).
- **Index:** FAISS Index 3, same embedding model as Arm A.
- **Runtime use:** retrieve top-3 phrasing examples for the current missing slot, feed them to the local Llama as few-shot examples in the phrasing prompt (this is what makes a 1B/3B model produce natural-sounding output — it's pattern-completing from real examples, not composing from instructions alone, which small models are bad at).
- **Extra value from your own transcripts specifically:** since these are real doctor-patient conversations, they're also a good place to mine **naturalistic red-flag language** — how patients actually describe emergency symptoms in their own words, not textbook phrasing. Worth a manual pass over a sample of these transcripts specifically to enrich your red-flag keyword list (Part 3) with real phrasing variants, separate from the RAG indexing itself.

### Arm C — Diagnosis & department routing (hybrid, as decided)
- **Classifier makes the call:** XGBoost, trained on Symptom2Disease (primary) + DDXPlus if available (adds more diseases/better feature coverage).
- **Feature schema** (unchanged from v1, restated for completeness): multi-hot vector over canonical symptom vocabulary + severity (ordinal 1–5) + duration (bucketed) + age bucket + sex.
- **Disease → department mapping:** Symptom2Disease and DDXPlus both label *diseases*, not *departments* — you still need to build this mapping table by hand (e.g., "Migraine" → Neurology, "Eczema" → Dermatology). This is unavoidable manual work in either data path; budget an afternoon for it against your department taxonomy (propose below, confirm with your team).
- **RAG explains, doesn't decide:** once the classifier outputs `(department, confidence, contributing_symptoms)`, Arm A retrieves supporting facts, and the local Llama generates the patient-facing explanation from a tightly constrained prompt — see template below. The prompt must explicitly forbid the model from suggesting a different department than the classifier chose; its only job is to explain the one it's given.

**Explanation prompt template (concrete):**
```
System: You are explaining a triage recommendation to a patient. Use ONLY the
facts provided below. Do not diagnose. Do not suggest a different department
than the one given. If the facts are insufficient to explain clearly, say so
and recommend an in-person evaluation. Keep the response under 120 words.

Recommended department: {department}
Patient's reported symptoms: {symptom_list}
Supporting facts (from MedQuad):
1. {fact_1}
2. {fact_2}
3. {fact_3}

Write the explanation now.
```
Run this with `num_predict: 150`, low temperature (~0.3) for consistency, and the Ollama JSON/structured mode is not needed here since the output is free text, not a data structure to parse.

---

## Part 3 — Red-Flag Gate (unchanged from v1, still Node 0)

Still the first node, still fully deterministic, still must not depend on the LLM being fast, warm, or even running. Pattern/keyword match against a maintained list (chest pain radiating to arm/jaw, sudden "worst headache of my life," one-sided weakness/facial droop/slurred speech, severe breathing difficulty, uncontrolled bleeding, suicidal ideation, etc.), checked against raw text *and* extracted entities. Enrich this list using the phrasing-mining step from Arm B2 above. Get informal clinical sign-off from your mentor if at all possible — an ML team's best guess at this list is a real gap worth naming in your documentation, not hiding.

---

## Part 4 — Revised Architecture

```
[Patient text input]  (voice/multilingual: deferred, see Part 6)
        │
        ▼
┌────────────────────────────────┐
│ NODE 0: Red-Flag Gate           │  deterministic, no LLM, <10ms
└────────────────────────────────┘
   │ no                    │ yes → [Emergency Protocol: escalate + alert]
   ▼
┌────────────────────────────────┐
│ NODE 1: Entity Extraction       │  primary: regex + biomedical-ner-all
│                                  │  fallback only: Llama 3.2 (rare calls)
└────────────────────────────────┘
   ▼
┌────────────────────────────────┐
│ NODE 2: Normalization           │  lookup table → FAISS Index 1
│                                  │  fallback (cosine ≥ 0.82 threshold)
└────────────────────────────────┘
   ▼
┌────────────────────────────────┐
│ NODE 3: Completeness Check      │  deterministic, queries NetworkX
│                                  │  requirement graph (Arm B1)
└────────────────────────────────┘
   │ incomplete                        │ complete / turn-cap (4) hit
   ▼                                    ▼
┌────────────────────────────────┐  ┌────────────────────────────────┐
│ NODE 4: Next-Best-Question      │  │ NODE 5: XGBoost Classification  │
│  KG → required slot             │  │  → department + confidence      │
│  FAISS Index 3 → phrasing ex.   │  └────────────────────────────────┘
│  Llama 3.2 → composes question  │             ▼
│  (Redis-cached by slot type)    │  ┌────────────────────────────────┐
└────────────────────────────────┘  │ NODE 6: Explanation Generation  │
   │                                 │  FAISS Index 1 → facts (Arm A)  │
   ▼                                 │  Llama 3.2 → patient explanation│
[back to user, await reply]         │  (constrained prompt, Part 2C)  │
                                     └────────────────────────────────┘
                                              ▼
                                     [Write final state → Supabase →
                                      hand off to Scheduling Service]
```

Redis sits in front of Supabase for hot-path reads/writes and, separately, caches Node 4's phrasing outputs by `(symptom_category, missing_slot)` key as described in Part 1.

---

## Part 5 — Department Taxonomy (proposed, please confirm)

Needed for both the classifier's label space and Arm A's `specialty_tag`. Proposing a standard outpatient set — trim or extend based on what your problem statement/eval rubric actually expects:

`General Medicine, Cardiology, Dermatology, ENT, Gastroenterology, Neurology, Orthopedics, Pediatrics, Psychiatry, Pulmonology, Gynecology, Ophthalmology, Urology, Endocrinology`

---

## Part 6 — Multilingual (deferred, extension point only)

Not building this now, but designing so it doesn't require a rework later: the correct insertion point is a translation step **before Node 1**, converting non-English input to English before entity extraction, and translating Node 4/6 outputs back before sending to the patient. When you get to this, note that a 1B/3B local Llama is a weak choice for translation quality on Indian languages specifically — this will likely be the point where you want a separate, purpose-built translation model or API rather than overloading your local Llama further. Not a decision to make now, just don't build anything into Node 1–6 that assumes English-only text structurally (e.g., don't hardcode English regex patterns into the red-flag gate without a plan to run them post-translation).

---

## Part 7 — Roadmap

**Phase 1 — Data (start here, still the highest-risk item)**
1. Decide: pursue DDXPlus or not (recommended: yes, see Arm B1). If yes, download and process first — it gates the graph-quality decision below.
2. Build canonical symptom vocabulary + lookup table.
3. Process Symptom2Disease into classifier training format; build disease→department mapping table (Part 5 taxonomy).
4. Build the requirement graph (Path A with DDXPlus, or Path B via MedDialog+transcripts extraction+validation).
5. Process your own transcripts: strip to question-turns, tag by symptom category, feed into Index 3 alongside MedDialog. Separately, hand-review a sample for red-flag phrasing enrichment.
6. Build FAISS Index 1 (MedQuad) with specialty tags, FAISS Index 3 (MedDialog + your transcripts).

**Phase 2 — Deterministic Core**
7. Red-flag gate + list (Part 3).
8. Nodes 1–3 (extraction, normalization, completeness) — no LLM dependency at this stage, get this working and tested first since it's the part most likely to be correct on the first try.

**Phase 3 — Classifier**
9. Train/validate XGBoost; report per-department accuracy and confusion matrix.

**Phase 4 — LLM Integration (last, deliberately)**
10. Wire Ollama + Llama 3.2, structured output mode, Node 4 phrasing (with Redis cache) and Node 6 explanation (constrained prompt).
11. Load-test on your actual hardware early — confirm real per-call latency before you're relying on it during a demo.

**Phase 5 — Integration & Eval**
12. Wire to scheduling/doctor portal (unchanged from v1, no ML involved).
13. Build a held-out synthetic conversation test set; measure specialty accuracy, red-flag recall (target: as close to 100% as achievable — this is the metric that matters most), average turns-to-completion, and Node 4/6 latency under your actual local hardware.

---

## Part 8 — Assumptions to Verify (flagging rather than blocking on)

- Assuming Symptom2Disease's schema is `(symptom_text, disease_label)` single-turn pairs, not sequential — worth a quick check once you have the file in hand.
- Assuming your own transcripts have no embedded department/outcome label anywhere in the text (e.g., a header or closing note) — if they do, that's free weak-label data for the classifier and changes Phase 1 step 5 meaningfully. Worth a quick look before you commit to treating them as label-free.
- Department taxonomy (Part 5) is a starting proposal, not final — confirm against whatever your competition brief or mentor expects before it's baked into the classifier's label space, since changing it later means retraining.
