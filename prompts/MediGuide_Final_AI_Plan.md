# TriagePlus — AI Subsystem: Complete Technical Plan

---

## Table of Contents

1. [Locked Design Decisions](#1-locked-design-decisions)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Full Event Flow: Patient Entry to Appointment Confirmed](#3-full-event-flow-patient-entry-to-appointment-confirmed)
4. [Per-Turn Pipeline: What Happens on Every Message](#4-per-turn-pipeline-what-happens-on-every-message)
5. [Component 1: Emergency Detector](#5-component-1-emergency-detector)
6. [Component 2: Lay-Term Normalizer](#6-component-2-lay-term-normalizer)
7. [Component 3: Specialty Classifier](#7-component-3-specialty-classifier)
8. [Component 4: RAG System](#8-component-4-rag-system)
9. [Component 5: Slot Extractor and Conversation Engine (Gemini)](#9-component-5-slot-extractor-and-conversation-engine-gemini)
10. [Component 6: Triage Level Mapper](#10-component-6-triage-level-mapper)
11. [Component 7: Prognosis Helper](#11-component-7-prognosis-helper)
12. [Component 8: Doctor Brief Generator](#12-component-8-doctor-brief-generator)
13. [Component 9: Doctor Recommendation Ranker](#13-component-9-doctor-recommendation-ranker)
14. [Component 10: Multilingual Voice Input](#14-component-10-multilingual-voice-input)
15. [Classifier Training: Data Pipeline](#15-classifier-training-data-pipeline)
16. [Classifier Training: End-to-End Procedure](#16-classifier-training-end-to-end-procedure)
17. [RAG Index Construction](#17-rag-index-construction)
18. [Evaluation Criteria](#18-evaluation-criteria)
19. [Implementation Checklist](#19-implementation-checklist)
20. [Known Limitations to Document](#20-known-limitations-to-document)

---

## 1. Locked Design Decisions

These decisions are fixed before any code is written. Everything downstream depends on them.

### 1.1 Specialty Taxonomy — 9 Classes

```
Cardiology
Dermatology
Orthopedics
Gastroenterology
Neurology
Pediatrics
Psychiatry
General Medicine
Respiratory
```

Respiratory is kept as a distinct 9th class rather than folded into General Medicine. The corpus has 213 Respiratory transcripts — merging them into General Medicine would make General Medicine the dominant class and destroy the classifier's ability to learn what genuine General Medicine looks like. Respiratory complaints also benefit from specialist routing (pulmonologist), not a GP.

### 1.2 Classifier Input Format (Fixed Contract)

At every turn, the classifier receives exactly this string:

```
SYMPTOM: <normalized chief complaint, or raw patient message if not yet extracted>
ONSET: <value if known, else "unknown">
SEVERITY: <value if known, else "unknown">
CONTEXT: <up to 400 characters of the patient's latest message verbatim>
```

This format is identical at turn 1 (when only the patient's first message is available) and at the final pass (when all slots are filled). Training samples must be in this exact format. The same function builds this string during training data construction and during live inference — never two separate implementations.

```python
def build_classifier_input(
    chief_complaint: str,
    onset: str | None,
    severity: str | None,
    latest_patient_message: str,
    max_context_chars: int = 400
) -> str:
    return (
        f"SYMPTOM: {chief_complaint}\n"
        f"ONSET: {onset or 'unknown'}\n"
        f"SEVERITY: {severity or 'unknown'}\n"
        f"CONTEXT: {latest_patient_message[:max_context_chars]}"
    )
```

### 1.3 Triage Levels

```
Level 1 — Emergency   → route to emergency services; do not book appointment
Level 2 — Urgent      → same-day or next-day slot; priority queue
Level 3 — Soon        → within 3 days
Level 4 — Routine     → any available slot within 7 days
```

Triage level is computed by a deterministic rule-based mapper, not a separate ML model.

### 1.4 LLM

**Gemini 1.5 Flash** (drop-in alternative: GPT-4o-mini).

Rationale: a session runs 8–15 turns and accumulates 3,000–5,000 tokens of conversation history plus ~600 tokens of RAG chunks per turn. The LLM needs to be (a) cheap per token, (b) fast enough for conversational latency (<2s), (c) reliable at following structured JSON extraction prompts, and (d) capable of handling a full session in context. Gemini 1.5 Flash and GPT-4o-mini both satisfy all four requirements. Heavier models add cost and latency without improving the bottleneck, which is instruction-following quality on a well-defined intake task, not raw reasoning.

The same LLM is used for slot extraction, conversation turn generation, prognosis helper generation, and doctor brief generation — with different system prompts for each role.

### 1.5 Embedding Model

**`sentence-transformers/all-MiniLM-L6-v2`** — 384-dimensional, ~80MB, CPU-friendly, no API call.

This model is loaded **once** at server startup and shared across all uses: classifier input embedding, FAISS query embedding for RAG retrieval, and RAG index construction. Never reload per request.

### 1.6 Classifier Head

**`sklearn.linear_model.LogisticRegression`** with frozen MiniLM-L6-v2 embeddings.

Rationale: this is a 9-class text routing problem with a few hundred to a few thousand training samples per class. Fine-tuning a transformer in this regime risks overfitting, especially for thin classes that will rely partly on synthetic data. Logistic Regression over frozen embeddings is fast (< 1ms inference), deterministic, produces calibrated probabilities (needed for the confidence display), and trivial to retrain and audit.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE / STARTUP                                  │
│                                                                             │
│  Training data pipeline → LR Classifier (pickled)                          │
│  RAG corpus chunking + embedding → FAISS Index A (conversations)           │
│  Condition corpus embedding → FAISS Index B (prognosis knowledge)          │
│  Embedding model: MiniLM-L6-v2 (loaded once, shared)                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          RUNTIME (per session)                              │
│                                                                             │
│  Voice/Text Input                                                           │
│       │                                                                     │
│  [ASR + Translation if non-English]                                         │
│       │                                                                     │
│  Emergency Detector (regex, every message)                                  │
│       │                                                                     │
│  Lay-Term Normalizer (dictionary lookup)                                    │
│       │                                                                     │
│  build_classifier_input() ───────────────────────────────┐                 │
│       │                                                   │                 │
│  MiniLM embed (once per turn) ──────────── shared ────────┤                │
│       │                                                   │                 │
│  LR Classifier           FAISS Index A (specialty filter) │                │
│  → specialty             → top-3 Q&A chunks              │                │
│  → confidence                    │                        │                │
│       │                          │                        │                │
│       └──────────────────────────┘                        │                │
│                          │                                │                │
│                    Gemini prompt (RAG chunks + session state + history)     │
│                          │                                                  │
│                    Gemini → next intake question OR slot extraction JSON    │
│                          │                                                  │
│                    SessionState updated                                     │
│                          │                                                  │
│                    All slots filled? → Final classifier pass               │
│                          │                                                  │
│                    Triage Level Mapper                                      │
│                          │                                                  │
│                    Prognosis Helper (FAISS Index B + Gemini)               │
│                          │                                                  │
│                    Result screen → Scheduling → Payment → Doctor Brief     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Two FAISS indexes:**

- **Index A — Conversation index.** Contains Q&A exchange chunks from real doctor-patient transcripts. Used during intake to retrieve exemplar conversations that guide Gemini's next question. Queried on every patient turn.
- **Index B — Knowledge index.** Contains condition descriptions from MedQuAD and MedlinePlus. Used only once, at the prognosis helper step, to ground the general condition notes. Never queried during intake.

---

## 3. Full Event Flow: Patient Entry to Appointment Confirmed

```
┌─────────────────────────────────────────────────────────────────┐
│                      PATIENT ENTRY POINT                        │
│   Text (web / mobile)    │    Voice (local language)            │
└──────────────┬───────────┴──────────────┬───────────────────────┘
               │                          │
               │                   ASR: Whisper / Google STT
               │                   Language detect
               │                   Translation: IndicTrans2
               │                          │
               └──────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │     EMERGENCY DETECTOR       │  ← every message, first
                 │     (regex, § 5)             │
                 └──────────────┬───────────────┘
                                │  no match
                                ▼
                 ┌──────────────────────────────┐
                 │     LAY-TERM NORMALIZER      │  ← dictionary lookup (§ 6)
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────────┐
                 │   build_classifier_input()               │
                 │   SYMPTOM / ONSET / SEVERITY / CONTEXT   │
                 └──────────────┬───────────────────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
          ┌──────────────────┐   ┌──────────────────────────┐
          │  MiniLM embed    │   │  (same vector reused)    │
          │  → 384-dim       │   │  for FAISS A query       │
          └────────┬─────────┘   └────────────┬─────────────┘
                   │                           │
                   ▼                           ▼
          ┌──────────────────┐   ┌──────────────────────────┐
          │  LR CLASSIFIER   │   │  FAISS INDEX A           │
          │  → specialty     │   │  specialty-filtered      │
          │  → confidence    │   │  → top-3 Q&A chunks      │
          └────────┬─────────┘   └────────────┬─────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │   GEMINI PROMPT (§ 9)                    │
                 │   System role + retrieved Q&A examples   │
                 │   + session state + conversation history  │
                 │   + task: extract slots + ask next Q     │
                 └──────────────┬───────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────────┐
                 │   Gemini response:                       │
                 │   - Slot extraction JSON                 │
                 │   - Next intake question (patient-facing)│
                 └──────────────┬───────────────────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────────┐
                 │   SessionState updated                   │
                 │   All 3 required slots filled?           │
                 └──────────────┬───────────────────────────┘
                                │
              NO: loop ◄────────┤────────► YES
              (next patient     │
               message)         ▼
                    ┌───────────────────────────────────────┐
                    │   FINAL CLASSIFIER PASS               │
                    │   on fully-filled slot string         │
                    │   → authoritative specialty           │
                    │   → confidence label                  │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   TRIAGE LEVEL MAPPER (§ 10)          │
                    │   severity + keyword rules            │
                    │   → Level 1 / 2 / 3 / 4              │
                    └──────────────┬────────────────────────┘
                                   │
                    Level 1 ───────┤──────── Level 2–4
                    (emergency     │         (continue)
                     path)         ▼
                    ┌───────────────────────────────────────┐
                    │   PROGNOSIS HELPER (§ 11)             │
                    │   FAISS Index B query                 │
                    │   → top-3 condition knowledge chunks  │
                    │   Gemini call → 3 general notes       │
                    │   Output filter (no "you have X")     │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   PATIENT RESULT SCREEN               │
                    │   • Predicted specialty               │
                    │   • Confidence (High/Medium/Low)      │
                    │   • Triage level                      │
                    │   • 3 general notes + disclaimer      │
                    │   • Doctor recommendations            │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   SCHEDULING ENGINE                   │
                    │   Filter by specialty                 │
                    │   Rank by triage + rating + slots     │
                    │   Patient selects slot                │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   PAYMENT GATEWAY (mock)              │
                    └──────────────┬────────────────────────┘
                                   │  confirmed
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   DOCTOR BRIEF GENERATOR (§ 12)       │
                    │   Gemini call, clinical format        │
                    │   → Appointment.ai_brief              │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────────┐
                    │   APPOINTMENT CONFIRMED               │
                    │   Patient: confirmation + wait time   │
                    │   Doctor portal: brief in queue       │
                    └───────────────────────────────────────┘
```

---

## 4. Per-Turn Pipeline: What Happens on Every Message

Every patient message passes through the same sequence. This diagram shows one complete turn in detail using a concrete example.

```
Patient sends: "My chest feels really tight, especially when I walk fast"
                              │
                              ▼
        ┌─────────────────────────────────┐
        │  Emergency check               │  → no match, continue
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  Lay-term normalization        │
        │  "chest feels tight"           │
        │  → "chest tightness"           │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  build_classifier_input()      │
        │  SYMPTOM: chest tightness      │
        │  ONSET: unknown                │
        │  SEVERITY: unknown             │
        │  CONTEXT: My chest feels...    │
        └──────────────┬──────────────────┘
                       │
             ┌─────────┴──────────┐
             │  (one embed call)  │
             ▼                    ▼
  ┌──────────────────┐  ┌──────────────────────────┐
  │  LR Classifier   │  │  FAISS Index A           │
  │  → Cardiology    │  │  filter: Cardiology       │
  │  → p=0.72, High  │  │  → 3 Q&A chunks          │
  └────────┬─────────┘  └────────────┬─────────────┘
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  Gemini prompt assembled:                 │
        │                                           │
        │  SYSTEM: You are a medical intake         │
        │  assistant. Goal: collect chief           │
        │  symptom, severity, onset.                │
        │                                           │
        │  RETRIEVED EXAMPLES [Cardiology]:         │
        │  D: Does the pain spread to your arm?     │
        │  P: No, it stays in my chest.             │
        │  ---                                      │
        │  D: How long does each episode last?      │
        │  P: Maybe five to ten minutes.            │
        │  ---                                      │
        │  D: Is it sharp or more like pressure?    │
        │  P: More like pressure, like a weight.    │
        │                                           │
        │  SESSION STATE:                           │
        │  chief_complaint: chest tightness [filled]│
        │  severity: unknown [missing → ask this]   │
        │  onset: unknown [missing]                 │
        │                                           │
        │  CONVERSATION HISTORY: [this session]     │
        │                                           │
        │  TASK: Ask for SEVERITY next.             │
        │  Use the examples as style guidance,      │
        │  not verbatim scripts.                    │
        └──────────────┬────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────────────────┐
        │  Gemini outputs TWO things:               │
        │                                           │
        │  1. Slot extraction JSON (internal):      │
        │  {                                        │
        │    "chief_complaint": "chest tightness",  │
        │    "severity": null,                      │
        │    "onset": null,                         │
        │    "all_required_slots_filled": false     │
        │  }                                        │
        │                                           │
        │  2. Patient-facing message:               │
        │  "Thanks for letting me know. On a scale  │
        │   of 1 to 10, how would you rate the      │
        │   tightness — and does it ever come with  │
        │   any sweating or shortness of breath?"   │
        └──────────────┬────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────────────────┐
        │  SessionState updated:                    │
        │  provisional_specialty: Cardiology        │
        │  confidence: High                         │
        │  chief_complaint: chest tightness [✓]    │
        │  severity: [pending]                      │
        │  onset: [pending]                         │
        └───────────────────────────────────────────┘
```

**Key implementation note:** the embedding is computed once per turn and fed to both the LR classifier and the FAISS query. Do not embed twice. The shared `SentenceTransformer` instance handles both uses.

---

## 5. Component 1: Emergency Detector

**Type:** Rule-based regex. No ML. No API call. Deterministic.

Runs on every patient message before any other component. If it matches, the pipeline stops and returns a fixed emergency template immediately.

```python
import re

EMERGENCY_PATTERNS = [
    # Cardiac
    r"\bchest\s+pain\b",
    r"\bchest\s+tightness\b",
    r"\bradiat(e|ing|ed)\s+(to|down)\s+(arm|jaw|neck|shoulder)\b",
    r"\bheart\s+attack\b",
    r"\bmyocardial\b",

    # Neurological
    r"\bstroke\b",
    r"\bsudden\s+(numbness|weakness|confusion|vision\s+loss)\b",
    r"\bface\s+(drooping|droop)\b",
    r"\bspeech\s+(slurred|loss)\b",
    r"\bseizure\b",
    r"\bunconscious\b",
    r"\bfainted\b",

    # Respiratory
    r"\bcan'?t\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bsevere\s+shortness\s+of\s+breath\b",
    r"\bblue\s+(lips|fingers|face)\b",
    r"\bcyanosis\b",

    # Trauma / bleeding
    r"\bsevere\s+bleeding\b",
    r"\buncontrolled\s+bleed\b",
    r"\bknife\b.*\bwound\b",
    r"\bgunshot\b",

    # Self-harm / overdose
    r"\bsuicid(e|al)\b",
    r"\bself.?harm\b",
    r"\boverdos(e|ed|ing)\b",
    r"\bswallowed?\s+(poison|bleach|pill)\b",
    r"\bno\s+pulse\b",
    r"\bnot\s+breathing\b",
]

def check_emergency(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(p, normalized) for p in EMERGENCY_PATTERNS)
```

**On match:** return a hardcoded emergency response with relevant helpline numbers. Log the matched pattern and session ID for audit. Do not pass the message to any downstream component.

**Important:** this list is not exhaustive. A regex system will miss novel phrasings. It must be reviewed by a medical professional before deployment.

---

## 6. Component 2: Lay-Term Normalizer

**Type:** Dictionary lookup. No ML. No API call.

Translates patient lay language into normalized clinical terms before embedding. The original patient text is preserved for display — patients never see clinical jargon reflected back at them.

**Implementation:** a CSV file with ~200–300 entries, loaded at startup into a dict. Apply longest-match, case-insensitive substring replacement.

```
lay_term,clinical_term
stomach ache,abdominal pain
tummy,abdomen
racing heart,palpitations
runny nose,rhinorrhea
throwing up,vomiting
dizzy,dizziness
pins and needles,paresthesia
fits,seizures
pee problem,urinary symptoms
shortness of breath,dyspnea
tired all the time,fatigue
swollen ankles,peripheral edema
yellow skin,jaundice
blurred vision,visual disturbance
ringing in ears,tinnitus
memory problems,cognitive impairment
shaking hands,tremor
chest tightness,chest pressure
```

Build coverage for all 9 specialties. The normalizer does not need to be exhaustive — the embedding model handles semantic similarity for terms not in the dictionary. Its purpose is to close the gap between extremely common lay phrasings ("my tummy hurts") and their clinical equivalents ("abdominal pain") where the embedding distance would otherwise be misleading.

---

## 7. Component 3: Specialty Classifier

### 7.1 Role

The classifier has exactly two jobs:

1. **Drive RAG retrieval on every turn.** Its predicted specialty is the filter key for FAISS Index A. If it says Cardiology, retrieval returns Cardiology intake examples. This is its most important function — every turn, from turn 1.
2. **Produce the final routing decision.** At the end of intake, its final pass on the fully-filled slot string determines which specialty the patient gets booked into.

The classifier does not generate questions, extract slots, compute triage, or substitute for Gemini in any scenario. It is a fast, local routing signal.

### 7.2 Architecture

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Loaded once at startup — never reloaded per request
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42
)
```

### 7.3 Inference

```python
def classify_specialty(session_state: SessionState, latest_message: str) -> dict:
    # Build fixed-format input string
    input_str = build_classifier_input(
        chief_complaint=session_state.chief_complaint or latest_message,
        onset=session_state.onset,
        severity=session_state.severity,
        latest_patient_message=latest_message
    )

    # Embed — same model instance used for FAISS query
    embedding = embedding_model.encode([input_str])  # shape: (1, 384)

    # Classify
    proba = clf.predict_proba(embedding)[0]           # shape: (9,)
    top_idx = np.argmax(proba)
    top_class = le.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]

    # Confidence bucketing
    if confidence >= 0.70:
        confidence_label = "High"
    elif confidence >= 0.45:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Top-2 for low-confidence fallback
    top2_idx = np.argsort(proba)[-2:][::-1]
    top2 = le.inverse_transform(top2_idx).tolist()

    return {
        "specialty": top_class,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "top2": top2,
        "all_probas": dict(zip(le.classes_, proba.tolist()))
    }
```

### 7.4 Confidence behavior

| Confidence | Label | Behavior |
|---|---|---|
| ≥ 0.70 | High | Use top-1 specialty for FAISS filter; retrieve 3 chunks from top-1 |
| 0.45–0.69 | Medium | Use top-1 specialty for FAISS filter; retrieve 3 chunks from top-1 |
| < 0.45 | Low | Use top-2 specialties; retrieve 2 chunks from each (4 total); add General Medicine as secondary recommendation on result screen |

**Specialty update mid-session:** after each Gemini slot extraction, re-run the classifier on the newly extracted chief_complaint. If the new top class has probability > 0.60 and differs from the current provisional specialty, update the RAG filter silently for the next turn. Do not alert the patient.

**When to show the specialty to the patient:** only on the final result screen, after all slots are filled and the final classifier pass is complete. Never show provisional predictions during intake.

### 7.5 Why not fine-tune the embedding model?

Fine-tuning changes the embedding space. If you fine-tune MiniLM on medical text, the FAISS index built from frozen MiniLM embeddings is no longer compatible — you would need to re-embed the entire corpus. For a hackathon, keeping MiniLM frozen eliminates this dependency and means one offline embedding run covers both classifier training and FAISS index construction.

---

## 8. Component 4: RAG System

### 8.1 What RAG is doing

RAG helps Gemini conduct a better intake conversation. It gives Gemini two things it otherwise lacks:

- **Examples of how doctors ask follow-up questions** for a given symptom in a given specialty, so Gemini's questions sound like a real clinical intake rather than a generic chatbot.
- **Awareness of what associated symptoms are clinically relevant** for the chief complaint, so Gemini probes the right things ("does the pain radiate to your jaw?") rather than cycling through generic questions.

Without RAG, Gemini asks the same three questions in every session. With RAG and real doctor transcript examples in context, it asks specialty-appropriate follow-ups informed by how real clinicians handle similar presentations.

### 8.2 Two separate indexes

**Index A — Conversation index** (used during intake, every turn)

Source: Q&A exchange pairs from doctor-patient transcripts. Each chunk is one `D: ... / P: ...` exchange. Retrieved chunks go directly into Gemini's prompt as few-shot conversation exemplars.

**Index B — Knowledge index** (used once, at prognosis helper step only)

Source: condition descriptions from MedQuAD and MedlinePlus. Retrieved chunks provide condition context that grounds the prognosis helper generation.

These two indexes must never be queried together. Their content types are incoherent if mixed: condition descriptions are not conversation examples, and conversation examples are not condition knowledge.

### 8.3 Index A: Conversation index

**Sources:**

| Source | Content | How to use |
|---|---|---|
| `conversations.zip` (272 transcripts) | Real doctor-patient intake Q&A in D: / P: format | Primary source. Extract every D/P exchange pair as a chunk |
| MedDialog (English) — Hugging Face `medical_dialog` | ~250k real doctor-patient dialogues from HealthCareMagic and iCliniq, tagged by department | Extract Q&A turn pairs. Map department labels to your 9 specialties. Fills thin specialties |

Do not add mtsamples, Symptom2Disease, MedQuAD, CounselChat, or MedlinePlus to Index A. None of them are Q&A conversation exchanges.

**Chunking:**

```
Each chunk = one D: ... / P: ... exchange pair

Example:
  text: "D: Does the pain spread to your left arm or jaw? P: No, it stays in my chest but sometimes goes up to my neck."
  metadata:
    specialty: "Cardiology"
    source_file: "cardiology/transcript_003.txt"
    chunk_id: "cardiology_003_turn_05"
    turn_index: 5
```

Each chunk is ~40–80 tokens, small enough that 3 chunks fit in a Gemini prompt alongside full session history without crowding out the conversation.

**Expected chunk counts after adding MedDialog:**

| Specialty | Approx. chunks |
|---|---|
| Respiratory | ~10,200 |
| Orthopedics | ~2,200 |
| Neurology | ~2,000 |
| Psychiatry | ~2,500 |
| Cardiology | ~1,500 |
| Pediatrics | ~1,200 |
| General Medicine | ~1,000 |
| Dermatology | ~800 |
| Gastroenterology | ~500 |

Total: ~21,900 chunks. Use `IndexIVFFlat` with `nlist=256`.

**Retrieval:**

```python
def retrieve_conversation_examples(
    query_embedding: np.ndarray,
    predicted_specialty: str,
    confidence_label: str,
    top_k: int = 3,
    min_score: float = 0.28
) -> list[dict]:
    """
    1. Search FAISS for top-20 candidates from the full index.
    2. Filter to predicted_specialty metadata.
    3. Apply cosine score threshold (0.28).
    4. If confidence_label == "Low": use top-2 specialties, retrieve 2 from each.
    5. If fewer than 2 chunks pass after filtering: relax specialty filter, retry.
    6. Return top_k chunks.
    """
```

### 8.4 Index B: Knowledge index

**Sources:**

| Source | Content | How to use |
|---|---|---|
| **MedQuAD** (NIH / Hugging Face `keivalya/MedQuad-MedicalQnADataset`) | ~47,000 QA pairs from NIH websites covering symptoms, causes, treatments by condition | Chunk at QA pair level. Tag with specialty inferred from NIH source category |
| **MedlinePlus XML dump** (freely available) | ~1,000 plain-English condition summaries written for patients | Chunk at section level: symptoms section, causes section, and "when to see a doctor" section as separate chunks. Tag with specialty |

**Do not** add PubMed abstracts to Index B for patient-facing use. The language register is clinical and dense; it is not appropriate for generating patient-facing general notes.

**Retrieval:** queried once, at the prognosis helper step, after the final specialty is determined. Query is the fully-filled slot string, filtered by predicted specialty.

### 8.5 How retrieved chunks appear in the Gemini prompt (Index A)

```
SYSTEM PROMPT:
You are a medical intake assistant conducting a structured symptom interview.
Your goal is to collect three pieces of information from the patient:
  1. Chief symptom (what is bothering them most)
  2. Severity (how bad it is, 1–10 or in their own words)
  3. Onset (how long they have had this symptom)

Ask ONE question per turn. Be conversational, warm, and empathetic.
Never use clinical jargon. Never suggest a diagnosis.
Never ask for information the patient has already provided.

The following are real examples of how similar patient cases were handled by doctors.
Use them as guidance for the style and depth of follow-up questions.
Do not copy them verbatim.

RETRIEVED EXAMPLES [Cardiology]:
---
D: Does the discomfort get worse when you exert yourself, like climbing stairs?
P: Yes, especially going uphill. I have to stop and rest.
---
D: How long does each episode of chest pain last?
P: Maybe five to ten minutes, then it fades.
---
D: Is it a sharp stabbing pain or more like pressure?
P: More like pressure, like something heavy sitting on my chest.
---

CURRENT SESSION STATE:
- Chief symptom: chest tightness [collected]
- Severity: unknown → ask this next
- Onset: unknown

CONVERSATION HISTORY:
[full conversation so far]

YOUR TASK:
The next missing required slot is: SEVERITY.
Ask a natural follow-up question that gathers severity.
The retrieved examples show that Cardiology intakes often ask about exertion,
episode duration, and character of pain — factor this in.

Respond with a JSON object:
{
  "patient_message": "<your conversational question to the patient>",
  "extracted": {
    "chief_complaint": "<if newly found, else null>",
    "severity": "<if newly found, else null>",
    "onset": "<if newly found, else null>",
    "associated_symptoms": ["<any mentioned>"],
    "medical_history_flags": ["<any mentioned>"]
  },
  "slots_filled": {
    "chief_complaint": true,
    "severity": false,
    "onset": false
  },
  "all_required_slots_filled": false
}
```

---

## 9. Component 5: Slot Extractor and Conversation Engine (Gemini)

### 9.1 Role

Gemini serves as the conversational backbone during intake. On each turn it does two things simultaneously: extracts any slot information present in the patient's message, and generates the next patient-facing question. Both happen in one API call.

### 9.2 Slots tracked

| Slot | Required | Description | Example |
|---|---|---|---|
| `chief_complaint` | Yes | Primary symptom, normalized | "chest tightness" |
| `severity` | Yes | Patient-reported intensity | "7", "severe", "comes and goes" |
| `onset` | Yes | How long ago it started | "3 days", "since last week" |
| `associated_symptoms` | No | Other symptoms mentioned | ["sweating", "shortness of breath"] |
| `medical_history_flags` | No | Any PMH volunteered | ["diabetes", "hypertension"] |

### 9.3 Session state

Maintain a `SessionState` object server-side, keyed by session ID. Updated after every Gemini response. This is the single source of truth for what has been collected.

```python
@dataclass
class SessionState:
    session_id: str
    chief_complaint: str | None = None
    severity: str | None = None
    severity_numeric: int | None = None
    onset: str | None = None
    onset_days: int | None = None
    associated_symptoms: list[str] = field(default_factory=list)
    medical_history_flags: list[str] = field(default_factory=list)
    provisional_specialty: str | None = None
    provisional_confidence: float | None = None
    turn_count: int = 0
    slots_filled: dict = field(default_factory=lambda: {
        "chief_complaint": False,
        "severity": False,
        "onset": False
    })

    @property
    def all_required_filled(self) -> bool:
        return all(self.slots_filled.values())
```

### 9.4 Rules for Gemini during intake

- Ask ONE question per turn.
- Ask for slots in order: chief_complaint → severity → onset. Never jump ahead.
- Never ask for information already provided.
- Never suggest a diagnosis or use a disease name.
- If the patient volunteers slot information without being asked (e.g., "I've had chest pain for three days"), extract it and ask about the next missing slot.
- If onset is vague ("recently", "for a while"), accept it as filled.
- If severity is given qualitatively ("quite bad"), convert to numeric estimate internally but store both.

---

## 10. Component 6: Triage Level Mapper

**Type:** Rule-based. No ML. Takes `SessionState` as input after all slots are filled.

```python
LEVEL1_KEYWORDS = [
    "chest pain", "cannot breathe", "stroke", "seizure",
    "unconscious", "not breathing", "no pulse"
]
LEVEL2_KEYWORDS = [
    "high fever", "severe pain", "blood in stool", "blood in urine",
    "sudden vision loss", "sudden hearing loss", "severe headache",
    "shortness of breath"
]

def compute_triage_level(state: SessionState) -> int:
    """Returns 1 (Emergency), 2 (Urgent), 3 (Soon), 4 (Routine)."""

    # Level 1
    if state.severity_numeric and state.severity_numeric >= 9:
        return 1
    if state.chief_complaint and any(
        kw in state.chief_complaint for kw in LEVEL1_KEYWORDS
    ):
        return 1

    # Level 2
    if state.severity_numeric and state.severity_numeric >= 7:
        return 2
    if state.onset_days is not None and state.onset_days <= 1:
        if state.severity_numeric and state.severity_numeric >= 5:
            return 2
    if state.chief_complaint and any(
        kw in state.chief_complaint for kw in LEVEL2_KEYWORDS
    ):
        return 2

    # Level 3
    if state.severity_numeric and state.severity_numeric >= 5:
        return 3
    if state.onset_days is not None and state.onset_days <= 7:
        return 3

    # Level 4
    return 4
```

**Triage level feeds the scheduling engine directly:** Level 2 patients are filtered to doctors with same-day or next-day availability. Level 1 triggers an emergency response and no appointment is booked.

---

## 11. Component 7: Prognosis Helper

### 11.1 What it is

A single Gemini call that generates up to 3 general condition notes contextually relevant to the patient's symptom profile. It is not a diagnosis. It is not presented as one. It is grounded in MedQuAD / MedlinePlus condition descriptions retrieved from Index B.

### 11.2 When it runs

Once, after the final classifier pass and triage level computation, before the result screen is shown. Its output is used in two places: the patient-facing result screen and the doctor brief. The same generated text is stored and reused — not regenerated.

### 11.3 Prompt

```
SYSTEM:
You are a general health information assistant. Your role is to provide background
context only. You must never state or imply a diagnosis.

Strict rules:
- Never say "you have", "you are diagnosed with", "this is likely [condition]",
  or any phrasing that asserts a specific condition for this patient.
- Produce only a numbered list. No prose outside the numbered items.
- Maximum 3 items.
- Each item: one sentence naming a condition commonly associated with similar
  presentations, followed by one sentence of general description.
- Do not include any statement about whether this patient has this condition.

USER:
Patient profile:
- Chief symptom: {chief_complaint}
- Severity: {severity}
- Onset: {onset}
- Predicted specialty: {predicted_specialty}

Retrieved condition context:
---
{knowledge_chunk_1}
---
{knowledge_chunk_2}
---
{knowledge_chunk_3}
---

Using only the context above, list up to 3 conditions commonly discussed in
similar presentations. Do not use general knowledge beyond what is provided here.
```

### 11.4 Output filter

Before this output reaches either the patient screen or the doctor brief:

```python
DIAGNOSTIC_ASSERTION_PATTERNS = [
    r"\byou\s+have\b",
    r"\byou'?re\s+(diagnosed|suffering)\b",
    r"\bthis\s+is\s+(likely|probably|certainly)\b",
    r"\byou\s+(likely|probably|definitely)\b",
    r"\byour\s+(condition|diagnosis)\s+is\b",
    r"\bI\s+(believe|think|suspect)\s+you\b",
]

def filter_prognosis_output(text: str) -> str:
    for pattern in DIAGNOSTIC_ASSERTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return rewrite_prognosis(text)  # one Gemini rewrite attempt
    return text
```

If rewrite fails or triggers the filter again, fall back to this fixed safe response:

```
Based on similar presentations, conditions related to the {predicted_specialty}
system are worth discussing with your doctor. Please consult a qualified medical
professional for a proper assessment.
```

### 11.5 Fixed disclaimer (hardcoded in frontend)

```
⚠ This information is general in nature and does not constitute a medical
diagnosis. Always consult a qualified medical professional for advice
specific to your situation.
```

This text is never generated by the model. It is hardcoded in the frontend and cannot be modified or omitted by any model output.

---

## 12. Component 8: Doctor Brief Generator

### 12.1 Role

A separate Gemini call triggered after payment confirmation. It produces a structured clinical brief for the doctor, stored as `Appointment.ai_brief`. Not visible to the patient.

The doctor brief is not a reuse of the prognosis helper's patient-facing output. Doctors and patients are different audiences: a doctor needs structured clinical-style intake, not conversational general notes.

### 12.2 Prompt

```
SYSTEM:
You are a medical intake assistant generating a pre-consultation brief for a doctor.
Produce a concise structured brief using clinical language.
Do not include any content the patient has not explicitly stated.
Do not infer conditions. Do not diagnose.

Output format (use these exact headings):
PRESENTING COMPLAINT: <one sentence>
SYMPTOM DETAIL: <chief symptom, severity, onset in clinical shorthand>
ASSOCIATED SYMPTOMS: <bullet list, or "None reported">
RELEVANT HISTORY: <any PMH the patient mentioned, or "Not provided">
TRIAGE LEVEL: <1–4 with label, e.g. "Level 3 — Soon">
PREDICTED SPECIALTY: <name> (Confidence: <High / Medium / Low>)
CONTEXT NOTES: <prognosis helper output, reformatted as clinical shorthand>
SYSTEM NOTE: The above notes are AI-generated and unverified.
             Clinical judgment supersedes this brief.

USER:
Session data:
{session_state_json}

Prognosis context (for background only):
{prognosis_helper_text}
```

### 12.3 Storage

Store in `Appointment.ai_brief` as a text field. Render in the doctor portal alongside scheduled patients. Never expose to the patient-facing interface.

---

## 13. Component 9: Doctor Recommendation Ranker

Deterministic scoring function. No ML required.

```python
def score_doctor(doctor: Doctor, triage_level: int) -> float:
    rating_norm = doctor.rating / 5.0
    days_to_next_slot = doctor.next_available_slot_days
    availability_score = 1.0 / (1.0 + days_to_next_slot)
    feedback_norm = doctor.feedback_score / 5.0

    return (
        0.5 * rating_norm +
        0.3 * availability_score +
        0.2 * feedback_norm
    )

def recommend_doctors(
    specialty: str,
    triage_level: int,
    all_doctors: list[Doctor],
    top_n: int = 5
) -> list[Doctor]:
    filtered = [
        d for d in all_doctors
        if d.specialty == specialty
        and len(d.available_slots) > 0
        and (triage_level > 2 or d.next_available_slot_days <= 1)
        # Level 2: only doctors with same-day/next-day slots
    ]
    return sorted(filtered, key=lambda d: score_doctor(d, triage_level), reverse=True)[:top_n]
```

---

## 14. Component 10: Multilingual Voice Input

### 14.1 ASR

**Recommended:** OpenAI Whisper (medium model). Handles Indian English, Hindi, and several regional languages with reasonable accuracy. Can be run locally or via the Whisper API.

**Alternative:** Google Cloud STT with BCP-47 language codes (`hi-IN`, `ta-IN`, `kn-IN`, `te-IN`, `mr-IN`). Better accuracy for specific regional languages; requires API calls.

### 14.2 Translation

If detected language is not English, translate before passing to the pipeline. Store both the original-language text and the English translation in `SessionState`.

**Recommended:** IndicTrans2 (open-source, IITM, supports all 22 scheduled Indian languages). Run locally.

**Fallback:** Google Translate API (requires API key; reliable but costs money).

**Return path:** translate Gemini's patient-facing output back to the patient's language before display. Use the same translation model/service in both directions.

```
Voice input → Whisper ASR → language detect → IndicTrans2 (→ English)
→ [full pipeline, English internally]
→ Gemini output (English) → IndicTrans2 (→ patient's language) → display
```

---

## 15. Classifier Training: Data Pipeline

### 15.1 What a training sample looks like

A training sample is a `(input_string, specialty_label)` pair where `input_string` is in the exact format of `build_classifier_input()`. The preprocessing applied during training must be identical to what runs at inference.

**Turn-1 sample (from a transcript):**
```
SYMPTOM: chest tightness
ONSET: unknown
SEVERITY: unknown
CONTEXT: I've been having this tightness in my chest, it gets worse when I climb stairs
Label: Cardiology
```

**Multi-turn sample (from the same transcript, after 3 turns):**
```
SYMPTOM: chest tightness on exertion
ONSET: 3 days
SEVERITY: 6
CONTEXT: it comes with a bit of sweating and I have to stop to rest when it happens
Label: Cardiology
```

Two samples per transcript: one turn-1 style and one multi-turn style. This teaches the classifier to handle both sparse and rich input consistently.

### 15.2 Training data sources

These sources are for the **classifier only**, not for the RAG index. The two pipelines are separate.

| Priority | Source | Samples per class (approx.) | Notes |
|---|---|---|---|
| 1 | **`conversations.zip`** | Respiratory ~426, Ortho ~92, others ~10–24 | Highest quality; real patient language |
| 2 | **MedDialog (English)** — patient's first 1–3 turns only | 150–250 per class | Real; good phrasing diversity; fills Neurology, Psychiatry |
| 3 | **mtsamples** — chief complaint field + first paragraph | 80–150 per class | Short, clean; mapped to specialty by sample type tag |
| 4 | **Symptom2Disease** — symptom description text | 30–60 per class | After disease→specialty mapping; limited phrasing variety |
| 5 | **CounselChat** — patient's opening message only | ~50 for Psychiatry | Verify each sample describes a symptom; discard life-situation text |
| 6 | **Synthetic (Gemini-generated)** | Up to 100 per thin class | Only for Pediatrics and Dermatology if still below 150 after sources 1–5 |

### 15.3 Symptom2Disease → Specialty mapping

```json
{
  "Fungal infection": "Dermatology",
  "Allergy": "General Medicine",
  "GERD": "Gastroenterology",
  "Chronic cholestasis": "Gastroenterology",
  "Drug Reaction": "General Medicine",
  "Peptic ulcer disease": "Gastroenterology",
  "AIDS": "General Medicine",
  "Diabetes": "General Medicine",
  "Gastroenteritis": "Gastroenterology",
  "Bronchial Asthma": "Respiratory",
  "Hypertension": "Cardiology",
  "Migraine": "Neurology",
  "Cervical spondylosis": "Orthopedics",
  "Paralysis (brain hemorrhage)": "Neurology",
  "Jaundice": "Gastroenterology",
  "Malaria": "General Medicine",
  "Chicken pox": "General Medicine",
  "Dengue": "General Medicine",
  "Typhoid": "General Medicine",
  "Hepatitis A/B/C/D/E": "Gastroenterology",
  "Tuberculosis": "Respiratory",
  "Common Cold": "Respiratory",
  "Pneumonia": "Respiratory",
  "Dimorphic hemorrhoids": "Gastroenterology",
  "Heart attack": "Cardiology",
  "Varicose veins": "Cardiology",
  "Hypothyroidism": "General Medicine",
  "Hyperthyroidism": "General Medicine",
  "Hypoglycemia": "General Medicine",
  "Osteoarthritis": "Orthopedics",
  "Arthritis": "Orthopedics",
  "Acne": "Dermatology",
  "Urinary tract infection": "General Medicine",
  "Psoriasis": "Dermatology",
  "Impetigo": "Dermatology"
}
```

### 15.4 Synthetic data generation

Used only for Pediatrics and Dermatology if those classes are still below 150 real examples after all other sources. Synthetic samples go into the training set only — never into the evaluation set.

**Prompt template:**

```
Generate a patient's initial symptom description for a {SPECIALTY} case.
Requirements:
- Write in first person, informal English, as a patient would type (not a doctor)
- Vary vocabulary: use lay terms, with occasional medical terms mixed in
- Include: main symptom, approximate onset, rough severity indication
- Length: 1–3 sentences
- Do NOT include a diagnosis or disease name
- Do NOT include a doctor's response
- Make each generation distinct in phrasing

Example:
"I've had this sharp pain in my lower right side for about two days now. It
gets worse when I press on it and I had a slight fever last night."

Generate 1 sample.
```

Manually review every synthetic sample before adding it. Reject any sample that: contains a disease name; sounds like it was written by a clinician; or could plausibly belong to a different specialty.

Cap synthetic samples at 40% of any class's training total.

### 15.5 Dataset targets

| Class | Target train | Min real | Max synthetic |
|---|---|---|---|
| Respiratory | 300 | 250 | 50 |
| Orthopedics | 250 | 150 | 100 |
| Gastroenterology | 200 | 100 | 100 |
| Cardiology | 200 | 100 | 100 |
| Neurology | 200 | 150 | 50 |
| Psychiatry | 200 | 150 | 50 |
| General Medicine | 200 | 120 | 80 |
| Dermatology | 180 | 80 | 100 |
| Pediatrics | 180 | 80 | 100 |

**Held-out evaluation set:** 30 real, human-verified samples per class = 270 total. Zero synthetic samples. Never used in training. Built before training begins.

### 15.6 Sample extraction from transcripts

```python
import re

def extract_onset(text: str) -> str | None:
    match = re.search(r'(\d+)\s*(day|week|month|year)s?', text, re.IGNORECASE)
    return match.group(0) if match else None

def extract_severity(text: str) -> str | None:
    match = re.search(r'(\d+)\s*/\s*10|(\b(mild|moderate|severe|very\s+bad)\b)', text, re.IGNORECASE)
    return match.group(0) if match else None

def extract_training_samples(transcript_path: str, specialty: str) -> list[dict]:
    with open(transcript_path) as f:
        lines = f.readlines()

    # Parse into (speaker, text) pairs
    turns = []
    for line in lines:
        if line.startswith("D:"):
            turns.append(("D", line[2:].strip()))
        elif line.startswith("P:"):
            turns.append(("P", line[2:].strip()))

    patient_turns = [text for speaker, text in turns if speaker == "P"]
    if not patient_turns:
        return []

    samples = []

    # Turn-1 style sample
    first_turn = patient_turns[0]
    symptom = normalize_lay_terms(first_turn)
    samples.append({
        "input": build_classifier_input(
            chief_complaint=symptom,
            onset=None,
            severity=None,
            latest_patient_message=first_turn
        ),
        "label": specialty
    })

    # Multi-turn sample (turns 1–3 accumulated)
    if len(patient_turns) >= 3:
        accumulated = " ".join(patient_turns[:3])
        symptom_full = normalize_lay_terms(accumulated)
        onset = extract_onset(accumulated)
        severity = extract_severity(accumulated)
        samples.append({
            "input": build_classifier_input(
                chief_complaint=symptom_full,
                onset=onset,
                severity=severity,
                latest_patient_message=patient_turns[2]
            ),
            "label": specialty
        })

    return samples
```

---

## 16. Classifier Training: End-to-End Procedure

```python
import pickle
import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# ── STEP 1: Assemble labeled dataset ──────────────────────────────────────────

# Load all (input_string, label) pairs from your data pipeline
# Separate synthetic samples before splitting
all_real_samples = [...]      # list of {"input": str, "label": str}
all_synthetic_samples = [...]  # list of {"input": str, "label": str}

real_texts   = [s["input"] for s in all_real_samples]
real_labels  = [s["label"] for s in all_real_samples]
synth_texts  = [s["input"] for s in all_synthetic_samples]
synth_labels = [s["label"] for s in all_synthetic_samples]

# ── STEP 2: Stratified train/eval split on REAL samples only ─────────────────

train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    real_texts, real_labels,
    test_size=0.20,
    stratify=real_labels,
    random_state=42
)

# Add synthetic samples to train set only — never to eval
train_texts  = train_texts  + synth_texts
train_labels = train_labels + synth_labels

# ── STEP 3: Embed ─────────────────────────────────────────────────────────────

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Embedding training set...")
train_embeddings = embedding_model.encode(
    train_texts, batch_size=64, show_progress_bar=True
)  # shape: (N_train, 384)

print("Embedding eval set...")
eval_embeddings = embedding_model.encode(
    eval_texts, batch_size=64, show_progress_bar=True
)  # shape: (N_eval, 384)

# ── STEP 4: Train ─────────────────────────────────────────────────────────────

le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
eval_labels_enc  = le.transform(eval_labels)

clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42
)
clf.fit(train_embeddings, train_labels_enc)

# ── STEP 5: Evaluate ──────────────────────────────────────────────────────────

eval_preds = clf.predict(eval_embeddings)
report = classification_report(
    eval_labels_enc, eval_preds,
    target_names=le.classes_,
    output_dict=True
)
print(classification_report(eval_labels_enc, eval_preds, target_names=le.classes_))
print(confusion_matrix(eval_labels_enc, eval_preds))

macro_f1 = report["macro avg"]["f1-score"]
print(f"\nMacro-F1: {macro_f1:.3f}")
assert macro_f1 >= 0.70, (
    f"Macro-F1 {macro_f1:.3f} below acceptance threshold of 0.70. "
    "Fix data before proceeding."
)

# ── STEP 6: Serialize ─────────────────────────────────────────────────────────

with open('models/specialty_classifier.pkl', 'wb') as f:
    pickle.dump({'clf': clf, 'label_encoder': le}, f)

with open('models/classifier_metadata.json', 'w') as f:
    json.dump({
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'num_classes': len(le.classes_),
        'classes': list(le.classes_),
        'trained_on': datetime.now().isoformat(),
        'macro_f1': round(macro_f1, 4),
        'per_class_f1': {
            cls: round(report[cls]['f1-score'], 4)
            for cls in le.classes_
        }
    }, f, indent=2)

print("Model saved to models/specialty_classifier.pkl")
```

**At server startup:**

```python
with open('models/specialty_classifier.pkl', 'rb') as f:
    saved = pickle.load(f)
clf = saved['clf']
le  = saved['label_encoder']

# Shared with RAG retrieval — one instance, loaded once
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

The model is never retrained at runtime. Retrain only when new labeled data is deliberately added, as a manual offline step.

---

## 17. RAG Index Construction

### 17.1 Index A (Conversation index) — offline build script

```python
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

chunks = []

# ── Extract from conversations.zip ────────────────────────────────────────────
for specialty_folder in Path('conversations/').iterdir():
    specialty = specialty_folder.name
    for transcript_file in specialty_folder.glob('*.txt'):
        turns = parse_transcript(transcript_file)  # returns [(speaker, text)]
        for i in range(len(turns) - 1):
            if turns[i][0] == 'D' and turns[i+1][0] == 'P':
                chunk_text = f"D: {turns[i][1]} P: {turns[i+1][1]}"
                chunks.append({
                    "chunk_id": f"{specialty}_{transcript_file.stem}_turn_{i:03d}",
                    "specialty": specialty,
                    "text": chunk_text,
                    "source": str(transcript_file),
                    "turn_index": i
                })

# ── Extract from MedDialog (after specialty mapping) ──────────────────────────
# Similar extraction; map MedDialog department labels to your 9 specialties

# ── Embed all chunks ──────────────────────────────────────────────────────────
texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks...")
embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize for cosine

# ── Build FAISS index ─────────────────────────────────────────────────────────
dim = embeddings.shape[1]  # 384
nlist = 256
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)

# ── Save ─────────────────────────────────────────────────────────────────────
faiss.write_index(index, 'indexes/conversation_index.faiss')
with open('indexes/conversation_chunks.json', 'w') as f:
    json.dump(chunks, f)

print(f"Index A built: {len(chunks)} chunks")
```

### 17.2 Index B (Knowledge index) — same procedure, different sources

Replace the chunk extraction step with:

```python
# MedQuAD: one chunk per QA pair
# MedlinePlus: one chunk per section (symptoms / causes / when-to-see-a-doctor)
# Tag each chunk with specialty inferred from source category
```

Save as `indexes/knowledge_index.faiss` and `indexes/knowledge_chunks.json`.

### 17.3 Index loading at server startup

```python
import faiss, json

conv_index  = faiss.read_index('indexes/conversation_index.faiss')
conv_chunks = json.load(open('indexes/conversation_chunks.json'))

know_index  = faiss.read_index('indexes/knowledge_index.faiss')
know_chunks = json.load(open('indexes/knowledge_chunks.json'))

# Build fast specialty→chunk_id lookup for filtering
from collections import defaultdict
specialty_to_conv_ids = defaultdict(list)
for i, chunk in enumerate(conv_chunks):
    specialty_to_conv_ids[chunk["specialty"]].append(i)
```

---

## 18. Evaluation Criteria

### 18.1 Classifier

Report all of the following. Do not report accuracy alone.

| Metric | Target | Notes |
|---|---|---|
| Macro-F1 | ≥ 0.70 | Unweighted average across all 9 classes |
| Per-class F1 | ≥ 0.60 for each class | If Pediatrics or Dermatology fall below 0.60, add a disclaimer in the patient-facing UI when those specialties are predicted |
| Confusion matrix | Inspect manually | Respiratory/General Medicine is the expected hardest pair |
| Calibration | Inspect probability histograms | High-confidence predictions should be correct more often |

Reject the model and fix data if macro-F1 < 0.70. Do not wire an underperforming classifier into the live system.

### 18.2 RAG retrieval (Index A)

Manual spot-check: for 5 symptom descriptions per specialty (45 total), run retrieval and verify:

- Top-3 chunks are from the correct specialty (or a sensibly related one for thin specialties).
- Chunks are semantically similar to the query.
- Each chunk contains a doctor follow-up question that would be plausible in this clinical context.

This takes ~2 hours. Do it before wiring RAG into the live prompt.

### 18.3 Prognosis helper

Manually review 20 generated outputs (2–3 per specialty) and verify:

- No diagnostic assertion language slipped through.
- The content is grounded in the retrieved chunks, not invented.
- The fixed disclaimer is visible on the result screen alongside the output.

### 18.4 Emergency detector

Test against 20 adversarial inputs, covering: standard phrasings ("chest pain"), indirect phrasings ("I feel like my heart is stopping"), false positives ("my back is killing me"), and multilingual phrasings post-translation.

False negatives are more dangerous than false positives. When uncertain, the detector should trigger.

---

## 19. Implementation Checklist

### Phase 1 — Data and Offline Artifacts

- [ ] Download MedDialog (English) from Hugging Face
- [ ] Download mtsamples from Kaggle
- [ ] Download Symptom2Disease from Kaggle
- [ ] Download MedQuAD from Hugging Face (`keivalya/MedQuad-MedicalQnADataset`)
- [ ] Download MedlinePlus XML dump
- [ ] Build disease→specialty mapping JSON (§ 15.3)
- [ ] Write transcript extraction function; test on 5 transcripts
- [ ] Extract training samples from conversations.zip; verify label distribution
- [ ] Extract training samples from MedDialog patient turns
- [ ] Extract training samples from mtsamples chief complaint fields
- [ ] Extract training samples from Symptom2Disease (post-mapping)
- [ ] Extract Psychiatry samples from CounselChat (manual verify each)
- [ ] Generate synthetic samples for Pediatrics + Dermatology only (§ 15.4)
- [ ] Manually review all synthetic samples before adding
- [ ] Assemble final labeled dataset per § 15.5 targets
- [ ] Build held-out eval set (30 real, verified per class; 0 synthetic)
- [ ] Run full training procedure (§ 16)
- [ ] Confirm macro-F1 ≥ 0.70 and per-class F1 ≥ 0.60
- [ ] Inspect confusion matrix; document Respiratory/General Medicine overlap
- [ ] Build Index A (§ 17.1): chunk transcripts + MedDialog, embed, FAISS
- [ ] Build Index B (§ 17.2): chunk MedQuAD + MedlinePlus, embed, FAISS
- [ ] Spot-check Index A retrieval for 5 queries per specialty (45 total)

### Phase 2 — AI Pipeline Integration

- [ ] Implement and unit-test Emergency Detector; test all regex patterns (§ 5)
- [ ] Build lay-term normalizer dictionary (200+ entries); verify on 20 phrasings (§ 6)
- [ ] Implement `build_classifier_input()` — single shared function (§ 1.2)
- [ ] Implement `classify_specialty()` with confidence bucketing (§ 7.3)
- [ ] Implement FAISS retrieval for Index A with specialty filter + fallback (§ 8.3)
- [ ] Implement FAISS retrieval for Index B (§ 11.2)
- [ ] Implement Gemini slot extractor + conversation engine with full prompt template (§ 9)
- [ ] Test Gemini slot extraction across 10 varied patient inputs; verify JSON schema
- [ ] Implement `SessionState` with update logic (§ 9.3)
- [ ] Implement triage level mapper; unit test all branches (§ 10)
- [ ] Implement prognosis helper with output filter (§ 11)
- [ ] Manually review 20 prognosis outputs
- [ ] Implement doctor brief generator (§ 12)
- [ ] Verify `Appointment.ai_brief` is never exposed to patient-facing API routes
- [ ] Confirm fixed disclaimer is hardcoded in frontend (not generated)
- [ ] Implement confidence threshold behavior including Low-confidence fallback (§ 7.4)
- [ ] Implement mid-session specialty update on slot extraction contradiction (§ 7.4)

### Phase 3 — System Integration

- [ ] Connect triage level to scheduling engine slot priority
- [ ] Implement doctor recommendation ranker (§ 13)
- [ ] Build doctor portal (brief display, queue, slot management)
- [ ] Implement payment mock
- [ ] Integrate ASR + translation pipeline (§ 14)
- [ ] End-to-end test: voice input (regional language) → full session → appointment booked → brief visible in doctor portal

### Phase 4 — Evaluation and Hardening

- [ ] Run full held-out eval set; record per-class F1 and confusion matrix for tech report
- [ ] Run emergency detector adversarial test (20 inputs)
- [ ] Confirm output filter blocks all diagnostic-assertion patterns in sampled prognosis outputs
- [ ] Measure per-turn latency; verify Gemini call is the bottleneck; local components add < 50ms
- [ ] Verify embedding model loads once at startup; confirm no per-request reloading

---

## 20. Known Limitations to Document

**Pediatrics is the weakest class.** Public pediatric-specific symptom corpora are scarce. This class will carry the highest proportion of synthetic training data, and its per-class F1 should be reported separately in the tech report.

**Respiratory / General Medicine boundary is fuzzy.** These two classes share many surface features (cough, fatigue, breathlessness). The confusion matrix will likely show the highest off-diagonal cell at this pair. Acceptable mitigation: when the classifier is Low confidence and the top-2 are Respiratory and General Medicine, show both as options on the result screen.

**Synthetic data carries LLM biases.** Any class whose training set exceeds 40% synthetic may perform well on the evaluation set but underperform on real users who phrase things in ways the LLM does not generate. Document the synthetic percentage per class in the tech report.

**Emergency detector has no recall guarantee.** A regex system will miss novel or indirect phrasings. This is acceptable for a hackathon prototype but must be stated explicitly as a known limitation. Production deployment would require clinical review, ongoing monitoring, and likely a dedicated safety classifier.

**No data residency analysis.** Patient symptom text is sent to a third-party LLM API (Gemini / OpenAI). For a real medical deployment, this raises HIPAA, GDPR, and data residency questions that are out of scope for a hackathon but must be flagged before any real-world use.

**Retrieval quality degrades for thin specialties.** Index A has ~10x more chunks for Respiratory than for Gastroenterology or Dermatology. RAG-driven conversation quality will be noticeably better for well-represented specialties. Document this gap and consider it when interpreting demo results.
