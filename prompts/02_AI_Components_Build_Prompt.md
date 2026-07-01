# TriagePlus — AI Components Build Prompt
### IIT Dharwad Summer of Innovation · Hardly Human
> This file covers every AI component: the Emergency Detector, Lay-Term Normalizer, Specialty Classifier (training + inference), RAG retrieval, Gemini integration, Triage Mapper, Prognosis Helper, Doctor Brief Generator, Doctor Recommendation Ranker, and Multilingual Voice pipeline. Implement each component in the order listed.

---

## Locked Design Decisions (Read Before Writing Any Code)

| Decision | Locked Value |
|---|---|
| LLM | Gemini 1.5 Flash (`gemini-1.5-flash`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, loaded once) |
| Classifier head | `sklearn.linear_model.LogisticRegression` over frozen MiniLM embeddings |
| Specialty taxonomy | 9 classes (see below) |
| Triage system | 4 levels, rule-based — no ML |
| Synthetic data cap | ≤ 40% of any class's training total |

### 9 Specialty Classes

```
Cardiology · Dermatology · Orthopedics · Gastroenterology
Neurology · Pediatrics · Psychiatry · General Medicine · Respiratory
```

Respiratory is a distinct class — not merged into General Medicine. The corpus has 213+ Respiratory transcripts.

### Fixed Classifier Input Contract

Every call to the classifier — during training data construction and during live inference — uses this exact string format via a single shared function:

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

Never create two separate implementations of this function. Training and inference must use the exact same code path.

---

## Component 1: Emergency Detector

**Type:** Regex. No ML. No API call. Runs first on every single patient message.

If any pattern matches, the pipeline stops immediately and returns the emergency template. The patient message never reaches any downstream component.

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

**On match:**
1. Log to `AuditLog`: `event_type="emergency_detected"`, `payload=<matched pattern>`.
2. Send this fixed response (never LLM-generated):
   ```
   ⚠️ This sounds like a medical emergency. Please call 112 immediately or go to your nearest emergency room. Do not wait for an appointment.
   ```
3. Close the WebSocket connection.
4. Do not call Gemini, the classifier, or any other component.

**Testing requirement:** test against 20 adversarial inputs covering: standard phrasings, indirect phrasings ("my heart is stopping"), false positives ("my back is killing me"), post-translation multilingual phrasings. False negatives are more dangerous than false positives — when uncertain, the detector should trigger.

---

## Component 2: Lay-Term Normalizer

**Type:** Dictionary lookup. No ML. No API call. Runs after emergency check, before embedding.

Load the CSV at startup into a Python dict. Apply longest-match, case-insensitive substring replacement to the patient message. The original patient text is preserved for display — patients never see the normalized clinical terms reflected back at them. Only the normalized form is passed to the embedding model.

### Minimum required entries (build to ~200–300 total):

```csv
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
hair falling,alopecia
sad all the time,persistent low mood
heart racing,palpitations
can't breathe,dyspnea
yellow eyes,jaundice
tummy ache,abdominal pain
```

Build coverage across all 9 specialties. The normalizer does not need to be exhaustive — its purpose is to close the gap between the most common lay phrasings and their clinical equivalents.

---

## Component 3: Specialty Classifier

### Training Data Pipeline

The classifier takes `build_classifier_input()` strings and outputs a specialty label. Training must use the exact same `build_classifier_input()` function used at inference.

**Sources (use in this priority order):**

| Priority | Source | Approx. samples/class | Notes |
|---|---|---|---|
| 1 | `conversations.zip` | Respiratory ~426, Ortho ~92, others ~10–24 | Highest quality; real patient language |
| 2 | MedDialog (English, Hugging Face `medical_dialog`) | 150–250/class | Patient's first 1–3 turns only |
| 3 | mtsamples (Kaggle) | 80–150/class | Chief complaint field + first paragraph |
| 4 | Symptom2Disease (Kaggle) | 30–60/class | After disease→specialty mapping (see JSON below) |
| 5 | CounselChat | ~50 for Psychiatry only | Patient's opening message; discard life-situation text |
| 6 | Synthetic (Gemini-generated) | Up to 100/thin class | Pediatrics and Dermatology only; review each sample manually |

**Dataset targets:**

| Class | Target train | Min real | Max synthetic |
|---|---|---|---|
| Respiratory | 300 | 250 | 50 |
| Orthopedics | 250 | 150 | 100 |
| Neurology | 200 | 150 | 50 |
| Psychiatry | 200 | 150 | 50 |
| Gastroenterology | 200 | 100 | 100 |
| Cardiology | 200 | 100 | 100 |
| General Medicine | 200 | 120 | 80 |
| Dermatology | 180 | 80 | 100 |
| Pediatrics | 180 | 80 | 100 |

**Held-out evaluation set:** 30 real, human-verified samples per class = 270 total. Zero synthetic. Build before training begins. Never used in training.

**Symptom2Disease → Specialty mapping:**

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

**Sample extraction from transcripts:**

```python
def extract_training_samples(transcript_path: str, specialty: str) -> list[dict]:
    with open(transcript_path) as f:
        lines = f.readlines()

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

    # Turn-1 sample (minimal information)
    first_turn = patient_turns[0]
    symptom = normalize_lay_terms(first_turn)
    samples.append({
        "input": build_classifier_input(
            chief_complaint=symptom, onset=None, severity=None,
            latest_patient_message=first_turn
        ),
        "label": specialty
    })

    # Multi-turn sample (turns 1–3 accumulated)
    if len(patient_turns) >= 3:
        accumulated = " ".join(patient_turns[:3])
        samples.append({
            "input": build_classifier_input(
                chief_complaint=normalize_lay_terms(accumulated),
                onset=extract_onset(accumulated),
                severity=extract_severity(accumulated),
                latest_patient_message=patient_turns[2]
            ),
            "label": specialty
        })

    return samples
```

**Synthetic data generation prompt (only for Pediatrics and Dermatology, ≤40% of class total):**

```
Generate a patient's initial symptom description for a {SPECIALTY} case.
Requirements:
- Write in first person, informal English, as a patient would type
- Use lay terms; avoid disease names
- Include: main symptom, approximate onset, rough severity indication
- Length: 1–3 sentences
- Do NOT include a diagnosis or disease name
- Do NOT include a doctor's response
Generate 1 sample.
```

Manually review every synthetic sample. Reject any that contain a disease name, sound clinical, or could plausibly belong to a different specialty.

### End-to-End Training Procedure

```python
import pickle, json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# ── STEP 1: Assemble dataset ───────────────────────────────────────────────────
all_real_samples  = [...]   # list of {"input": str, "label": str}
all_synth_samples = [...]   # synthetic samples — training only, never eval

real_texts  = [s["input"] for s in all_real_samples]
real_labels = [s["label"] for s in all_real_samples]

# ── STEP 2: Stratified train/eval split on REAL samples only ──────────────────
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    real_texts, real_labels, test_size=0.20, stratify=real_labels, random_state=42
)
# Append synthetic to train only — NEVER to eval
train_texts  = train_texts  + [s["input"] for s in all_synth_samples]
train_labels = train_labels + [s["label"] for s in all_synth_samples]

# ── STEP 3: Embed ─────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_embeddings = embedding_model.encode(train_texts, batch_size=64, show_progress_bar=True)
eval_embeddings  = embedding_model.encode(eval_texts,  batch_size=64, show_progress_bar=True)

# ── STEP 4: Train ─────────────────────────────────────────────────────────────
le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
eval_labels_enc  = le.transform(eval_labels)

clf = LogisticRegression(
    multi_class='multinomial', solver='lbfgs',
    class_weight='balanced', max_iter=1000, C=1.0, random_state=42
)
clf.fit(train_embeddings, train_labels_enc)

# ── STEP 5: Evaluate — must pass before wiring into live system ───────────────
eval_preds = clf.predict(eval_embeddings)
report = classification_report(
    eval_labels_enc, eval_preds, target_names=le.classes_, output_dict=True
)
print(classification_report(eval_labels_enc, eval_preds, target_names=le.classes_))
print(confusion_matrix(eval_labels_enc, eval_preds))

macro_f1 = report["macro avg"]["f1-score"]
print(f"\nMacro-F1: {macro_f1:.3f}")
assert macro_f1 >= 0.70, (
    f"Macro-F1 {macro_f1:.3f} below 0.70. Fix data before wiring into system."
)
# Also check: if any per-class F1 < 0.60 for Pediatrics or Dermatology,
# add a UI disclaimer when those specialties are predicted.

# ── STEP 6: Save ──────────────────────────────────────────────────────────────
import os
os.makedirs('models', exist_ok=True)
with open('models/specialty_classifier.pkl', 'wb') as f:
    pickle.dump({'clf': clf, 'label_encoder': le}, f)

with open('models/classifier_metadata.json', 'w') as f:
    json.dump({
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'classes': list(le.classes_),
        'trained_on': datetime.now().isoformat(),
        'macro_f1': round(macro_f1, 4),
        'per_class_f1': {cls: round(report[cls]['f1-score'], 4) for cls in le.classes_}
    }, f, indent=2)
```

### Live Inference (`/backend/app/services/classifier.py`)

```python
def classify_specialty(session_state: SessionState, latest_message: str) -> dict:
    input_str = build_classifier_input(
        chief_complaint=session_state.chief_complaint or latest_message,
        onset=session_state.onset,
        severity=session_state.severity,
        latest_patient_message=latest_message
    )

    # Embed — this is the SAME vector fed to FAISS retrieval
    embedding = embedding_model.encode([input_str])  # (1, 384)

    proba     = clf.predict_proba(embedding)[0]       # (9,)
    top_idx   = np.argmax(proba)
    top_class = le.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]

    if confidence >= 0.70:
        confidence_label = "High"
    elif confidence >= 0.45:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

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

**Confidence behavior at runtime:**

| Label | Condition | RAG behavior | Result screen behavior |
|---|---|---|---|
| High | ≥ 0.70 | 3 chunks from top-1 specialty | Show top-1 specialty |
| Medium | 0.45–0.69 | 3 chunks from top-1 specialty | Show top-1 specialty |
| Low | < 0.45 | 2 chunks from each of top-2 specialties | Show both + suggest General Medicine |

**Mid-session specialty update:** after each Gemini slot extraction, re-run the classifier on the newly extracted chief_complaint. If the new top class has probability > 0.60 and differs from the provisional specialty, silently update the RAG filter for the next turn. Never alert the patient during intake. Only show the final specialty on the result screen.

**Load at server startup:**

```python
with open('models/specialty_classifier.pkl', 'rb') as f:
    saved = pickle.load(f)
clf = saved['clf']
le  = saved['label_encoder']
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# This single embedding_model instance is shared with FAISS retrieval — never reload per request
```

---

## Component 4: RAG System

### Two indexes — content must never be mixed:

- **Index A:** D:/P: exchange chunks from transcripts and MedDialog. Used on every patient turn during intake. Guides Gemini's follow-up questions.
- **Index B:** MedQuAD QA pairs and MedlinePlus condition sections. Used once at the prognosis helper step only.

### FAISS Retrieval for Index A (`/backend/app/services/rag.py`)

```python
async def retrieve_conversation_examples(
    query_text: str,
    predicted_specialty: str,
    confidence_label: str,
    top_k: int = 3,
    min_score: float = 0.28,
    session_id: str = None
) -> list[dict]:
    """
    1. Check Redis for cached embedding: key = f"embed:{session_id}", TTL 30min
    2. If miss: embed with MiniLM (in thread pool — never block event loop)
    3. Search full Index A for top-20 candidates
    4. Filter to predicted_specialty metadata
    5. Apply cosine score threshold (0.28)
    6. If confidence_label == "Low": query top-2 specialties, 2 chunks each
    7. If fewer than 2 chunks pass: relax specialty filter, retry with full index
    8. Return top_k chunks
    """

    # Embedding (always run in executor)
    loop = asyncio.get_event_loop()
    cached = await redis_client.get(f"embed:{session_id}")
    if cached:
        query_vec = np.frombuffer(cached, dtype=np.float32).reshape(1, -1)
    else:
        query_vec = await loop.run_in_executor(
            None, partial(embedding_model.encode, [query_text], normalize_embeddings=True)
        )
        await redis_client.setex(
            f"embed:{session_id}", 1800, query_vec.astype(np.float32).tobytes()
        )

    # FAISS search
    scores, indices = conv_index.search(query_vec.astype(np.float32), 20)

    # Specialty filter + score threshold
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = conv_chunks[idx]
        if chunk["specialty"] == predicted_specialty and score >= min_score:
            results.append({"chunk": chunk, "score": float(score)})

    # Fallback if thin
    if len(results) < 2:
        results = [
            {"chunk": conv_chunks[i], "score": float(s)}
            for s, i in zip(scores[0], indices[0])
            if i != -1 and scores[0][list(indices[0]).index(i)] >= min_score
        ]

    return [r["chunk"] for r in results[:top_k]]
```

### FAISS Retrieval for Index B (prognosis helper only)

```python
async def retrieve_knowledge_chunks(
    query_text: str,
    predicted_specialty: str,
    top_k: int = 3,
    min_score: float = 0.30
) -> list[str]:
    """
    Used once, after the final specialty is determined.
    Query: the fully-filled slot string from build_classifier_input().
    Filter by predicted_specialty metadata.
    """
    # Same embed+search pattern as Index A
    # Returns text content of top_k chunks
```

---

## Component 5: Slot Extractor and Conversation Engine (Gemini)

### Slots tracked:

| Slot | Required | Description |
|---|---|---|
| `chief_complaint` | Yes | Primary symptom, normalized |
| `severity` | Yes | Patient-reported intensity ("7", "severe", "comes and goes") |
| `onset` | Yes | When it started ("3 days", "since last week") |
| `associated_symptoms` | No | Other symptoms mentioned |
| `medical_history_flags` | No | PMH volunteered by patient |

### Rules (enforced via system prompt):

- Ask ONE question per turn.
- Ask slots in order: chief_complaint → severity → onset. Never jump ahead.
- Never ask for information already provided.
- Never suggest a diagnosis or use a disease name.
- If the patient volunteers slot info without being asked, extract it and ask about the next missing slot.
- If onset is vague ("recently"), accept it as filled.

### Gemini SDK setup:

```python
import google.generativeai as genai

genai.configure(api_key=settings.GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=genai.GenerationConfig(temperature=0.1)
)
```

### System prompt template:

```python
def build_system_prompt(
    state: SessionState,
    rag_chunks: list[dict],
    classifier_result: dict
) -> str:
    missing = [f for f in ['chief_complaint', 'severity', 'onset'] if not getattr(state, f)]
    next_slot = missing[0] if missing else None

    examples_text = "\n---\n".join(
        c["text"][:300] for c in rag_chunks
    ) if rag_chunks else "None available"

    specialist_hint = classifier_result.get("specialty", "unknown") if classifier_result else "unknown"
    conf_hint = classifier_result.get("confidence_label", "") if classifier_result else ""

    return f"""You are TriagePlus, an AI medical intake assistant.

STRICT RULES:
- NEVER diagnose. NEVER say "you have [condition]". Say "this sounds like something a [specialist] should evaluate".
- NEVER recommend specific treatments or medications.
- Ask only ONE question per turn.
- If patient mentions chest pain, difficulty breathing, stroke symptoms, or says they are dying: tell them to call 112 immediately.
- Extract info using the JSON schema whenever the patient shares symptom details.
- Be warm, empathetic, concise. The patient may be worried.
- Address the patient by name ({state.patient_name}) occasionally.

CURRENT SESSION STATE:
- chief_complaint: {state.chief_complaint or 'unknown'}
- severity: {state.severity or 'unknown'}
- onset: {state.onset or 'unknown'}
- Missing slots: {missing}
- Next slot to collect: {next_slot}

SYSTEM CLASSIFIER (internal — never share with patient):
Provisional specialty: {specialist_hint} (confidence: {conf_hint})

RETRIEVED CONVERSATION EXAMPLES [{specialist_hint}]:
---
{examples_text}
---
Use these as style guidance for follow-up questions. Do not copy them verbatim.

TASK:
The next missing required slot is: {next_slot or 'ALL FILLED — transition to recommendation'}.
Ask a natural follow-up question that gathers {next_slot}.

Respond with a JSON object:
{{
  "patient_message": "<your conversational question to the patient>",
  "extracted": {{
    "chief_complaint": "<if newly found in this message, else null>",
    "severity": "<if newly found, else null>",
    "onset": "<if newly found, else null>",
    "associated_symptoms": ["<any mentioned>"],
    "medical_history_flags": ["<any mentioned>"]
  }},
  "slots_filled": {{
    "chief_complaint": {str(bool(state.chief_complaint)).lower()},
    "severity": {str(bool(state.severity)).lower()},
    "onset": {str(bool(state.onset)).lower()}
  }},
  "all_required_slots_filled": false
}}"""
```

### Calling Gemini:

```python
response = await model.generate_content_async(
    patient_message,
    request_options={"timeout": 30},  # Always set timeout
)
```

### Required error handling:

```python
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted

try:
    response = await model.generate_content_async(...)
except DeadlineExceeded:
    return "I'm having trouble connecting right now. Could you repeat that?"
except ResourceExhausted:
    return "Please try again in a moment."
```

### Guardrails:

**Input guard (run before any text reaches Gemini):**
```python
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"pretend\s+you\s+are",
    r"forget\s+(everything|your\s+instructions)",
    r"disregard\s+your",
    r"system\s*:\s*you",
]

def check_prompt_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)
```

**Output guard (run before any Gemini output reaches the patient or doctor brief):**
```python
DIAGNOSTIC_ASSERTION_PATTERNS = [
    r"\byou\s+have\b",
    r"\byou'?re\s+(diagnosed|suffering)\b",
    r"\bthis\s+is\s+(likely|probably|certainly)\b",
    r"\byou\s+(likely|probably|definitely)\b",
    r"\byour\s+(condition|diagnosis)\s+is\b",
    r"\bI\s+(believe|think|suspect)\s+you\b",
]

def check_diagnosis_assertion(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in DIAGNOSTIC_ASSERTION_PATTERNS)
```

If the output guard triggers, make one Gemini rewrite attempt. If the rewrite also triggers, fall back to the fixed safe response (see Prognosis Helper below).

### Pydantic validation of Gemini JSON output:

All Gemini responses must be validated against `IntakeSlots` before the FSM accepts them:

```python
from pydantic import BaseModel

class ExtractedSlots(BaseModel):
    chief_complaint: str | None = None
    severity: str | None = None
    onset: str | None = None
    associated_symptoms: list[str] = []
    medical_history_flags: list[str] = []

class GeminiIntakeResponse(BaseModel):
    patient_message: str
    extracted: ExtractedSlots
    slots_filled: dict[str, bool]
    all_required_slots_filled: bool
```

If JSON parsing or validation fails, send a fallback question to the patient and do not update `SessionState`.

---

## Component 6: Triage Level Mapper

**Type:** Rule-based. No ML. No API call. Runs after all slots are filled.

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

    if state.severity_numeric and state.severity_numeric >= 9:
        return 1
    if state.chief_complaint and any(kw in state.chief_complaint for kw in LEVEL1_KEYWORDS):
        return 1

    if state.severity_numeric and state.severity_numeric >= 7:
        return 2
    if state.onset_days is not None and state.onset_days <= 1:
        if state.severity_numeric and state.severity_numeric >= 5:
            return 2
    if state.chief_complaint and any(kw in state.chief_complaint for kw in LEVEL2_KEYWORDS):
        return 2

    if state.severity_numeric and state.severity_numeric >= 5:
        return 3
    if state.onset_days is not None and state.onset_days <= 7:
        return 3

    return 4
```

**Triage level feeds the scheduling engine:**
- Level 1 → trigger emergency response, no appointment booked.
- Level 2 → filter doctors to same-day/next-day availability only.
- Level 3 → slots within 3 days.
- Level 4 → any slot within 7 days.

---

## Component 7: Prognosis Helper

A single Gemini call that generates up to 3 general condition notes grounded in Index B knowledge chunks. It is not a diagnosis. It is not presented as one.

Runs once, after the final classifier pass and triage computation. Its output is stored and reused for both the patient result screen and the doctor brief — never regenerated.

### Prompt:

```python
def build_prognosis_prompt(state: SessionState, knowledge_chunks: list[str]) -> str:
    chunks_text = "\n---\n".join(knowledge_chunks[:3])
    return f"""SYSTEM:
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
- Chief symptom: {state.chief_complaint}
- Severity: {state.severity}
- Onset: {state.onset}
- Predicted specialty: {state.provisional_specialty}

Retrieved condition context:
---
{chunks_text}
---

Using only the context above, list up to 3 conditions commonly discussed in
similar presentations. Do not use general knowledge beyond what is provided here."""
```

### Output filter:

```python
def filter_prognosis_output(text: str, state: SessionState) -> str:
    if check_diagnosis_assertion(text):
        # One rewrite attempt
        rewrite = gemini_rewrite(text)
        if check_diagnosis_assertion(rewrite):
            # Fall through to hardcoded safe response
            return (
                f"Based on similar presentations, conditions related to the "
                f"{state.provisional_specialty} system are worth discussing with your doctor. "
                f"Please consult a qualified medical professional for a proper assessment."
            )
        return rewrite
    return text
```

### Fixed disclaimer (hardcoded in frontend — never LLM-generated):

```
⚠ This information is general in nature and does not constitute a medical
diagnosis. Always consult a qualified medical professional for advice
specific to your situation.
```

This text must be hardcoded as a constant in the frontend React component. It cannot be modified or omitted by any model output.

---

## Component 8: Doctor Brief Generator

A separate Gemini call triggered after payment confirmation. Produces a structured clinical brief stored as `Appointment.ai_brief`. Never visible to the patient.

The doctor brief is not a reuse of the prognosis helper output. Doctors and patients are different audiences.

```python
def build_doctor_brief_prompt(state: SessionState, prognosis_text: str) -> str:
    return f"""SYSTEM:
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
{state.model_dump_json(indent=2)}

Prognosis context (for background only):
{prognosis_text}"""
```

Store the result in `Appointment.ai_brief`. Render only in the doctor portal. Verify that `ai_brief` is never included in any patient-facing API response schema.

---

## Component 9: Doctor Recommendation Ranker

Deterministic scoring. No ML.

```python
def score_doctor(doctor: Doctor, triage_level: int) -> float:
    rating_norm       = doctor.rating / 5.0
    days_to_next_slot = doctor.next_available_slot_days
    availability_score = 1.0 / (1.0 + days_to_next_slot)
    feedback_norm     = doctor.feedback_score / 5.0

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
        # Level 2 (Urgent): only doctors with same-day/next-day slots
    ]
    return sorted(filtered, key=lambda d: score_doctor(d, triage_level), reverse=True)[:top_n]
```

---

## Component 10: Multilingual Voice Input

### English — browser-native (zero cost, zero server calls):

```typescript
// In VoiceButton.tsx
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = "en-IN";
recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    // Submit transcript as a normal chat message through the existing WebSocket
};
// Hide the mic button if SpeechRecognition is not available in the browser
if (!SpeechRecognition) hideMicButton();
```

### Regional languages — server-side Whisper:

```
POST /api/v1/voice/transcribe
Content-Type: multipart/form-data
Body: audio=<webm blob>, language=<"hi"|"kn"|"ta"|"te"|"mr">
Returns: {"transcript": "..."}
```

**ASR:** OpenAI Whisper API (`whisper-1` model). Fallback: Google Cloud STT with BCP-47 codes (`hi-IN`, `kn-IN`, `ta-IN`, `te-IN`).

**Translation pipeline:**
```
Voice → Whisper ASR → detect language → if non-English:
  → IndicTrans2 (local, open-source, 22 scheduled Indian languages)
  → English text → standard AI pipeline
  → Gemini output (English) → IndicTrans2 → patient's language → display
```

Fallback if IndicTrans2 fails: Google Translate API.

Store both the original-language text and the English translation in `SessionState`.

---

## Evaluation Requirements

### Classifier — report all of these:

| Metric | Target |
|---|---|
| Macro-F1 | ≥ 0.70 |
| Per-class F1 | ≥ 0.60 for every class |
| Confusion matrix | Inspect manually; Respiratory/General Medicine is the expected hardest pair |

Reject the model and fix data if macro-F1 < 0.70. Do not wire an underperforming classifier into the live system.

If Pediatrics or Dermatology per-class F1 < 0.60, add a UI disclaimer on the result screen when those specialties are predicted.

### RAG spot-check (Index A):

For 5 symptom descriptions per specialty (45 total), verify:
- Top-3 retrieved chunks are from the correct specialty (or a sensibly related one for thin specialties).
- Each chunk contains a follow-up question plausible for that clinical context.

Do this before wiring RAG into the live Gemini prompt.

### Prognosis helper:

Manually review 20 outputs (2–3 per specialty). Verify: no diagnostic assertion language, content is grounded in retrieved chunks, and the fixed disclaimer is visible alongside the output in the UI.

### Emergency detector:

Test against 20 adversarial inputs: standard phrasings, indirect phrasings, false positives, post-translation phrasings.

---

## Known Limitations to Document

- **Pediatrics is the weakest class.** Report its per-class F1 separately in the tech report.
- **Respiratory/General Medicine boundary is fuzzy.** When classifier is Low confidence and top-2 are Respiratory and General Medicine, show both as options on the result screen.
- **Synthetic data carries LLM biases.** Document synthetic percentage per class.
- **Emergency detector has no recall guarantee.** A regex system will miss novel phrasings. State this explicitly as a known limitation.
- **No data residency analysis.** Patient symptom text is sent to Gemini API. Flag this for any real-world deployment.

---

*TriagePlus · IIT Dharwad Summer of Innovation · Hardly Human · Mentor: Prof. B. N. Bharath*
