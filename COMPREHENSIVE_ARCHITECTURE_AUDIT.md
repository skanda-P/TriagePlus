# COMPREHENSIVE LANGGRAPH ARCHITECTURE AUDIT

## Executive Summary
Reviewed against your specific requirements:
- **MedQuAD handling**: ❌ NOT IMPLEMENTED CORRECTLY
- **Hybrid retrieval (BM25 + Dense)**: ❌ NOT IMPLEMENTED
- **Medical embedding model (MedCPT/PubMedBERT)**: ✅ USING PubMedBERT (correct)
- **KG-RAG integration**: ⚠️ PARTIAL (KG exists, but not integrated with RAG properly)
- **Graph-based retrieval**: ❌ NOT IMPLEMENTED
- **Three-layer emergency detection**: ❌ ONLY KEYWORD LAYER IMPLEMENTED
- **Ollama fallback with hardcoded slots**: ⚠️ INCOMPLETE
- **NER model**: ❌ DUMMY NER - placeholder only

---

## CRITICAL ISSUES FOUND

### 1. RAG IMPLEMENTATION - MEDQUAD CHUNKING VIOLATION ❌❌
**File**: `backend/app/core/rag.py`
**Severity**: CRITICAL

**Current Implementation** (Lines 1-60):
```python
self.medquad_index = FAISS.load_local(medquad_path, self.embeddings)
results = self.medquad_index.similarity_search_with_score(query, k=k)
```

**Problems**:
1. **No QA-pair atomic chunking**: The code assumes pre-built FAISS indices exist but doesn't implement the requirement that MedQuAD should be chunked as:
   - Question + Answer = ONE chunk (not split by token count)
   - Long answers (500+ tokens) split by paragraph with parent-child linking
   - Metadata tags: `focus_disease`, `question_type` (symptoms/treatment/prognosis), `source`

2. **No filtering by metadata**: The query function doesn't restrict retrieval based on question_type. When extracting symptoms, it should filter `question_type=symptoms` only.

3. **No build script**: `backend/scripts/build_faiss_indices.py` DOES NOT EXIST. You cannot build these indices without a build script.

**Questions Before Fixing**:
- Should I create the `build_faiss_indices.py` script that:
  1. Parses MedQuAD JSON/XML
  2. Chunks as QA pairs with parent-child linking for long answers
  3. Attaches metadata (focus_disease, question_type, source)
  4. Builds separate FAISS indices for medquad and medquad_long_form?

- Should I also implement the paragraph-level retrieval logic that joins child chunks back to parent?

---

### 2. RETRIEVAL TECHNIQUE - MISSING HYBRID BM25 + DENSE ❌
**File**: `backend/app/core/rag.py` → `query_medquad()` method
**Severity**: HIGH

**Current**: Pure dense vector similarity only
```python
results = self.medquad_index.similarity_search_with_score(query, k=k)
```

**Requirement**: Hybrid BM25 + Dense (medical terms like "chest pain", "dyspnea" must be found even if embedding similarity is low)

**Questions**:
- Should I implement hybrid search using:
  1. **Option A**: LangChain's `BM25Retriever` + `EnsembleRetriever` (simpler)
  2. **Option B**: Custom Reciprocal Rank Fusion (RRF) to combine BM25 and FAISS scores (more control)
  3. **Option C**: Use DuckDB + FAISS hybrid search (more powerful but heavier)

- What weighting should BM25 vs Dense have? (e.g., 0.4 BM25 + 0.6 Dense, or 0.5/0.5?)

---

### 3. KNOWLEDGE GRAPH - RAG INTEGRATION MISSING ❌
**File**: `backend/app/core/kg.py` and `backend/app/core/rag.py`
**Severity**: HIGH

**Current State**:
- KG exists with DDXPlus data (conditions, symptoms, antecedents)
- RAG exists separately (MedQuAD + MedDialog vectors)
- **NO CONNECTION BETWEEN THEM**

**Requirement**: "KG as index" means:
1. Use KG for symptom-to-disease routing → "what should I ask next?"
2. Query KG to find candidate conditions
3. Retrieve RAG documents only for those conditions

**Current behavior** (wrong):
```python
# In node_next_question():
next_symptom = kg.rank_next_questions(present_symptoms, asked_symptoms)  # Good
rag_examples = rag.query_conversations("patient symptoms", k=2)  # BAD: generic query, not KG-aware
```

**Correct flow should be**:
```python
next_symptom = kg.rank_next_questions(present_symptoms, asked_symptoms)
# Find candidate conditions in KG that COULD explain present_symptoms
candidate_conditions = kg.get_candidate_conditions(present_symptoms)
# Retrieve MedQuAD docs for those conditions AND symptom question_type
rag_context = rag.query_medquad_filtered(
    query=f"How to recognize {next_symptom}?",
    focus_diseases=candidate_conditions,
    question_type="symptoms"
)
```

**Questions**:
- Should the RAG filtering work as:
  1. **Option A**: Retrieve all documents, then filter by KG-suggested diseases?
  2. **Option B**: Pre-filter by KG, then retrieve within that subset?
  3. **Option C**: Use KG for semantic rewriting of the query (e.g., "chest pain" → "thoracic pain", "angina")?

---

### 4. GRAPH-BASED RETRIEVAL FOR DDXPLUS - NOT IMPLEMENTED ❌
**File**: `backend/app/core/kg.py` 
**Severity**: HIGH (core feature missing)

**Requirement**: Multi-hop graph traversal for "what to ask next":
1. Seed graph with confirmed evidences (present_symptoms)
2. Traverse to candidate disease nodes
3. Rank by edge weight (information gain)
4. Surface unasked evidence nodes as next clarifying question

**Current implementation** (Lines 64-89 in kg.py):
```python
def rank_next_questions(self, present_symptoms, asked_symptoms):
    candidate_conditions = set()
    for symptom in present_symptoms:
        if self.graph.has_node(symptom):
            candidate_conditions.update(self.graph.predecessors(symptom))
    
    # Count frequency, rank by information gain
    symptom_counts = {}
    for condition in candidate_conditions:
        for neighbor in self.graph.successors(condition):
            if neighbor not in present_symptoms and neighbor not in asked_symptoms:
                symptom_counts[neighbor] = symptom_counts.get(neighbor, 0) + 1
    
    # Sort by distance from split point
    target_freq = len(candidate_conditions) / 2.0
    ranked = sorted(symptom_counts.items(), key=lambda x: abs(x[1] - target_freq))
    return ranked[0][0] if ranked else None
```

**Issues**:
1. **Information gain calculation is wrong**: Just counts symptom frequency, doesn't use conditional probabilities from DDXPlus release_evidences.json
2. **No edge weights**: DDXPlus provides P(evidence|disease) — this should weight the traversal
3. **No graph traversal optimization**: Could use multi-hop BFS to find deeper discriminative features

**Questions**:
- Should I:
  1. Load `release_evidences.json` to compute conditional probabilities P(evidence|disease)?
  2. Use Shannon entropy for true information gain: IG = H(current) - sum(H(split))?
  3. Implement k-hop BFS to find the most discriminative evidence within X hops?
  4. Cache the ranked evidences for performance (since graph is static)?

---

### 5. EMERGENCY DETECTION - ONLY KEYWORD LAYER IMPLEMENTED ⚠️
**File**: `backend/app/core/triage_graph.py` lines 24-33
**Severity**: HIGH (safety-critical)

**Current Implementation**:
```python
def evaluate_red_flags(symptoms: List[str], text: str) -> bool:
    text_lower = text.lower()
    if any(x in text_lower for x in ["loss of consciousness", "suicid", "kill myself", 
                                       "can't breathe", "bleeding heavily"]):
        return True
    
    if "E_55" in symptoms and ("shortness of breath" in text_lower or "jaw pain" in text_lower):
        return True
    return False
```

**Issues**:
1. **Only keyword layer exists** — missing severity escalation (layer 2) and KG-based severity (layer 3)
2. **Undersensitive**: Many red flag combos not covered (e.g., "severe chest pain + nausea + diaphoresis" = ACS)
3. **Oversensitive**: Catch-all keywords could over-flag (e.g., "I can't breathe properly" vs "I'm hyperventilating")

**Requirement**: Three-layer architecture with safe defaults:
- **Layer 1 (Keyword)**: Explicit keywords only, zero external dependencies
- **Layer 2 (Severity Escalation)**: If 3+ severe symptoms + duration, escalate
- **Layer 3 (KG-based)**: Query KG for condition severity, flag if ESI 1-2
- **Default**: Any layer failure defaults to HIGHER urgency, never lower

**Questions**:
- For Layer 2 (severity escalation), should I define:
  1. **Severe symptom list**: Which symptoms justify escalation? (chest pain, difficulty breathing, confusion, loss of consciousness, bleeding, suicidal ideation, etc.)
  2. **Threshold**: How many severe symptoms trigger escalation? (2, 3, or any combination?)
  3. **Temporal component**: Should duration matter? (e.g., "chest pain for 2 hours" vs "chest pain for 2 days"?)

- For Layer 3 (KG-based), should I:
  1. Map each condition in DDXPlus to ESI level (1=critical, 5=minimal)?
  2. If any candidate condition from KG is ESI 1-2, automatically flag as emergency?

---

### 6. NER MODEL - DUMMY IMPLEMENTATION ONLY ❌
**File**: `backend/app/core/triage_graph.py` lines 15-22
**Severity**: MEDIUM

**Current**:
```python
def dummy_ner(text: str) -> List[str]:
    symptoms = []
    text_lower = text.lower()
    if "chest pain" in text_lower: symptoms.append("E_55")
    if "headache" in text_lower: symptoms.append("E_53")
    if "fever" in text_lower: symptoms.append("E_91")
    return symptoms
```

**Issues**:
1. **Only 3 symptoms covered** — won't scale to DDXPlus's 1600+ symptoms
2. **Exact string match only** — misses variations ("chest ache", "pains in chest", "my chest hurts")
3. **No negation handling** — "no chest pain" matches as positive

**Options for real NER**:
1. **Option A**: Use SpaCy + d4data/biomedical-ner-all model (heavy, ~500MB)
2. **Option B**: Use BioBERT + token classification (medium, ~350MB)
3. **Option C**: Fuzzy matching + medical synonym expansion (lightweight, <1MB)
4. **Option D**: Load DDXPlus symptom list, use RapidFuzz for fuzzy matching (already have RapidFuzz dependency)

**Questions**:
- Which NER approach do you prefer? I'd recommend **Option D (RapidFuzz)** for now since:
  - You already have `rapidfuzz` imported elsewhere
  - DDXPlus provides official symptom IDs/names
  - Can add similarity threshold (e.g., only match if fuzzy score > 80)
  - Low latency, small model

- Should fallback responses for Ollama handle variations better? (e.g., templated slot-filling like "Do you experience pain, burning, or discomfort in your [BODY_PART]?")

---

### 7. OLLAMA FALLBACK - INCOMPLETE SLOT-FILLING ⚠️
**File**: `backend/app/core/triage_graph.py` lines 249-278, 342-381
**Severity**: MEDIUM

**Current** (incomplete):
```python
# node_next_question fallback:
state["messages"].append(f"QUESTION: Do you have symptom {next_symptom}?")

# node_explain fallback:
explanation = f"DIAGNOSIS_EXPLANATION: Based on your symptoms, this might be {state['final_diagnosis']}. Please consult..."
```

**Issues**:
1. **Generic slot-filling**: Doesn't vary question structure
2. **No context awareness**: Same phrasing for all symptoms
3. **Doesn't fill slots sequentially**: Should ask follow-ups like onset, duration, severity

**Requirement**: "Fill slots sequentially that fit any symptom"

**Suggested template structure**:
```python
SLOT_FILLING_TEMPLATES = {
    "onset": "When did this {symptom} start? (today, days ago, weeks ago?)",
    "duration": "How long does the {symptom} typically last?",
    "severity": "On a scale of 1-10, how severe is the {symptom}?",
    "characteristics": "How would you describe the {symptom}? (sharp, dull, burning?)",
    "frequency": "How often does the {symptom} happen? (constant, intermittent?)",
    "triggers": "Does anything make the {symptom} better or worse?",
}
```

**Questions**:
- Should the fallback questions follow this sequence: `onset → severity → duration → characteristics → triggers`?
- Or should slot order depend on the symptom type? (e.g., for "chest pain", ask severity first)
- Should I generate symptom-specific descriptions from the KG/RAG data?

---

### 8. XGBOOST MODEL PATH MISMATCH ❌
**File**: `backend/app/core/triage_graph.py` line 287, **vs** `backend/scripts/train_xgboost.py` line 82
**Severity**: HIGH

**Problem**:
- Training script saves as: `xgb_model.json`
- Inference expects: `xgb_model.pkl`

**Current inference code** (line 287):
```python
xgb_path = os.path.join(model_dir, "xgb_model.json")
import xgboost as xgb
clf = xgb.XGBClassifier()
clf.load_model(xgb_path)  # This works for .json
```

**But then tries to load encoders**:
```python
with open(os.path.join(model_dir, "mlb.pkl"), "rb") as f:  # This expects .pkl
    mlb = pickle.load(f)
```

**Fix needed**: Standardize on `.pkl` (pickle format) for all models so they're loadable together.

---

## SYSTEM PROMPTS REVIEW

### Current System Prompts (from triage_graph.py):

**1. node_next_question** (line 259-262):
```python
"You are a friendly, professional AI medical assistant. "
"Ask the user if they are experiencing a specific symptom. Keep it brief and conversational (1-2 sentences). "
"Do not give medical advice. Just ask the question."
```

**2. node_explain** (line 346-350):
```python
"You are a friendly, professional AI medical assistant. "
"Explain to the patient that based on their symptoms, they might have a specific condition. "
"Keep it empathetic and reassuring (2-3 sentences). "
"Always clarify that this is not a definitive medical diagnosis and they should consult the doctor."
```

**Issues**:
1. **Too generic** — doesn't leverage RAG context
2. **No medical grounding** — doesn't cite sources or explain reasoning
3. **Doesn't connect symptoms** — doesn't show the logical chain from symptoms → diagnosis

**Suggested improvements**:
```python
# Better system prompt for node_next_question with context:
f"""You are a medical triage AI. You are systematically gathering symptoms to narrow down 
possible diagnoses. The patient has already reported: {', '.join(present_symptoms)}.

Your job: Ask the next most discriminative symptom to differentiate between remaining 
candidate conditions: {', '.join(candidate_conditions)}.

Guidelines:
- Ask about ONE symptom at a time
- Use conversational language but be medically accurate
- Include context if helpful (e.g., "Given your chest discomfort...")
- Don't suggest diagnoses
- Offer reasonable options if it's a yes/no question"""
```

---

## MEDQUAD DATA STRUCTURE CLARIFICATION

**Questions about your MedQuAD dataset**:
1. **Format**: Is MedQuAD provided as JSON or XML?
2. **Structure**: Does each entry have:
   - `question` (string)
   - `answer` (string)
   - `focus_disease` (string)
   - `question_type` (enum: symptoms/treatment/prognosis/susceptibility/etc.)
   - `source` (string: GARD, NIDDK, etc.)
3. **Size**: How many QA pairs total? (typical MedQuAD has ~2.7M pairs)
4. **Long answers**: Are there answers >500 tokens? Should I split them by paragraph?

---

## SUMMARY OF IMPLEMENTATION TASKS

| # | Component | Current | Required | Priority | Effort |
|---|-----------|---------|----------|----------|--------|
| 1 | MedQuAD chunking & build script | ❌ Missing | QA-pair atomic + metadata | CRITICAL | 3-4hrs |
| 2 | Hybrid BM25 + Dense retrieval | ❌ Missing | Hybrid search | HIGH | 2hrs |
| 3 | KG-RAG integration | ❌ Missing | KG filters/guides RAG | HIGH | 2-3hrs |
| 4 | Graph-based DDXPlus retrieval | ❌ Missing | Multi-hop with IG ranking | HIGH | 3-4hrs |
| 5 | Three-layer emergency detection | ⚠️ Partial | Severity + KG escalation | HIGH | 2-3hrs |
| 6 | Real NER (RapidFuzz) | ❌ Dummy | Fuzzy symptom matching | MEDIUM | 1-2hrs |
| 7 | Ollama fallback slots | ⚠️ Generic | Sequential slot-filling | MEDIUM | 1-2hrs |
| 8 | XGBoost model format | ⚠️ Mismatch | Standardize on .pkl | MEDIUM | 0.5hr |
| 9 | System prompt improvement | ⚠️ Generic | RAG-grounded context | LOW | 1hr |

---

## NEXT STEPS

1. **Answer clarifying questions** (above) on:
   - Chunking strategy details (MedQuAD format, long answer handling)
   - Hybrid retrieval weighting
   - KG-RAG integration pattern
   - Graph-based retrieval depth
   - Emergency detection layers
   - NER approach
   - Fallback slot-filling order

2. **Provide data details**:
   - MedQuAD file format and structure
   - DDXPlus mapping to ESI levels
   - Any pre-existing taxonomy for symptom groups

3. **Priority**: Should I implement in order: [1, 2, 3, 4, 5, 6, 7, 8, 9] or different?

