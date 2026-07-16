# Final LangGraph Architecture Audit

**Status:** Multiple critical issues identified. All fixable.  
**Priority:** High - affects core retrieval quality, emergency detection, and slot-filling logic

---

## CRITICAL ISSUES

### 1. **RAG IMPLEMENTATION - MASSIVE GAPS**

#### Issue 1.1: No Hybrid BM25+Dense Retrieval
**File:** `backend/app/core/rag.py`  
**Current:** Pure dense vector search using FAISS  
**Required:** Hybrid retrieval with 0.3 BM25 + 0.7 Dense weighting  
**Impact:** Missing exact medical terminology matches (drug names, ICD codes)

```python
# CURRENTLY: Just dense search
results = index.search(query_vector, k=5)

# REQUIRED: Hybrid with BM25+Dense
bm25_scores = bm25.get_scores(query_terms)
dense_scores = index.search(query_vector, k=5)
hybrid_scores = 0.3 * normalize(bm25_scores) + 0.7 * normalize(dense_scores)
```

#### Issue 1.2: Wrong Embedding Model
**Current:** `all-MiniLM-L6-v2` (generic, 384-dim)  
**Required:** `MedCPT` or `PubMedBERT-derived` (medical domain)  
**Impact:** Poor medical term alignment, vocabulary gap between patient language and corpus

#### Issue 1.3: MedQuAD Chunking Broken
**Current:** Likely splitting by fixed token window (300 tokens)  
**Required:** Preserve QA pair atomicity, split long answers (500+ tokens) by paragraph with parent-child linking  
**Impact:** Breaking QA relationships, retrieving orphaned paragraphs without context

#### Issue 1.4: No Metadata Filtering
**Current:** Retrieving all chunks equally  
**Required:** Attach `question_type` (symptoms/treatment/prognosis), `focus_disease`, `source` metadata to each chunk  
**Usage:** Filter by question_type based on current intent (if extracting symptoms, only retrieve question_type="symptoms")  
**Impact:** 50%+ irrelevant retrievals from wrong context

#### Issue 1.5: MedDialog Not Integrated Correctly
**Current:** Disconnected in codebase  
**Required:** Chunk conversational turns (patient desc + doctor response) as atomic units with 1-turn overlap  
**Usage:** Few-shot example retrieval for clarifying questions, NOT factual grounding  
**Impact:** Not leveraging prior conversation patterns for slot-filling

#### Issue 1.6: DDXPlus Graph Retrieval Not Implemented
**Current:** Not used for retrieval  
**Required:** Multi-hop graph traversal to find next high-value clarifying questions  
**Usage:** 
  - Seed graph with confirmed patient evidences
  - Traverse to candidate disease nodes
  - Rank by edge weight (conditional probability)
  - Surface unasked evidence nodes with highest information gain
**Impact:** Clinical questions are generic/templated instead of intelligent

#### Issue 1.7: Embedding Dimension Mismatch Risk
**Current:** No startup validation of embedding dimension  
**Risk:** Mismatched dimension (384 vs 768) silently fails at inference  
**Required:** Store embedding_dim in index metadata, validate at startup

---

### 2. **KNOWLEDGE GRAPH INTEGRATION**

#### Issue 2.1: KG Not Integrated with RAG
**File:** `backend/app/core/kg.py`  
**Current:** KG built but used only for question prioritization  
**Required:** KG as index layer - query KG first to identify candidate conditions, then retrieve RAG docs for those conditions  
**Missing:**
  - No function to query KG by symptoms → candidate diseases
  - No function to retrieve evidence nodes for a disease
  - No information-gain calculation for next question

#### Issue 2.2: Missing Graph Traversal for Next Questions
**Required:** Multi-hop traversal algorithm  
```python
def next_question_from_kg(confirmed_symptoms, candidate_diseases):
    """Find high-value next question via graph traversal"""
    # Seed graph with confirmed symptoms
    # Traverse to disease nodes
    # For each disease, calculate information gain of unasked evidences
    # Return evidence with highest gain
```

#### Issue 2.3: Condition Card Generation Not Done
**Required:** Generate templated text blurbs from DDXPlus structured data for each disease  
```python
# Example output:
# "Based on your symptoms of chest pain and shortness of breath, 
#  this could be related to acute coronary syndrome (ACS), 
#  pulmonary embolism, or pneumothorax. A doctor needs to examine you urgently."
```

---

### 3. **EMERGENCY DETECTION - OVERSENSITIVE**

#### Issue 3.1: Emergency Keywords Too Aggressive
**File:** `backend/app/core/triage_graph.py` lines 26-28  
**Current:** Triggers on "suicid", "kill myself" - too broad  
**Required:** Conservative list only:
```python
CRITICAL_KEYWORDS = {
    "chest pain",
    "difficulty breathing", "can't breathe", "shortness of breath",
    "loss of consciousness", "unconscious", "passed out",
    "heavy bleeding", "hemorrhage", "severe bleeding",
    "suicidal thoughts", "suicide"  # More specific than "suicid"
}
```

#### Issue 3.2: Combination Trigger Logic
**Current:** Line 31 hardcodes chest pain + SOB  
**Correct:** Should work but too simplistic. Need:
  - Multi-layer safety net:
    1. **Layer 1 (floor):** Keyword matching (cannot be bypassed)
    2. **Layer 2 (enrichment):** Severity escalation (3+ severe symptoms → emergency)
    3. **Layer 3 (deepest):** KG severity lookup
  - **Safety rule:** Each layer can ONLY raise urgency, never lower it
  - **Fallback:** On timeout/error, default to higher urgency tier, not lower

#### Issue 3.3: evaluate_red_flags Has Typo Risk
**Current:** Line 27 uses `"suicid"` (prefix match only)  
**Better:** Exact terms like `"suicidal"` or regex

---

### 4. **NER - DUMMY IMPLEMENTATION**

#### Issue 4.1: Dummy NER Too Simplistic
**Current:** Lines 15-22 hardcoded string matching (chest pain → E_55, etc.)  
**Required:** 
  - Use actual biomedical NER model (`d4data/biomedical-ner-all` or spaCy biomedical)
  - Extract entities with proper confidence scores
  - Handle abbreviations, synonyms

**For now, improved dummy NER:**
```python
def dummy_ner(text: str) -> List[str]:
    """Improved dummy NER with more comprehensive symptom mapping"""
    symptoms = []
    text_lower = text.lower()
    
    # Use exact/case-insensitive phrase matching instead of substring
    symptom_map = {
        ("chest pain", "chest discomfort", "heart pain"): "E_55",
        ("headache", "migraine", "head pain"): "E_53",
        ("fever", "high temperature", "elevated temp"): "E_91",
        ("shortness of breath", "dyspnea", "can't breathe"): "E_38",
        ("nausea", "feeling sick"): "E_77",
        ("dizziness", "vertigo", "lightheaded"): "E_42",
    }
    
    for phrases, code in symptom_map.items():
        if any(phrase in text_lower for phrase in phrases):
            symptoms.append(code)
    
    return symptoms
```

#### Issue 4.2: No Confidence Scoring
**Current:** Symptoms returned as flat list  
**Required:** Return list of tuples: `[(symptom, confidence), ...]`  
**Usage:** Filter low-confidence extractions

---

### 5. **FALLBACK QUESTION LOGIC - NOT SLOT-FILLING SEQUENTIAL**

#### Issue 5.1: node_next_question Missing Sequential Logic
**File:** `backend/app/core/triage_graph.py` lines 249-277  
**Current:** Generates LLM-based question, but Ollama fallback is generic  
**Required:** Hardcoded templated questions that fill slots sequentially

```python
# REQUIRED: Fallback questions that progress through slots
FALLBACK_QUESTIONS = [
    "Do you have any additional symptoms like fever, fatigue, or nausea?",
    "When did these symptoms start? Was it sudden or gradual?",
    "Have you had any recent injuries, falls, or trauma?",
    "Do you have any pre-existing medical conditions (diabetes, hypertension, asthma)?",
    "Are you currently taking any medications?",
    "Have you traveled recently or been exposed to sick people?",
    "Do you have any allergies to medications?",
]

# Cycle through these sequentially when Ollama fails
question_index = state.get("fallback_question_index", 0) % len(FALLBACK_QUESTIONS)
state["messages"].append(f"QUESTION: {FALLBACK_QUESTIONS[question_index]}")
state["fallback_question_index"] = question_index + 1
```

#### Issue 5.2: node_explain Fallback Template Missing
**Current:** Generic explanation  
**Required:** Templated response with slot information
```python
explanation = (
    f"Based on your symptoms of {', '.join(state['present_symptoms'])}, "
    f"this could be related to {state['final_diagnosis']}. "
    f"Please consult with a {state['department']} specialist for proper diagnosis. "
    f"In the meantime, avoid triggers and monitor for changes."
)
```

---

### 6. **SYSTEM PROMPTS - NEED REVIEW**

#### Issue 6.1: node_next_question System Prompt
**Current:** Lines ~270-273  
**Review needed:**
- Does it emphasize medical accuracy?
- Does it avoid suggesting dangerous advice?
- Is it patient-friendly?

#### Issue 6.2: node_explain System Prompt
**Current:** Lines ~356-360  
**Review needed:**
- Does it properly disclaimize ("not a diagnosis")?
- Does it emphasize seeing a doctor?
- Does it cite sources from RAG retrieval?

---

### 7. **GRAPH ARCHITECTURE QUESTIONS**

#### Question 7.1: Route Entry Flow
**File:** Lines 502-520  
**Issue:** After `node_extract_symptoms`, does it correctly route to clinical loop or classification?

#### Question 7.2: Fallback Loop Termination
**Issue:** If Ollama fails repeatedly, should there be a max-attempt limit before moving to classification?

---

## SUMMARY TABLE

| Component | Status | Priority | Effort |
|-----------|--------|----------|---------|
| RAG Hybrid Retrieval | Missing | CRITICAL | 2-3 hours |
| RAG Medical Embeddings | Wrong model | HIGH | 1 hour |
| MedQuAD Chunking | Broken | HIGH | 2 hours |
| MedQuAD Metadata | Missing | HIGH | 1 hour |
| MedDialog Integration | Incomplete | MEDIUM | 1 hour |
| DDXPlus Graph Retrieval | Not implemented | CRITICAL | 3-4 hours |
| KG-RAG Integration | Disconnected | CRITICAL | 3 hours |
| Emergency Detection | Oversensitive | MEDIUM | 1 hour |
| NER Model | Dummy | MEDIUM | 1 hour (improve dummy) |
| Fallback Questions | Not sequential | HIGH | 1 hour |
| System Prompts | Not reviewed | MEDIUM | 0.5 hour |
| Graph Edges | Unclear | LOW | 0.5 hour |

**Total Effort:** 17-20 hours

---

## RECOMMENDED FIX ORDER

1. **Week 1 (Foundation):**
   - Fix emergency keywords (1 hour)
   - Improve dummy NER (1 hour)
   - Fix MedQuAD chunking (2 hours)
   - Add metadata filtering (1 hour)

2. **Week 2 (Retrieval):**
   - Switch embedding model to MedCPT (1 hour)
   - Implement hybrid BM25+Dense (2-3 hours)
   - Add startup validation (0.5 hour)

3. **Week 3 (Graph Integration):**
   - Build KG query functions (2 hours)
   - Implement graph traversal for next-question (3 hours)
   - Generate condition cards from DDXPlus (2 hours)

4. **Week 4 (Polish):**
   - Add fallback sequential questions (1 hour)
   - Review/improve system prompts (1 hour)
   - Integration testing (2 hours)

