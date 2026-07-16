# Quick Reference: New Architecture Modules

## 1. Hybrid RAG with Medical Embeddings
```python
from app.core.rag_hybrid import get_rag_engine

rag = get_rag_engine()

# Query by symptoms (patient language)
results = rag.query_medquad_by_symptoms("chest pain radiating to arm", k=5)
# Returns: [{"text": "...", "metadata": {...}, "score": 0.85, ...}]

# Query by condition (clinical terms)
results = rag.query_medquad_by_condition("myocardial infarction", k=3)

# Generic hybrid query with filters
results = rag.query_medquad_hybrid(
    query="chest pain",
    k=5,
    question_type_filter='symptoms',  # Only symptom Q&As
    bm25_weight=0.3,
    dense_weight=0.7
)
```

## 2. Conservative Emergency Detection
```python
from app.core.emergency_detection import EmergencyDetector, UrgencyLevel

urgency, details = EmergencyDetector.detect_emergency("chest pain and difficulty breathing")
# Returns: (UrgencyLevel.EMERGENT, {detection_details})

if urgency.value <= 2:  # EMERGENT or URGENT
    route_to_emergency()

# Safe colors for UI
color = EmergencyDetector.get_urgency_color(urgency)  # red/orange/yellow/green
```

## 3. Fallback When Ollama Unavailable
```python
from app.core.fallback_paths import get_fallback_manager

fallback = get_fallback_manager()

# Get next templated question (works for ANY symptom)
next_question = fallback.get_next_question()
# "What symptoms are you experiencing?"

# Process user response
result = fallback.process_user_response("chest pain")
# result = {
#   'filled_slot': FallbackSlot.SYMPTOMS,
#   'next_question': 'On a scale 1-10, how severe?',
#   'all_slots_filled': False
# }

# When all slots filled
if result['all_slots_filled']:
    diagnosis = fallback.generate_diagnosis()
    # {category: 'cardiac', department: 'Cardiology', ...}
```

## 4. Symptom Extraction with NER
```python
from app.core.ner_symptom_extractor import SymptomsExtractor, get_system_prompt

# Extract symptoms
symptoms = SymptomsExtractor.extract_symptoms("severe chest pain with dyspnea")
# [{symptom: 'chest_pain', severity: SEVERE, location: 'chest'}, ...]

# Get summary
summary = SymptomsExtractor.get_symptom_summary(symptoms)
# "chest_pain (severe), dyspnea (moderate)"

# System prompts for LLM nodes
prompt = get_system_prompt('triage')  # 'question_generation', 'explanation', etc.
```

## 5. KG-RAG Integration
```python
from app.core.kg_rag_integration import create_kg_rag_integration, TriagePhase

kg_rag = create_kg_rag_integration(kg_instance, rag_instance)

# Get next question with medical context
q_data = kg_rag.get_next_clarifying_question(
    confirmed_symptoms=['chest pain', 'dyspnea'],
    phase=TriagePhase.REFINEMENT
)
# {
#   'question': 'Do you have fever?',
#   'info_gain': 0.6,
#   'grounding_docs': [medical excerpts from RAG]
# }

# Generate verification card with grounding
card = kg_rag.generate_condition_card(
    condition='Acute Myocardial Infarction',
    confirmed_symptoms=['chest pain', 'dyspnea']
)
# {condition, description, typical_presentation, grounding_sources}

# Verify routing decision
verification = kg_rag.verify_routing_decision(
    condition='AMI',
    confirmed_symptoms=['chest pain', 'dyspnea', 'diaphoresis']
)
# {condition, confidence, recommendation, grounding_documents}
```

## Integration into Triage Graph

### In node_emergency_check
```python
from app.core.emergency_detection import EmergencyDetector

def node_emergency_check(state: TriageState) -> TriageState:
    msg = state['messages'][-1] if state['messages'] else ""
    urgency, details = EmergencyDetector.detect_emergency(msg)
    state['triage_level'] = urgency.value
    state['is_emergency'] = urgency.value <= 2
    state['messages'].append(f"Urgency: {urgency.name}")
    return state
```

### In node_extract_symptoms
```python
from app.core.ner_symptom_extractor import SymptomsExtractor

def node_extract_symptoms(state: TriageState) -> TriageState:
    msg = state['messages'][-1] if state['messages'] else ""
    symptoms = SymptomsExtractor.extract_symptoms(msg)
    state['present_symptoms'] = [s['symptom'] for s in symptoms]
    state['symptom_details'] = symptoms
    return state
```

### In node_next_question (with LLM + Fallback)
```python
def node_next_question(state: TriageState) -> TriageState:
    rag = get_rag_engine()
    kg_rag = create_kg_rag_integration(get_kg(), rag)
    
    try:
        # Try to use LLM with RAG grounding
        q_data = kg_rag.get_next_clarifying_question(
            confirmed_symptoms=state['present_symptoms'],
            phase=TriagePhase.REFINEMENT
        )
        question = q_data['question']
    except:
        # Fallback when LLM unavailable
        fallback = get_fallback_manager()
        question = fallback.get_next_question()
    
    state['messages'].append(f"QUESTION: {question}")
    return state
```

## Before You Start

1. **MedQuAD Data**: Need CSV at `backend/data/medquad.csv`
   - Columns: `question, answer, source, focus_area`
   - Run: `python backend/scripts/build_medquad_index.py`
   - Verify: 1000+ vectors indexed

2. **Dependencies**: Added `rank-bm25` to requirements.txt
   - Run: `pip install rank-bm25`

3. **XGBoost Model Fix**: `train_xgboost.py` saves as JSON, but nodes expect pickle
   - Change: `clf.save_model()` → `pickle.dump(clf, f)`

## Testing Quick Commands

```bash
# Test emergency detection
python -c "
from backend.app.core.emergency_detection import EmergencyDetector
print(EmergencyDetector.detect_emergency('chest pain')[0].name)  # EMERGENT
print(EmergencyDetector.detect_emergency('mild cough')[0].name)   # NON_URGENT
"

# Test NER
python -c "
from backend.app.core.ner_symptom_extractor import SymptomsExtractor
symptoms = SymptomsExtractor.extract_symptoms('severe chest pain')
print(SymptomsExtractor.get_symptom_summary(symptoms))
"

# Test fallback
python -c "
from backend.app.core.fallback_paths import get_fallback_manager
mgr = get_fallback_manager()
print(mgr.get_next_question())
mgr.process_user_response('chest pain')
print(mgr.get_next_question())
"
```

## Key Improvements Over Previous

| Component | Before | After |
|-----------|--------|-------|
| Embeddings | Generic (384-dim) | Medical (768-dim PubMedBERT) |
| Retrieval | Dense only | Hybrid BM25+Dense (0.3/0.7) |
| MedQuAD Chunks | Fixed 300-char windows | Atomic QA + paragraph splits |
| Emergency Detection | Keyword-only, single pass | 4-layer conservative with escalation safety |
| LLM Failure | System crashes | Graceful fallback with conversation continuation |
| NER | Dummy keyword list | Pattern-based with severity + location |
| KG-RAG | Disconnected | Full integration with graph traversal |

