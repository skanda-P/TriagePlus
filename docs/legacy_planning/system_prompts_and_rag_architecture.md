# TriagePlus — System Prompts & RAG Architecture (Final Design)

## Table of Contents
1. [First-Principles RAG Analysis](#1-first-principles-rag-analysis)
2. [Embedding Model Decision](#2-embedding-model-decision)
3. [Index Architecture](#3-index-architecture)
4. [Chunking Strategy](#4-chunking-strategy)
5. [Knowledge Graph Design](#5-knowledge-graph-design)
6. [Reranking Analysis](#6-reranking-analysis)
7. [System Prompts](#7-system-prompts)
8. [Dataset Utilization Map](#8-dataset-utilization-map)

---

## 1. First-Principles RAG Analysis

Before deciding on indices, models, or chunking — **where does this system actually need external knowledge?**

### Pipeline Stages & Their Knowledge Needs

| Stage | What It Does | Needs Retrieved Knowledge? | Why / Why Not |
|---|---|---|---|
| Emergency detection | Catches life-threatening messages | ❌ **No** | Deterministic keyword match + LLM yes/no classification. No external context needed — the question is "is this text an emergency?", not "what disease is this?" |
| Demographics collection | Name, age, gender, phone | ❌ **No** | Simple form-filling, no medical reasoning |
| Slot extraction | Parse symptoms from conversation | ❌ **No** | The LLM extracts from what the patient *said*, not from external sources. Injecting RAG here is how contamination happens. |
| **Next-question selection** | Decide what to ask next | ✅ **Yes — structured lookup** | Needs to know "given symptom X, what other evidences differentiate the possible pathologies?" This is the **knowledge graph** (DDXPlus), not vector retrieval. |
| **Follow-up question phrasing** | Generate the actual question text | ⚠️ **Debatable** | The LLM can phrase questions from DDXPlus evidence descriptions + system prompt. RAG-sourced conversation examples add contamination risk for marginal phrasing quality. **Decision: No RAG here.** |
| **Classification explanation** | Explain why department X was chosen | ✅ **Yes — vector retrieval** | After XGBoost decides the department, the LLM needs medical facts to explain *why*. This is the one place vector retrieval over MedQuAD genuinely helps. |
| Patient brief | Summarize for doctor portal | ❌ **No** | Summarizes data already collected during the session. No external knowledge needed. |

### Conclusion: We Need Exactly 2 Knowledge Sources

```
┌─────────────────────────────────────────────────────────┐
│ Knowledge Source 1: Knowledge Graph (NetworkX/DDXPlus)  │
│ Used by: next_question node                              │
│ Type: Structured graph lookup, NOT vector similarity     │
│ Content: 49 pathologies → 223 evidences → questions      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Knowledge Source 2: Clinical Facts Index (FAISS/MedQuAD)│
│ Used by: explanation node (ONLY)                         │
│ Type: Vector similarity search                           │
│ Content: MedQuAD Q&A chunks (~16,412 entries)            │
└─────────────────────────────────────────────────────────┘
```

**What's NOT an index:**
- ❌ ~~Phrasing templates~~ (contamination risk > phrasing benefit)
- ❌ ~~Symptom2Disease in FAISS~~ (moves to XGBoost training data)
- ❌ ~~MedDialog in FAISS~~ (too noisy for retrieval; useful only as classifier augmentation)
- ❌ ~~Synthetic conversations in FAISS~~ (severely imbalanced, contamination source)

---

## 2. Embedding Model Decision

### The Actual Constraint Set

The embedding model is used in **one place**: querying the MedQuAD clinical facts index during the explanation step. This means:
- It runs **once per completed triage** (not per message turn)
- Latency budget is generous (~200-500ms is fine, it's after the department decision)
- It must handle medical vocabulary well
- It should be simple to integrate with LangChain/FAISS

### Candidate Comparison

| Model | Dim | Max Tokens | Domain | Load Method | Memory | Query Speed (CPU) | Medical Vocab | LangChain Integration |
|---|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 256 | General | `SentenceTransformer()` | ~90MB | ~5ms | ⚠️ Poor | `HuggingFaceEmbeddings` |
| `NeuML/pubmedbert-base-embeddings` | 768 | 512 | Biomedical | `SentenceTransformer()` | ~420MB | ~15ms | ✅ Good | `HuggingFaceEmbeddings` |
| `nomic-embed-text` (Ollama) | 768 | **8192** | General+code | `ollama.embed()` | Shared w/ Ollama | ~20ms | ⚠️ Decent | `OllamaEmbeddings` |
| MedCPT dual-encoder | 768 | 512 | Biomedical | Custom `AutoModel` x2 | ~800MB | ~20ms | ✅ Best | ❌ Custom wrapper needed |
| `BAAI/bge-small-en-v1.5` | 384 | 512 | General | `SentenceTransformer()` | ~130MB | ~6ms | ⚠️ Decent | `HuggingFaceEmbeddings` |

### Recommendation: `NeuML/pubmedbert-base-embeddings`

**Reasoning:**
1. **Biomedical vocabulary is critical** — MedQuAD content is full of clinical terms (e.g., "anterior chamber", "optic nerve", "tonometry"). General models tokenize these into meaningless subwords.
2. **512-token limit** — MedQuAD answer chunks average 200-400 tokens. MiniLM's 256-token limit would silently truncate ~30% of chunks.
3. **Drop-in LangChain integration** — one line: `HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")`
4. **Single model** — same model for both indexing and querying (unlike MedCPT's dual-encoder complexity)
5. **~420MB memory** — acceptable alongside Ollama's ~4GB for llama3.2

> [!NOTE]
> **Why not Ollama embeddings?** `nomic-embed-text` via Ollama would simplify the stack (no separate `sentence-transformers` dependency). However, it's a general-domain model, and medical vocabulary handling matters here. If you want to minimize dependencies, `nomic-embed-text` is a viable fallback — the 8192-token context is a genuine advantage for long MedQuAD answers. But PubMedBERT's biomedical pretraining gives meaningfully better retrieval quality on medical text.

---

## 3. Index Architecture

### Index 1: Clinical Facts (FAISS + MedQuAD)

**Purpose:** Provide supporting medical facts for the explanation LLM node.

**Source data:** MedQuAD (`medquad.jsonl`) — 16,412 entries across 5,127 focus areas

**What goes in:**
```
For each MedQuAD entry:
  1. The question text → embedded as a standalone chunk (patients phrase things as questions)
  2. The answer text → split into structural chunks (see §4)
  
Each chunk carries metadata:
  - focus_area: "Glaucoma", "Pneumonia", etc.
  - source_type: "question" | "answer_definition" | "answer_symptoms" | "answer_causes" | "answer_treatment"
  - parent_question: the original question text
```

**What does NOT go in:**
- Symptom2Disease (training data for XGBoost, not retrieval content)
- MedDialog (too noisy — short doctor responses with links, not medical facts)
- Synthetic conversations (contamination source)

**Retrieval at runtime:**
```python
# In the explanation node ONLY — after XGBoost has decided the department
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3}
)

# Filter results to match the predicted department's medical domain
# (optional post-retrieval filter using focus_area metadata)
```

### Knowledge Source 2: DDXPlus Knowledge Graph (NetworkX)

Not a FAISS index. Detailed in §5.

---

## 4. Chunking Strategy

### The Problem with Current Chunking

Looking at actual MedQuAD data, the answers have clear internal structure:

```
"What is (are) Glaucoma?" →
  - Paragraph 1: Definition of glaucoma (what it is)
  - Paragraph 2: How it develops (mechanism)
  - Paragraph 3: Open-angle glaucoma (subtype)
  - Paragraph 4: Prognosis and treatment overview
```

Naive fixed-token splitting with overlap will create chunks that blend "what is X" with "how X develops" — degrading retrieval precision.

### Revised Chunking: Structure-Aware + Parent Reference

```python
import re
from langchain.schema import Document

def chunk_medquad_entry(question: str, answer: str, focus_area: str) -> list[Document]:
    """
    Structure-aware chunking for MedQuAD entries.
    
    Strategy:
    1. Embed the question as its own chunk (query-shaped)
    2. Split the answer on structural boundaries (paragraphs, section headers)
    3. Sub-split any oversized sections at sentence boundaries
    4. Every chunk carries parent metadata for context expansion
    """
    docs = []
    
    # --- Chunk type 1: The question itself ---
    docs.append(Document(
        page_content=question.strip(),
        metadata={
            "source": "medquad",
            "chunk_type": "question",
            "focus_area": focus_area,
            "parent_question": question.strip(),
        }
    ))
    
    if not answer or not answer.strip():
        return docs
    
    # --- Chunk type 2: Answer sections ---
    # Split on double newlines, or section-like headers
    sections = re.split(r'\n\s*\n|\n(?=[A-Z][^.]{5,50}\s*\n)', answer.strip())
    sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 30]
    
    # If no structural splits found, treat entire answer as one section
    if not sections:
        sections = [answer.strip()]
    
    for i, section in enumerate(sections):
        word_count = len(section.split())
        
        if word_count <= 400:
            # Section fits within token budget — keep as-is
            docs.append(Document(
                page_content=section,
                metadata={
                    "source": "medquad",
                    "chunk_type": "answer_section",
                    "focus_area": focus_area,
                    "parent_question": question.strip(),
                    "section_index": i,
                    "parent_answer_preview": answer[:300],  # for context expansion
                }
            ))
        else:
            # Oversized section — sub-split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = []
            current_words = 0
            
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_words + s_words > 350 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": "medquad",
                            "chunk_type": "answer_section",
                            "focus_area": focus_area,
                            "parent_question": question.strip(),
                            "section_index": i,
                            "parent_answer_preview": answer[:300],
                        }
                    ))
                    # Overlap: keep last sentence
                    current_chunk = [current_chunk[-1], sentence] if current_chunk else [sentence]
                    current_words = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_words += s_words
            
            if current_chunk:
                docs.append(Document(
                    page_content=' '.join(current_chunk),
                    metadata={
                        "source": "medquad",
                        "chunk_type": "answer_section",
                        "focus_area": focus_area,
                        "parent_question": question.strip(),
                        "section_index": i,
                        "parent_answer_preview": answer[:300],
                    }
                ))
    
    return docs
```

### Build-Time Validation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")

def validate_all_chunks(documents: list[Document], max_tokens: int = 512):
    violations = []
    for i, doc in enumerate(documents):
        token_count = len(tokenizer.encode(doc.page_content))
        if token_count > max_tokens:
            violations.append((i, token_count, doc.page_content[:80]))
    
    if violations:
        for idx, tc, preview in violations:
            print(f"VIOLATION: chunk {idx} has {tc} tokens: {preview}...")
        raise AssertionError(
            f"{len(violations)} chunks exceed {max_tokens} token limit. "
            f"Fix chunking before building the index."
        )
    print(f"✓ All {len(documents)} chunks within {max_tokens} token limit")
```

---

## 5. Knowledge Graph Design

### Data Shape (from DDXPlus analysis)

```
49 pathologies
223 evidences:
  - 110 symptoms (things the patient experiences now)
  - 113 antecedents (medical history, risk factors)
  
Evidence data types:
  - 208 binary (yes/no) — e.g., "Do you have a fever?"
  - 10 categorical (single-select) — e.g., "How fast did the pain appear?" (0-10 scale)
  - 5 multi-choice (multi-select) — e.g., "Where is the pain?" (190+ body locations)

14 evidence groups (code_question) — related sub-questions:
  - E_53 group: pain → location (E_55), character (E_54), intensity (E_56), 
    radiation (E_57), onset speed (E_59), precision (E_58)
  - E_129 group: skin lesions → location (E_133), color (E_130), pain (E_134),
    swelling (E_132), itching (E_136), size (E_135), peeling (E_131)
```

### Graph Structure

```python
import networkx as nx
import json
import pickle
from pathlib import Path

def build_knowledge_graph(ddxplus_dir: Path) -> nx.DiGraph:
    with open(ddxplus_dir / "release_conditions.json") as f:
        conditions = json.load(f)
    with open(ddxplus_dir / "release_evidences.json") as f:
        evidences = json.load(f)
    
    G = nx.DiGraph()
    
    # --- Evidence nodes ---
    for eid, edata in evidences.items():
        G.add_node(eid,
            node_type="evidence",
            name=edata["name"],
            question_en=edata["question_en"],
            is_antecedent=edata["is_antecedent"],
            data_type=edata["data_type"],
            code_question=edata["code_question"],
            possible_values=edata.get("possible-values", []),
            value_meanings={
                k: v.get("en", "")
                for k, v in edata.get("value_meaning", {}).items()
            }
        )
    
    # --- Pathology nodes + edges ---
    for pname, pdata in conditions.items():
        G.add_node(pname,
            node_type="pathology",
            icd10=pdata.get("icd10-id", ""),
            severity=pdata.get("severity", 5),
            name_en=pdata.get("cond-name-eng", pname)
        )
        for eid in pdata.get("symptoms", {}):
            if eid in evidences:
                G.add_edge(pname, eid, relation="has_symptom")
                G.add_edge(eid, pname, relation="symptom_of")
        for eid in pdata.get("antecedents", {}):
            if eid in evidences:
                G.add_edge(pname, eid, relation="has_antecedent")
                G.add_edge(eid, pname, relation="antecedent_of")
    
    # --- Sub-question grouping ---
    for eid, edata in evidences.items():
        parent = edata["code_question"]
        if parent != eid and parent in evidences:
            G.add_edge(parent, eid, relation="has_sub_question")
    
    return G
```

### Runtime Question Selection

```python
def get_next_questions(
    graph: nx.DiGraph,
    confirmed: set[str],    # evidence IDs patient confirmed
    denied: set[str],       # evidence IDs patient denied
    max_questions: int = 2
) -> list[dict]:
    """
    Information-gain-based question selection.
    
    Given collected evidence, find which unasked evidences best
    differentiate between still-possible pathologies.
    """
    already_asked = confirmed | denied
    
    # 1. Find candidate pathologies 
    #    (share at least one confirmed symptom with the patient)
    candidates = set()
    for eid in confirmed:
        if eid in graph:
            for _, target, data in graph.out_edges(eid, data=True):
                if data.get("relation") in ("symptom_of", "antecedent_of"):
                    candidates.add(target)
    
    if not candidates:
        # Cold start — return broad screening questions
        broad = [
            eid for eid, d in graph.nodes(data=True)
            if d.get("node_type") == "evidence"
            and not d.get("is_antecedent")
            and d.get("data_type") == "B"
            and d.get("code_question") == eid  # only top-level questions
            and eid not in already_asked
        ]
        # Sort by number of pathologies that have this symptom (most common first)
        broad.sort(key=lambda e: graph.in_degree(e), reverse=True)
        return [{
            "evidence_id": eid,
            "question": graph.nodes[eid]["question_en"],
            "data_type": graph.nodes[eid]["data_type"],
        } for eid in broad[:max_questions]]
    
    # 2. Eliminate pathologies contradicted by denied evidence
    for eid in denied:
        to_remove = set()
        for _, target, data in graph.out_edges(eid, data=True):
            if data.get("relation") in ("symptom_of", "antecedent_of"):
                # If this was a required symptom and patient denied it,
                # reduce (but don't eliminate) confidence in that pathology
                to_remove.add(target)
        # Don't hard-eliminate — just deprioritize
    
    # 3. Find unasked evidences that differentiate remaining candidates
    scored = []
    for eid, ndata in graph.nodes(data=True):
        if ndata.get("node_type") != "evidence":
            continue
        if eid in already_asked:
            continue
        
        # Only ask top-level questions (sub-questions follow automatically)
        if ndata.get("code_question") != eid:
            parent_q = ndata["code_question"]
            if parent_q not in confirmed:
                continue  # don't ask sub-question until parent is confirmed
        
        # Count candidates that have vs. don't have this evidence
        has_it = sum(1 for p in candidates if graph.has_edge(p, eid))
        if has_it == 0:
            continue  # no candidate has this — useless to ask
        
        hasnt_it = len(candidates) - has_it
        info_gain = min(has_it, hasnt_it)  # maximum when 50/50 split
        
        # Prioritize symptoms over antecedents (more actionable early)
        priority = 0 if not ndata.get("is_antecedent") else 1
        
        scored.append({
            "evidence_id": eid,
            "question": ndata["question_en"],
            "data_type": ndata["data_type"],
            "info_gain": info_gain,
            "is_antecedent": ndata.get("is_antecedent", False),
            "_sort_key": (-info_gain, priority),
        })
    
    scored.sort(key=lambda x: x["_sort_key"])
    return scored[:max_questions]
```

### Mapping Evidence to LLM Slots

The DDXPlus evidence questions replace the hardcoded `REQUIRED_SLOTS`. Instead of always asking about chief_complaint/duration/severity/location/associated_symptoms, the system asks **what the knowledge graph says is most informative** given the patient's specific symptoms.

---

## 6. Reranking Analysis

### Do We Need Reranking?

**Context:** Reranking means retrieving a larger candidate set (e.g., top-10) with a fast bi-encoder, then re-scoring those candidates with a slower but more accurate cross-encoder to pick the final top-3.

**Where retrieval happens in our pipeline:** Only in the explanation node, *after* the department decision is already made by XGBoost.

**What happens if retrieval is slightly noisy:**
- The explanation LLM receives some irrelevant MedQuAD chunks
- The explanation might be slightly less focused
- The **department decision is NOT affected** (XGBoost doesn't see RAG chunks)
- The **urgency score is NOT affected** (XGBoost output)

### Decision: Skip Reranking for MVP

**Reasoning:**
1. RAG quality here affects only explanation prose, not safety-critical decisions
2. A cross-encoder adds ~100-200ms latency per query
3. The bi-encoder + score threshold filtering (drop chunks below 0.3 similarity) already handles most noise
4. Adding a third model (cross-encoder) increases memory and complexity

**Post-MVP upgrade path:** If explanation quality is measurably poor, add `cross-encoder/ms-marco-MiniLM-L-6-v2` as a reranker:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)
```

---

## 7. System Prompts

### Design Principles

Every prompt follows these rules:
1. **Explicit role boundary** — tell the model exactly what it CAN and CANNOT do
2. **Output format specification** — JSON where structured output is needed, with exact schema
3. **Anti-hallucination guardrails** — explicit "if you don't know, say so" instructions
4. **No leaked context** — RAG content framed as external reference, never as "your knowledge"
5. **Temperature tuned per use case** — 0.0 for classification, 0.2-0.3 for generation

---

### Prompt 1: Emergency Detection

**Used by:** `emergency_check` LangGraph node
**Called:** Every message during symptom phases (`INITIAL_SYMPTOM`, `CONVERSATION`)
**Model:** Ollama `llama3.2` (only if keyword check doesn't trigger first)

```
SYSTEM:
You are a medical emergency screening function. Your ONLY job is to determine
if the patient's message describes an ACTIVE, LIFE-THREATENING emergency
that requires immediate emergency services (ambulance/ER).

Rules:
- Return {"is_emergency": true} ONLY for situations where delay could cause
  death or permanent harm: heart attack, stroke, severe bleeding, inability
  to breathe, loss of consciousness, anaphylaxis, suicidal intent, overdose.
- Return {"is_emergency": false} for everything else, including:
  - Past events ("I had chest pain last week")
  - Negative statements ("I do NOT have chest pain")
  - Chronic conditions ("I've had headaches for months")
  - Mild/moderate symptoms ("I have a fever", "my stomach hurts")
- When in doubt, return false. The keyword safety net handles obvious cases
  independently.

You must respond with ONLY a valid JSON object. No explanation, no prose.

USER:
Patient message: "{message}"

Output JSON:
```

**Settings:**
- Temperature: `0.0`
- `num_predict`: `64`
- Format: `json`
- Timeout: `5 seconds`
- Fallback on error: `False` (safe — keyword check already ran)

---

### Prompt 2: Slot Extraction

**Used by:** `symptom_collection` LangGraph node
**Called:** Each symptom-phase turn
**Model:** Ollama `llama3.2`

```
SYSTEM:
You are a medical intake data extraction system. Your job is to extract
structured symptom information from a patient-doctor conversation.

Given the current state of collected information and the latest exchange,
update ONLY the fields where the new turn provides NEW information.

Rules:
- Keep existing values unchanged unless the patient explicitly contradicts them.
- Use null for fields not yet mentioned by the patient.
- For associated_symptoms, APPEND new symptoms to the existing list. Do not
  remove previously mentioned symptoms unless the patient retracts them.
- Extract ONLY what the patient explicitly stated. Do NOT infer symptoms
  they didn't mention.
- "unknown", "not mentioned", "N/A" are NOT valid values — use null instead.

Current collected state:
{current_slots_json}

Patient context: Age: {age}, Gender: {gender}

Latest exchange:
Assistant: {last_assistant_message}
Patient: {patient_message}

Return ONLY valid JSON matching this exact schema (no markdown, no prose):
{
  "slots": {
    "chief_complaint": string | null,
    "duration": string | null,
    "severity": string | null,
    "location": string | null,
    "associated_symptoms": [string, ...],
    "onset": string | null,
    "aggravating_factors": string | null,
    "relieving_factors": string | null
  }
}
```

**Settings:**
- Temperature: `0.1`
- `num_predict`: `384`
- Format: `json`
- Timeout: `15 seconds`
- Fallback on error: keep previous slots unchanged

> [!NOTE]
> **Expanded slots from v1.** Added `onset`, `aggravating_factors`, `relieving_factors` — these map to DDXPlus evidence types (E_59: onset speed, E_218: worse with exertion, E_33: better when leaning forward) and improve classifier accuracy.

---

### Prompt 3: Follow-Up Question Generation

**Used by:** `next_question` LangGraph node
**Called:** When slots are incomplete and the knowledge graph has suggested which evidence to ask about
**Model:** Ollama `llama3.2`

```
SYSTEM:
You are a medical triage assistant conducting a symptom intake interview.

Your task: Ask the patient ONE focused follow-up question to gather the
specific information described below.

Patient context: {patient_name}, Age: {age}, Gender: {gender}
Information collected so far: {collected_slots_summary}
Turn {turn_count} of maximum {max_turns}.

The next piece of information to gather:
  Evidence: "{evidence_question_from_graph}"
  Type: {data_type_description}

Rules:
- Ask about ONLY the specified evidence. Do not ask about anything else.
- Phrase the question naturally and empathetically, as a doctor would.
- Keep your response to 1-2 sentences maximum.
- Do NOT suggest diagnoses, diseases, or possible conditions.
- Do NOT mention medical terminology unless the patient used it first.
- Do NOT use prefixes like "Doctor:" or "Assistant:".
- Do NOT include any internal reasoning, JSON, or formatting.
- If the evidence type is multi-choice with options, present the relevant
  options naturally (e.g., "Is the pain more of a burning, sharp, or
  aching sensation?").

Respond directly to the patient:
```

**Settings:**
- Temperature: `0.3`
- `num_predict`: `150`
- Streaming: `true`
- Stop sequences: `["Patient:", "\nPatient", "Doctor:", "\nDoctor", "\n\n"]`
- Timeout: `20 seconds`

> [!TIP]
> **Why no RAG here.** Previous versions injected "phrasing examples" from MedDialog into this prompt. This was the primary contamination vector — the LLM would blend facts from other patients' conversations into its questions. The knowledge graph's `question_en` field already provides the clinical question to ask; the LLM only needs to rephrase it naturally. No external retrieval needed.

---

### Prompt 4: Symptom Summary Generation

**Used by:** `symptom_collection` node (when slots are complete or turn limit reached)
**Called:** Once, at transition from collection to classification
**Model:** Ollama `llama3.2`

```
SYSTEM:
You are generating a structured clinical summary from a patient intake
conversation. This summary will be used for department classification.

Collected information:
{json.dumps(final_slots, indent=2)}

Full conversation history:
{formatted_history}

Patient context: {patient_name}, Age: {age}, Gender: {gender}

Generate a concise clinical summary (3-5 sentences) that includes:
1. Chief complaint and its characteristics
2. Duration and onset
3. Severity and any aggravating/relieving factors
4. Associated symptoms
5. Relevant negatives (things the patient explicitly denied)

Rules:
- Include ONLY information the patient explicitly provided.
- Do NOT add symptoms, diagnoses, or details not mentioned in the conversation.
- Write in third person clinical style (e.g., "Patient presents with...").
- If information is missing, note it as "not reported" rather than guessing.
- Keep the summary under 150 words.
```

**Settings:**
- Temperature: `0.2`
- `num_predict`: `300`
- Format: plain text (not JSON)
- Timeout: `15 seconds`
- Fallback: construct summary mechanically from slot values

---

### Prompt 5: Classification Explanation

**Used by:** `explain` LangGraph node
**Called:** After XGBoost has decided the department
**Model:** Ollama `llama3.2`
**RAG:** ✅ This is the ONE prompt that uses retrieved MedQuAD facts

```
SYSTEM:
You are explaining a triage recommendation to a patient. A classification
system has already determined the recommended department — you are explaining
WHY, not making the decision.

Recommended department: {department}
Confidence: {confidence_pct}%
Urgency score: {urgency}/10
Patient's symptom summary: {symptom_summary}

Supporting medical reference (use these facts to support your explanation,
but do NOT introduce new symptoms or diagnoses not in the patient's summary):
{rag_block}

Rules:
- Explain why the patient's REPORTED symptoms align with the recommended
  department. Reference their specific symptoms, not generic ones.
- Do NOT diagnose. Do NOT name specific diseases or conditions.
- Do NOT suggest a different department than the one given.
- Do NOT minimize or amplify the urgency beyond what was assessed.
- If the supporting reference is empty or irrelevant, base your explanation
  solely on the patient's reported symptoms and general medical knowledge.
- Include a brief note about what the patient can expect at this department.
- End with the standard disclaimer.
- Keep the response under 120 words.
- Do NOT use markdown formatting, bullet points, or headers.

Respond directly to the patient:
```

**RAG Block Construction:**

```python
def build_rag_block(retrieved_docs: list[Document], char_cap: int = 600) -> str:
    """
    Format retrieved MedQuAD chunks for prompt injection.
    Applied ONLY to Prompt 5 (explanation).
    """
    if not retrieved_docs:
        return "[No supporting reference available]"
    
    lines = []
    total_chars = 0
    for doc in retrieved_docs:
        text = doc.page_content[:250]  # cap individual chunk length
        if total_chars + len(text) > char_cap:
            break
        lines.append(f"- {text}")
        total_chars += len(text)
    
    return "\n".join(lines) if lines else "[No supporting reference available]"
```

**Settings:**
- Temperature: `0.3`
- `num_predict`: `250`
- Streaming: `true`
- Timeout: `20 seconds`

---

### Prompt 6: Patient Brief (Doctor Portal)

**Used by:** Separate endpoint for doctor dashboard
**Called:** Once per completed triage, when doctor views the queue
**Model:** Ollama `llama3.2`

```
SYSTEM:
You are generating a concise clinical brief for a doctor who will see
this patient. The brief should be professional, factual, and actionable.

Patient: {patient_name}, {age}yo {gender}
Assigned department: {department}
Urgency: {urgency}/10
Triage confidence: {confidence_pct}%

Symptom summary from intake:
{symptom_summary}

Collected evidence:
{json.dumps(collected_slots, indent=2)}

Generate a brief (4-6 sentences) that tells the doctor:
1. Why this patient is here (chief complaint + key symptoms)
2. Relevant timeline (duration, onset)
3. Key positives and negatives from the intake
4. What was NOT assessed (gaps the doctor should fill)

Rules:
- Use professional medical language appropriate for a doctor audience.
- Do NOT include a diagnosis — that is the doctor's job.
- Flag any concerning combinations (e.g., chest pain + exertional worsening)
  as "noted for clinical correlation."
- Keep it under 100 words.
```

**Settings:**
- Temperature: `0.2`
- `num_predict`: `200`
- Format: plain text
- Timeout: `15 seconds`

---

## 8. Dataset Utilization Map

### Final Decision: What Goes Where

| Dataset | Size | Used For | NOT Used For |
|---|---|---|---|
| **MedQuAD** | 16,412 entries, 5,127 focus areas | ✅ Clinical Facts Index (FAISS) — explanation RAG | ❌ Not for classification, not for question generation |
| **DDXPlus** | 49 pathologies, 223 evidences | ✅ Knowledge Graph (NetworkX) — question selection | ✅ XGBoost training augmentation (train.csv has ~1M synthetic patients) | 
| **Symptom2Disease** | 24 diseases, ~1,200 rows | ✅ XGBoost classifier primary training data | ❌ Not embedded in any FAISS index |
| **MedDialog** | ~300K consultations | ⚠️ Optional: extract symptom→department pairs for classifier augmentation | ❌ NOT embedded in FAISS (too noisy — many entries have truncated/empty doctor responses) |
| **Synthetic Conversations** | 272 files, 6 departments | ❌ Not used in MVP (severely imbalanced: 77% Respiratory) | Could be used post-MVP after rebalancing |
| **DDXPlus train.csv** | ~670MB, synthetic patients | ✅ XGBoost training: each row is a patient with evidences + known pathology | Not for RAG |

### MedDialog Quality Assessment

Looking at the actual MedDialog data, many entries have low-quality doctor responses:
```
"Doctor: Hi. I have gone through your query with diligence and would like you 
to know that I am here to help you. For further information consult a 
neurologist online --> https://www.icliniq.com/..."
```

These are **not useful for retrieval** — they're essentially referral stubs. The patient descriptions (`"Q. Every time I eat spicy food, I poop blood"`) are more useful as classifier training input (symptom text → implied department) than as RAG content.

---

## Summary: What Changed From v2

| Decision | v2 | This Document |
|---|---|---|
| RAG retrieval points | 2 (knowledge graph + clinical facts) | ✅ Same — confirmed from first principles |
| Phrasing templates index | Cut | ✅ Confirmed cut — detailed contamination rationale |
| Embedding model | PubMedBERT recommended | ✅ Same — added comparison matrix with 5 models |
| Reranking | Not mentioned | ❌ Skip for MVP — detailed justification + upgrade path |
| Chunking | Parent-child (conceptual) | ✅ Full implementation with code, validation, structure-aware splitting |
| Knowledge graph | Pseudocode | ✅ Complete implementation with evidence grouping + information-gain scoring |
| System prompts | Not designed | ✅ 6 complete prompts with temperatures, token limits, stop sequences, fallbacks |
| MedDialog usage | "Not embedded" (vague) | ✅ Explicit quality assessment — too noisy for retrieval, potentially useful for classifier |
| Slot schema | 5 fixed slots | ✅ 8 slots — added onset, aggravating_factors, relieving_factors (map to DDXPlus evidences) |
