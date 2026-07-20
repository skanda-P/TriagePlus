# Build Instructions — TriagePlus v4

End-to-end recipe for generating every runtime artifact the TriagePlus backend
needs:

| # | Artifact | Output path | Builder script |
|---|---------|------------|----------------|
| 1 | DDXPlus Knowledge Graph | `backend/data/ddxplus_kg.pkl` | `backend/scripts/build_ddxplus_kg.py` |
| 2 | Symptom → Department mapping | `backend/model/symptom_dept_mapping.json` + `symptom_name_dept_mapping.json` | `backend/scripts/create_symptom_dept_mapping.py` |
| 3 | XGBoost triage classifier | `backend/model/xgb_model.json`, `mlb.pkl`, `label_encoder.pkl`, `training_manifest.json` | `backend/scripts/train_xgboost.py` |
| 4 | MedQuAD FAISS + BM25 index | `backend/data/faiss/medquad/{medquad.index, medquad_metadata.pkl, medquad_bm25.pkl, medquad_summary.json}` | `backend/scripts/build_medquad_index.py` |
| 5 | Conversations FAISS + BM25 index | `backend/data/faiss/conversations/{conversations.index, conversations_metadata.pkl, conversations_bm25.pkl, conversations_summary.json}` | `backend/scripts/build_conversations_index.py` |
| 6 | MedDialog FAISS + BM25 index | `backend/data/faiss/meddialog/{meddialog.index, meddialog_metadata.pkl, meddialog_bm25.pkl, meddialog_summary.json}` | `backend/scripts/build_meddialog_qa_index.py` |

> **The contract between build scripts and runtime is enforced by
> `backend/tests/test_rag_path_contract.py`** — every file the runtime opens must
> be written by the corresponding build script. If you rename an output file,
> keep that test green.

---

## 0. Prerequisites

### 0.1 Environment
```bash
# From repo root
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt        # CPU build (default)
# OR, for CUDA 12.1+ hosts (much faster XGBoost + sentence-transformers):
pip install -r requirements-gpu.txt
```

### 0.2 Required source data

All raw datasets must live under `backend/data/` before any script runs. They
are **gitignored** — download or symlink them yourself.

| Path | Approx. size | Source |
|------|--------------|--------|
| `backend/data/DDXPlus/train.csv` | ~640 MB | DDXPlus dataset (release train split) |
| `backend/data/DDXPlus/validate.csv` | ~85 MB | DDXPlus dataset (optional, used for eval) |
| `backend/data/DDXPlus/test.csv` | ~85 MB | DDXPlus dataset |
| `backend/data/DDXPlus/eval_set.json` | ~400 KB | DDXPlus dataset (consumed by KG builder) |
| `backend/data/DDXPlus/release_conditions.json` | ~21 KB | DDXPlus metadata |
| `backend/data/DDXPlus/release_evidences.json` | ~120 KB | DDXPlus metadata |
| `backend/data/medquad.csv` | ~22 MB | MedQuAD corpus (columns: `question,answer,source,focus_area`) |
| `backend/data/medquad.jsonl` | ~22 MB | Optional — used by tooling only |
| `backend/data/en_medical_dialog.json` | ~281 MB | MedDialog EN (list of `{Description, Doctor, Patient}` objects) |
| `backend/data/meddialog_en_train.jsonl` | ~271 MB | MedDialog EN train split (optional alt format) |
| `backend/data/Symptom2Disease.csv` | ~200 KB | Symptom2Disease Kaggle dataset (`Disease,Symptom` columns) |

A symlink example (Windows, dev inner loop):
```powershell
New-Item -ItemType SymbolicLink -Path backend\data\DDXPlus -Target D:\datasets\ddxplus\DDXPlus
```

### 0.3 Conversation logs (for the Conversations index)
`backend/data/prompts/<Specialty>/*.txt` — formatted as:

```
Doctor: <question or statement>
Patient: <reply>
Doctor: <follow-up>
...
```

Existing directories shipped with the repo:
`Cardiology/`, `Dermatology/`, `Gastroenterology/`, `General/`,
`Musculoskeletal/`, `Respiratory/`.

### 0.4 Sanity check
```bash
python -c "import faiss, sentence_transformers, xgboost, networkx, rank_bm25, scipy; print('imports OK')"
```

If that prints `imports OK` you are ready to build.

---

## 1. Build the Knowledge Graph

```bash
# Run from backend/ (so DATA_DIR resolves correctly)
python scripts/build_ddxplus_kg.py
```

**What it does**
1. Loads `DDXPlus/release_conditions.json` + `release_evidences.json`
   (condition metadata, severity, evidence display text).
2. Streams `DDXPlus/eval_set.json` case-by-case; for every case it records:
   - `evidence_condition_counts[E][C]`  → # times evidence E was **present** for pathology C
   - `evidence_condition_absent_counts[E][C]` → # times E was **absent** for C
   - `condition_case_counts[C]` → total # cases per pathology
3. Builds a `networkx.DiGraph` with `condition → evidence` edges typed as
   `present` (weight 1.0) or `absent` (weight 0.0).
4. Pickles everything to **`backend/data/ddxplus_kg.pkl`**.

**Outputs**
```
backend/data/ddxplus_kg.pkl   (~200 KB)
```

**Verify**
```bash
python -c "import pickle; kg = pickle.load(open('backend/data/ddxplus_kg.pkl','rb')); \
           print('nodes:', kg['graph'].number_of_nodes(), \
                 'edges:', kg['graph'].number_of_edges(), \
                 'conditions:', len(kg['conditions']), \
                 'evidences:', len(kg['evidences']))"
# Expected: nodes ≈ 405, edges ≈ 30k, conditions = 49, evidences ≈ 356
```

The runtime uses this pickle for:
- `rank_next_questions()` — expected-posterior-entropy Information Gain for the
  next-symptom-question ranker (`app/core/kg.py`).
- `get_condition_specialty()` / `get_condition_severity()` — KG-driven department
  + ESI-style triage-level lookup (`app/core/triage_graph.py`).

---

## 2. Build the Symptom → Department mapping

```bash
python scripts/create_symptom_dept_mapping.py
```

**What it does**
Combines three evidence-code → department sources in priority order:

1. **KG-derived** — for each DDXPlus condition, look up its specialty (via the
   `keyword → specialty` table in the script), then propagate that specialty to
   every evidence that appears under the condition. Vote by Counter; keep the
   most common specialty per evidence code.
2. **Symptom2Disease.csv** — token-overlap match between evidence `question_en`
   text and symptoms listed in the CSV.
3. **Keyword fallback** — final fallback using the curated `DEPT_KEYWORDS` table
   (matches against `question_en` text so unmatched evidence codes still get a
   department).

**Outputs**
```
backend/model/symptom_dept_mapping.json        # evidence-code -> department
backend/model/symptom_name_dept_mapping.json   # human text     -> department
```

> **Order matters**: this script reads `backend/data/ddxplus_kg.pkl`, so run
> step 1 first if the KG is not yet built. If the KG file is missing, the script
> degrades gracefully and uses only sources 2 + 3.

---

## 3. Train the XGBoost triage classifier

```bash
# CPU (default — XGBoost auto-falls back from CUDA on failure)
python scripts/train_xgboost.py

# Force CPU even on a GPU box (useful for CI):
XGBOOST_DEVICE=cpu python scripts/train_xgboost.py

# Force CUDA (skips the auto-detection):
XGBOOST_DEVICE=cuda python scripts/train_xgboost.py
```

**What it does**
1. Reads `DDXPlus/train.csv` (~640 MB, ~1.2 M cases).
2. Parses `EVIDENCES` (a Python list literal stored as a string), strips the
   `_@_V_*` value suffixes so each row has a set of base evidence codes
   (e.g. `E_55`, `E_91`).
3. Binarizes the multi-label evidence vector with
   `sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)`.
4. Adds `AGE` and binary `SEX` (M=1 / F=0) features, hstacks as sparse matrix
   `[AGE, SEX, *evidence_bits]`.
5. Encodes the target `PATHOLOGY` with `LabelEncoder` → 49 classes.
6. Trains `xgboost.XGBClassifier(tree_method='hist', device=auto,
   n_estimators=100, max_depth=6, learning_rate=0.1,
   objective='multi:softprob')`.
7. On `XGBoostError` mentioning `cuda`/`gpu`/`cublas`, retries on CPU
   automatically.
8. Saves the booster, encoders, and a JSON manifest the runtime uses to restore
   constructor params (`objective`, `num_class`, `feature_order`) — without the
   manifest, `predict_proba` on the loaded booster can route through the wrong
   path.

**Outputs**
```
backend/model/xgb_model.json
backend/model/mlb.pkl
backend/model/label_encoder.pkl
backend/model/training_manifest.json
```

**Verify**
```bash
python -c "import json, pickle, xgboost as xgb; \
           m=xgb.XGBClassifier(); m.load_model('backend/model/xgb_model.json'); \
           print('num_class:', json.load(open('backend/model/training_manifest.json'))['num_class']); \
           print('mlb classes:', len(pickle.load(open('backend/model/mlb.pkl','rb')).classes_)); \
           print('le classes :', len(pickle.load(open('backend/model/label_encoder.pkl','rb')).classes_))"
# Expected: num_class = 49, mlb classes ≈ 356, le classes = 49
```

**Runtime consumer**: `app/core/triage_graph.py::_load_xgboost_artifacts()`
loads these four files once per process and produces the
`final_diagnosis` / `confidence` / `triage_level` for each chat turn.

**Tip — feature parity**: the manifest's `feature_order` documents the exact
order the runtime must use. If you change the training features (e.g. add
`GENDER=2` for non-binary), also update `_load_xgboost_artifacts` and
`_get_patient_demographics` in `app/core/triage_graph.py` so inference still
matches training.

---

## 4. Build the FAISS + BM25 indices (hybrid RAG)

The three indices are **independent** — build them in any order or in parallel
(each takes ~1–5 min on CPU, dominated by PubMedBERT embedding):

```bash
python scripts/build_medquad_index.py         # ~22 MB corpus, fast
python scripts/build_conversations_index.py  # ~6 specialties, fast
python scripts/build_meddialog_qa_index.py    # ~281 MB, slowest
```

### 4.1 MedQuAD index

```bash
python scripts/build_medquad_index.py
```
- **Source**: `backend/data/medquad.csv` (`question,answer,source,focus_area`).
- **Chunker**: atomic QA pairs, long answers split per-paragraph (parent +
  child link via `chunk_index` / `total_chunks`).
- **Embedding**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
  (768-d, L2-normalized at encode time). Corpus + query normalization is what
  makes `IndexFlatL2` behave like cosine.
- **Index**: `IndexIDMap(IndexFlatL2(768))` with explicit positional ids.
- **BM25**: `rank_bm25.BM25Okapi` over `answer_chunk` text, libc-RE
  `\b\w+\b` tokenizer (shared with the runtime query tokenizer in
  `app/core/unified_retrieval.py::_tokenize()`).
- **Hybrid weights** (runtime): `BM25 0.3 / Dense 0.7`.

**Outputs**
```
backend/data/faiss/medquad/medquad.index
backend/data/faiss/medquad/medquad_metadata.pkl
backend/data/faiss/medquad/medquad_bm25.pkl
backend/data/faiss/medquad/medquad_summary.json   (human-readable, not read at runtime)
```

### 4.2 Conversations index

```bash
python scripts/build_conversations_index.py
```
- **Source**: every `*.txt` under `backend/data/prompts/<Specialty>/`.
- **Chunker**: 3-turn sliding window (1-turn stride / 2-turn overlap). Each
  chunk carries the full `Patient: ... Doctor: ... Patient: ... Doctor: ...`
  text PLUS a pre-extracted `doctor_few_shot` field used for LLM prompt injection.
- **Embedding + Index**: same as MedQuAD (PubMedBERT-L2 + `IndexIDMap`).
- **BM25**: over `full_text` (so patient-side symptom tokens also get scored).
- **Hybrid weights** (runtime): `BM25 0.4 / Dense 0.6`.

**Outputs**
```
backend/data/faiss/conversations/conversations.index
backend/data/faiss/conversations/conversations_metadata.pkl
backend/data/faiss/conversations/conversations_bm25.pkl
backend/data/faiss/conversations/conversations_summary.json
```

### 4.3 MedDialog Q&A index

```bash
python scripts/build_meddialog_qa_index.py
```
- **Source**: `backend/data/en_medical_dialog.json` (a JSON list of
  `{Description, Doctor, Patient}` objects). The script also tolerates a dict
  wrapper.
- **Chunker**: atomic QA pairs (1 chunk per object). `full_text =
  Description + " " + Doctor` — patient tokens are NOT prepended again to avoid
  double-counting in BM25's document-length normalization.
- **Index**: `IndexIDMap(IndexFlatL2(768))` with positional ids
  (this used to be `IndexFlatL2.add_with_ids`, which is unsupported and would
  crash mid-build — the contract test in
  `backend/tests/test_rag_path_contract.py` guards against a regression).
- **Hybrid weights** (runtime): `BM25 0.5 / Dense 0.5`.

**Outputs**
```
backend/data/faiss/meddialog/meddialog.index
backend/data/faiss/meddialog/meddialog_metadata.pkl
backend/data/faiss/meddialog/meddialog_bm25.pkl
backend/data/faiss/meddialog/meddialog_summary.json
```

---

## 5. Full pipeline (one-shot)

Run all six build steps in order, using a fresh shell, from `backend/`:

```bash
cd backend
python -m venv venv && source venv/bin/activate    # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt

# 1) Knowledge Graph
python scripts/build_ddxplus_kg.py

# 2) Symptom → Department mapping (depends on step 1)
python scripts/create_symptom_dept_mapping.py

# 3) XGBoost triage model
python scripts/train_xgboost.py

# 4-6) RAG indices (run in parallel in three terminals if you like)
python scripts/build_medquad_index.py
python scripts/build_conversations_index.py
python scripts/build_meddialog_qa_index.py

# Sanity: confirm contract between build scripts and runtime
pytest tests/test_rag_path_contract.py -v
```

After this, your `backend/` tree should look like:

```
backend/
├── data/
│   ├── ddxplus_kg.pkl
│   ├── DDXPlus/                          (raw dataset, gitignored)
│   ├── prompts/<Specialty>/*.txt
│   └── faiss/
│       ├── medquad/      {medquad.index, medquad_metadata.pkl, medquad_bm25.pkl, ...}
│       ├── conversations/{conversations.index, conversations_metadata.pkl, conversations_bm25.pkl, ...}
│       └── meddialog/    {meddialog.index, meddialog_metadata.pkl, meddialog_bm25.pkl, ...}
└── model/
    ├── xgb_model.json
    ├── mlb.pkl
    ├── label_encoder.pkl
    ├── training_manifest.json
    ├── symptom_dept_mapping.json
    └── symptom_name_dept_mapping.json
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Could not find a version that satisfies ... faiss-cpu==1.8.0` | Stale requirements pin | Updated in `requirements.txt` to `faiss-cpu>=1.9,<2.0` |
| `Could not find ... langgraph-checkpoint-sqlite==1.1.1` | Stale requirements pin (1.x never shipped on PyPI) | Updated in `requirements.txt` to `>=2.0,<3.0` |
| `Microsoft Visual C++ 14.0 or greater is required` while installing `scikit-learn` on Python 3.13 | No prebuilt wheel for `scikit-learn<1.5` on 3.13 | Relax pin to `scikit-learn>=1.4,<2.0` (already done), OR use Python 3.10–3.12 |
| `xgb.core.XGBoostError: ... cuBLAS ...` | CUDA libs unavailable or driver mismatch | Script auto-falls back to CPU; or set `XGBOOST_DEVICE=cpu` |
| `RuntimeError: ... CUDA out of memory` while training XGBoost | GPU out of VRAM with the 640 MB CSV | Set `XGBOOST_DEVICE=cpu` (hist tree method is fine on CPU) |
| `HuggingFaceUnableToLoad ... biomedical-ner-all` | HF Hub unreachable / offline | Pre-download once via `huggingface-cli download d4data/biomedical-ner-all`; runtime falls back to regex keyword extraction if model load fails 3×. |
| Sentence-transformers downloading `microsoft/BiomedNLP-PubMedBERT-...` on every build | Default HF cache moved / unset | Set `HF_HOME=/path/to/cache` (or `SENTENCE_TRANSFORMERS_HOME`) so it persists across builds. |
| `IndexFlatL2.add_with_ids` error from a build script | (Should not happen — regression guard exists) | Ensure you're using `build_meddialog_qa_index.py` (the legacy `build_meddialog_index.py` was removed). |
| `FAISS index.ntotal != len(chunks)` warning at runtime | A build was interrupted / output files were partially overwritten | Re-run the corresponding build script — script writes atomically via `faiss.write_index` and `pickle.dump` after both pieces are computed. |
| `No FAISS indices could be loaded` warning at backend startup | Indices were built outside `backend/data/faiss/<source>/` or files were renamed | Verify the path-contract test: `pytest backend/tests/test_rag_path_contract.py -v` |

---

## 7. Rebuild triggers (when to re-run what)

| Change in the repo / data | Re-run |
|---------------------------|--------|
| New `DDXPlus/eval_set.json` cases | `build_ddxplus_kg.py` → `create_symptom_dept_mapping.py` → `train_xgboost.py` |
| New evidence codes in `release_evidences.json` | `build_ddxplus_kg.py` → `create_symptom_dept_mapping.py` → `train_xgboost.py` |
| New condition in `release_conditions.json` | same as above |
| Edited `medquad.csv` | `build_medquad_index.py` only |
| Added/edited `prompts/<Specialty>/*.txt` | `build_conversations_index.py` only |
| Updated `en_medical_dialog.json` | `build_meddialog_qa_index.py` only |
| Bumped embedding model name in `unified_retrieval.py` + the three build scripts | **all three** build_*_index.py scripts + a deploy of the backend (the runtime model name and the build-time model name MUST match) |
| Added a new medical specialty | Add `prompts/<Specialty>/` → `build_conversations_index.py`; extend `specialty_mapping` in `app/core/kg.py:get_condition_specialty()`; reseed Supabase `specialty` table |

---

## 8. CI-friendly quick check (no GPU, no large datasets required)

The pytest suite has explicit guards that the build scripts and runtime agree
on file paths and FAISS usage, without needing the actual datasets:

```bash
pip install -r requirements.txt
pytest backend/tests/test_rag_path_contract.py -v
pytest backend/tests/test_intake_fsm.py backend/tests/test_doctor_isolation.py -v
python -m ruff check backend/app backend/scripts backend/tests
python -m compileall -q backend/app backend/scripts backend/tests
```

All four commands should exit 0 with no output (other than passing test dots).
