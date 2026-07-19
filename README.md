# TriagePlus v4

AI-powered medical triage platform with LangGraph conversational engine, hybrid RAG (BM25+Dense), XGBoost symptom classification, and real-time Doctor Portal.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI        │────▶│   Supabase      │
│   (React/Vite)  │ WS  │   (LangGraph)    │     │   (PostgreSQL)  │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌────────────┐    ┌─────────────┐    ┌─────────────┐
        │  XGBoost   │    │  Hybrid     │    │  Knowledge  │
        │  Classifier│    │  RAG (BM25+ │    │  Graph      │
        │  (DDXPlus) │    │  Dense)     │    │  (DDXPlus)  │
        └────────────┘    └─────────────┘    └─────────────┘
              ▲                  ▲                  ▲
              │                  │                  │
        ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
        │HF NER     │      │FAISS      │      │Condition  │
        │(biomedical│      │Indices    │      │Specialties│
        │ ner-all)  │      │(3 sources)│      │           │
        └───────────┘      └───────────┘      └───────────┘
```

## Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Symptom Extraction** | `d4data/biomedical-ner-all` (HF) | NER → DDXPlus evidence codes (E_XX) |
| **Emergency Detection** | Conservative keyword + combo rules | ESI-like triage levels 1-5 |
| **Classification** | XGBoost (CUDA) on DDXPlus | 49 pathology classification |
| **Department Mapping** | KG specialty → Symptom2Disease → Keywords | Cardiology, Neurology, etc. |
| **Hybrid RAG** | FAISS + BM25 (3 sources) | MedQuAD 0.3/0.7, Conv 0.4/0.6, MedDialog 0.5/0.5 |
| **Next Question** | KG Information Gain | Intelligent symptom follow-up |
| **Real-time** | WebSocket + LangGraph checkpointer | Session persistence |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Supabase project
- CUDA 12.1+ (for GPU training) or CPU fallback

### 1. Database Setup
```bash
# Run SQL in Supabase SQL Editor
supabase/migrations/0001_init.sql
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create `backend/.env`:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
DEVELOPER_PASSWORD=devpass
OLLAMA_BASE_URL=http://localhost:11434
CONFIDENCE_FLOOR=0.3
```

### 4. Generate Artifacts (GPU Device)

**Step 1: Department Mapping**
```bash
python scripts/create_symptom_dept_mapping.py
# Output: backend/model/symptom_dept_mapping.json
```

**Step 2: Train XGBoost**
```bash
python scripts/train_xgboost.py
# Output: backend/model/xgb_model.json, mlb.pkl, label_encoder.pkl
```

**Step 3: Build FAISS Indices**
```bash
python scripts/build_medquad_index.py
python scripts/build_conversations_index.py
python scripts/build_meddialog_qa_index.py
# Output: backend/data/faiss/{medquad,conversations,meddialog}/
```

### 5. Frontend Setup
```bash
cd frontend
npm install
```

Create `frontend/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### 6. Run Application

**Backend** (terminal 1):
```bash
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (terminal 2):
```bash
cd frontend && npm run dev
```

**Ollama** (terminal 3):
```bash
ollama pull llama3.2
ollama serve
```

## Project Structure

```
triagePlus_v4/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── triage_graph.py      # LangGraph state machine
│   │   │   ├── ner_symptom_extractor.py  # HF NER + regex fallback
│   │   │   ├── unified_retrieval.py      # Hybrid BM25+Dense (3 sources)
│   │   │   ├── rag.py                    # FAISS query engine
│   │   │   ├── kg.py                     # DDXPlus Knowledge Graph
│   │   │   ├── symptom_to_dept.py        # Runtime dept prediction
│   │   │   └── emergency_detection.py    # Conservative ESI rules
│   │   ├── routers/
│   │   │   ├── chat.py           # Patient WebSocket
│   │   │   ├── diagnostics.py    # Dev monitor WebSocket
│   │   │   └── doctor.py         # Doctor Portal REST
│   │   └── main.py               # FastAPI + lifespan
│   ├── scripts/
│   │   ├── train_xgboost.py              # XGBoost training (CUDA)
│   │   ├── create_symptom_dept_mapping.py # Dept mapping generator
│   │   ├── build_medquad_index.py        # MedQuAD FAISS + BM25
│   │   ├── build_conversations_index.py  # Conv FAISS + BM25
│   │   ├── build_meddialog_qa_index.py   # MedDialog FAISS + BM25
│   │   └── build_ddxplus_kg.py           # Knowledge Graph builder
│   ├── data/
│   │   ├── faiss/               # Generated indices (gitignored)
│   │   ├── DDXPlus/             # Raw DDXPlus dataset
│   │   └── prompts/             # Doctor conversation logs
│   └── model/                   # Generated models (gitignored)
│       ├── xgb_model.json
│       ├── mlb.pkl
│       ├── label_encoder.pkl
│       └── symptom_dept_mapping.json
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Chat.tsx           # Patient chat
│   │   │   ├── DoctorPortal.tsx   # Doctor dashboard
│   │   │   └── RagMonitor.tsx     # Dev diagnostics
│   │   └── ...
│   └── ...
└── supabase/
    └── migrations/              # Database schema
```

## API Endpoints

### Patient WebSocket
```
WS /api/v1/ws/chat/{session_id}
```
Message types: `intake_form`, `message`, `booking_confirm`, `payment`

### Diagnostics Monitor
```
WS /api/v1/ws/diagnostics?token={DEVELOPER_PASSWORD}
```
Real-time LangGraph node events + state

### Doctor Portal (REST)
```
GET  /api/v1/doctor/slots
POST /api/v1/doctor/appointments
GET  /api/v1/public/specialties
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_FLOOR` | `0.3` | Confidence threshold for "Uncertain Diagnosis" |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `DEVELOPER_PASSWORD` | `devpass` | Diagnostics WS auth |

## Data Sources

| Source | Size | Use Case | Hybrid Weight |
|--------|------|----------|---------------|
| **MedQuAD** | ~12k QA | Medical knowledge | BM25 0.3 / Dense 0.7 |
| **Conversations** | ~5k turns | Few-shot doctor prompts | BM25 0.4 / Dense 0.6 |
| **MedDialog** | ~100k QA | Direct Q&A answering | BM25 0.5 / Dense 0.5 |

## Development

### Running Tests
```bash
cd backend
pytest tests/ -v
```

### Linting
```bash
ruff check backend/app/
```

### Adding New Specialty
1. Add conversation logs to `backend/data/prompts/{Specialty}/`
2. Re-run `build_conversations_index.py`
3. KG specialty mapping in `app/core/kg.py:get_condition_specialty()`

## License
MIT