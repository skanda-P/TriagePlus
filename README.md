# TriagePlus

TriagePlus is an AI-powered medical triage platform designed to streamline hospital intakes and patient routing. It combines an intuitive React frontend, a FastAPI WebSocket backend, and a robust LangGraph state machine powered by XGBoost and a structured Medical Knowledge Graph.

## Features

- **Interactive Triage Chat**: Real-time WebSocket streaming powered by LangGraph to gather structured symptoms from patients.
- **Safety Mitigations**: Built-in keyword safety nets (T1 Mitigation) and confidence-flooring algorithms (T2 Mitigation) to prioritize patient safety and emergency routing.
- **XGBoost Classifier**: A high-accuracy ML model trained on the expansive DDXPlus synthetic clinical dataset.
- **Doctor Portal**: Secure authentication and queue viewing powered by Supabase.

## Documentation

Detailed documentation on the technical systems can be found in the `docs/` folder:
- [Architecture Overview](docs/architecture.md)
- [LangGraph State Machine](docs/langgraph_architecture.md)
- [Knowledge Bases & ML Training](docs/knowledge_and_rag.md)

## Tech Stack

- **Backend**: FastAPI, Python, SQLite (Checkpointer), LangGraph, XGBoost
- **Frontend**: React, TailwindCSS, Vite
- **Database / Auth**: Supabase

## Setup Instructions

### Backend
1. `cd backend`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your Supabase credentials.
4. Run the API: `fastapi dev app/main.py`

### AI Engine (Model Training)
1. `cd ai_engine/ml_training`
2. `pip install xgboost scikit-learn pandas`
3. `python train_triage_model.py` (Downloads dataset and builds the XGBoost classifier locally)

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`
