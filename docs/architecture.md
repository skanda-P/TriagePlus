# TriagePlus Architecture Overview

TriagePlus is a modern medical triage assistant that uses a hybrid architecture combining Large Language Models (LLMs), a Medical Knowledge Graph, and a Machine Learning Classifier to route patients to the correct hospital department.

## System Components

1. **Frontend**: React-based patient portal for chat interactions and symptom input.
2. **Backend**: FastAPI server handling REST endpoints and WebSocket connections.
3. **AI Engine**: 
   - **LangGraph State Machine**: Manages conversation state, prompt logic, and multi-turn symptom gathering.
   - **XGBoost Classifier**: An ML model trained on DDXPlus to predict patient conditions from structured symptom vectors.
4. **Database**: Supabase PostgreSQL database handling authentication (Patient vs. Doctor roles) and triage session histories.

## Data Flow

1. Patient connects to `/ws/chat/{session_id}`.
2. A WebSocket connection is established and conversation state is tracked via **SQLite Saver**.
3. Patient describes symptoms.
4. The **LangGraph Triage Pipeline** processes the message:
   - **Emergency Screening**: Validates if the condition is life-threatening (T1 Mitigation).
   - **Symptom Extraction**: LLM maps natural language to precise `DDXPlus` evidence codes.
   - **Condition Classification**: XGBoost predicts the pathology with an attached confidence score.
   - **Department Routing**: Falls back gracefully to standard departments if specialized ones are missing.
5. The session is closed and results are written to Supabase.
