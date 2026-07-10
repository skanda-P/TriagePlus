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
   - **Symptom Extraction**: A HuggingFace Medical NER pipeline (`d4data/biomedical-ner-all`) extracts clinical entities and maps them to precise `DDXPlus` evidence codes.
   - **Emergency Screening**: Validates if the condition is life-threatening (T1 Mitigation).
   - **Next Question Generation**: LLM uses RAG on MedDialog and the Knowledge Graph to ask empathetic follow-up questions.
   - **Condition Classification**: XGBoost predicts the pathology with an attached confidence score.
   - **Explanation**: LLM uses RAG on MedQuAD to explain the diagnosis and route to a department.
   - **Booking & Payment**: The pipeline prompts the user to book a slot, fetches available clinician slots from Supabase, processes a payment, and confirms the appointment.
5. Live RAG and LangGraph node diagnostics are broadcasted to `/ws/diagnostics` for the Developer Monitor (doctor-authenticated WebSocket token required).
6. The session is closed and results are written to Supabase.

## Runtime Reliability Notes

- FAISS retrieval can run in degraded mode when MedDialog/MedQuAD indices are unavailable.
- The graph now surfaces retrieval/model health in state diagnostics (`rag_status`, `model_health`) instead of failing silently.
