# TriagePlus Codebase Documentation

This document provides a detailed overview of the TriagePlus project, breaking down the file structure, the LLM & RAG architecture, the databases utilized, and the system prompts that guide the AI responses.

## 1. Project Architecture and File Structure

The project is structured into frontend, backend, RAG (Retrieval-Augmented Generation), and various configuration and documentation folders.

### Root Directory
- **`README.md`**: Contains the project overview, setup instructions, and feature details.
- **`DESIGN.md`**: Specifies the design system for the frontend (colors like Canopy Green, typography like DM Sans, etc.).
- **`run.py`**: A convenient wrapper script to launch the FastAPI backend server from the root directory.
- **`requirements.txt`**: Lists all the Python dependencies required for the backend and RAG modules.

### Backend (`/backend`)
The backend is a FastAPI application managing the core logic, WebSockets, and state of patient triage.
- **`backend/app/main.py`**: The entry point of the FastAPI application. It configures CORS (allowing frontend connections) and registers the API routers (specifically the chat and diagnostic WebSockets).
- **`backend/app/api/v1/chat.py`**: The core controller for the application. It handles:
  - The WebSocket endpoint (`/ws/chat/{session_id}`) for patient communication.
  - An in-memory state machine (FSM) that walks the patient through demographic data collection (Name $\rightarrow$ Age $\rightarrow$ Gender $\rightarrow$ Phone) before passing control to the AI.
  - Real-time streaming of AI responses using Python asynchronous generators.
  - A diagnostic WebSocket (`/ws/diagnostics`) to broadcast real-time inference data and latencies to diagnostic clients.
  - Endpoints for mock doctor logins and queues.

### RAG & ML Training (`/RAG`)
This folder contains all the AI modeling and RAG implementation files.
- **`RAG/ml_training/gemini_inference.py`**: Despite the original name, this module uses **Ollama** with the `llama3.2` model. It handles:
  - Loading SentenceTransformer and FAISS globally.
  - `infer_department_interactive()`: Interactive loop that determines if enough symptom slots (`chief_complaint`, `duration`, `severity`, `location`, `associated_symptoms`) have been filled. If not, it streams a question back to the patient. 
  - `infer_department_final()`: Once the state machine considers the conversation complete, this function assigns a hospital department, confidence score, and urgency rating based on the gathered symptoms.
  - `check_emergency_llm()`: An asynchronous LLM call to identify if the current user input is a life-threatening medical emergency.
- **`RAG/ml_training/run_inference.py`**: A standalone inference script used for testing the Gemini model isolated from the main backend.
- **`RAG/ml_training/setup_and_train.py` & `recover_faiss.py`**: Scripts used to generate, build, and recover the FAISS indices from the training datasets.
- **`RAG/faiss/`**: Directory containing the generated FAISS index files (`index_a.faiss`, `index_b.faiss`) and their metadata JSONs.

### Frontend (`/frontend`)
The frontend is a React + Vite web application utilizing Tailwind CSS.
- **`frontend/src/App.tsx` & `main.tsx`**: Core React components and application entry point.
- **`frontend/src/index.css`**: Defines global CSS rules and integrates Tailwind directives.
- **`frontend/src/stores/chatStore.ts`**: A Zustand state management file that handles chat messages, appending streaming chunks, and the FSM (Finite State Machine) state of the application.
- **`frontend/src/stores/themeStore.ts`**: A Zustand state manager that toggles between Dark and Light mode, persisting the preference in local storage.

### Documentation & Prompts (`/prompts`)
- **`01_Database_Build_Prompt.md`, `02_AI_Components_Build_Prompt.md`, etc.**: Documentation describing the step-by-step build plan and prompts used to originally bootstrap the project's components.
- **`MediGuide_Final_AI_Plan.md`**: Contains the complete high-level architectural plan of the AI logic in the system.

---

## 2. LLM and RAG Architecture

The project employs a modern, localized Retrieval-Augmented Generation (RAG) architecture tightly coupled with a state machine to ensure accurate medical triage.

### Large Language Model (LLM) Setup
- **Model Used**: Localized **Ollama** running the `llama3.2` model. (Wait times are drastically reduced by running the model locally, bypassing typical cloud latency).
- **Interactive Triage (Slot Filling)**: 
  The LLM acts as an extraction and generation engine. On every turn, it uses a JSON extraction prompt to parse the conversation and update specific medical slots (Chief Complaint, Duration, Severity, Location, Associated Symptoms).
  If all slots are filled or a turn limit is reached, it passes the data to the final triage step. If not, the LLM is prompted to ask a short follow-up question.
- **Emergency Detection**: A dedicated rapid LLM call runs asynchronously on every user message to intercept life-threatening scenarios.

### Retrieval-Augmented Generation (RAG) Setup
- **Embeddings Model**: The system uses a domain-specific dual-encoder architecture (`ncbi/MedCPT-Query-Encoder` and `ncbi/MedCPT-Article-Encoder`) loaded locally with PyTorch/CUDA to create 768-dimensional dense vector representations.
- **Indices**:
  - **Index A (Conversational Cases)**: Used during the interactive chat phase. Contains subsampled MedDialog real patient consultations and capped synthetic cases. When a user mentions a symptom, the system queries this index to find similar historical patient-doctor conversations, appending them to the LLM context to guide its questioning.
  - **Index B (Medical Knowledge)**: Used during the final triage phase. Contains chunked MedQuAD QA pairs and Symptom2Disease facts that help the final inference step safely choose a hospital department and determine the urgency score.

---

## 3. Databases Used

The project avoids heavy relational databases to maintain speed and focus on real-time triage.
1. **Vector Database (FAISS)**: 
   - **FAISS (Facebook AI Similarity Search)** is the only external database paradigm used. It handles high-speed similarity search for dense vectors. The indices are pre-compiled and loaded entirely into RAM on startup, ensuring ultra-low latency context retrieval for the LLM.
2. **Persistent Session Store (SQLite)**: 
   - The backend (`chat.py`) uses a local SQLite database (`sessions.db`) to manage the WebSocket connections, patient demographic data, chat history, and current FSM state. This persists the session state across server restarts and page refreshes while avoiding the overhead of external dependencies like Redis for a single-node deployment.

---

## 4. System Prompts 

The LLM logic relies on highly engineered prompts to ensure consistent JSON outputs and professional tone.

### A. Extraction Prompt (JSON Mode)
Used during the chat loop to identify and update symptoms:
```text
You are extracting structured medical intake data from a doctor-patient exchange.
Given the running state and the new turn, return ONLY updated JSON (no prose, no markdown fences).
PATIENT CONTEXT: Age: ..., Gender: ...
Current state: { ... }
New turn: Doctor: ... Patient: ...
Update only the fields the new turn provides new information for. Keep existing values unchanged unless contradicted.
Return strict JSON matching this schema:
{"slots": {"chief_complaint": str|null, "duration": str|null, "severity": str|null, "location": str|null, "associated_symptoms": [str, ...]}}
```

### B. Interactive Generation Prompt (Streaming Mode)
Used to generate the assistant's follow-up questions during triage:
```text
You are an expert medical triage assistant.
PATIENT CONTEXT: Age: ..., Gender: ...
already_collected: { ... }
still_needed: [ ... ]

UNRELATED PAST CASES (for question-phrasing style only — do NOT state, imply, or ask about any specific fact, diagnosis, or detail from these unless the current patient has said it themselves):
[ RAG Context Here ]

INSTRUCTION: Do NOT ask about any field in already_collected. Ask about exactly one field from still_needed next, or politely ask for any other details if still_needed is empty. Do not copy or reference specific details from the unrelated past cases below — they are style examples only.
Respond directly to the patient in 1-2 short sentences. Do not include any internal thoughts, formatting, prefixes like "Doctor:", or JSON.
```

### C. Final Triage Classification Prompt (JSON Mode)
Used when slot filling is complete to assign a department and urgency:
```text
You are an expert AI triage assistant.
Analyze the patient's symptom summary and assign them to the most appropriate hospital department.
Also, provide an urgency score from 1 to 10 (1 = non-urgent, 10 = life-threatening emergency).

If the symptoms are common, non-specific, and show no red-flag or specialty-defining features (e.g., isolated fever, mild headache, general fatigue, common cold symptoms), assign "General Medicine" rather than a specific specialty.

Medical knowledge reference:
[ RAG Context from Index B Here ]

Patient context: Age: ..., Gender: ...
Patient summary: { ... }

You must output ONLY a valid JSON object with these fields:
- "department": one of Cardiology, Dermatology, Emergency Medicine, ...
- "confidence": a float between 0.0 and 1.0 representing your certainty.
- "urgency_score": an integer from 1 to 10.
- "reasoning": a brief explanation of why you chose this department and urgency.
```

### D. Emergency Detection Prompt
Evaluates every message rapidly:
```text
Analyze the following patient message. Is it a life-threatening medical emergency (e.g. heart attack, stroke, severe bleeding, not breathing, unconscious)?
Ignore negative statements like "I do not have chest pain".
Reply ONLY with a valid JSON object: {"is_emergency": true/false}.

Patient Message: "..."
```
