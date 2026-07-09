# TriagePlus Setup Instructions

Welcome to TriagePlus! These instructions will guide you through running the platform locally on your machine.

## Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- A Supabase account (free tier is sufficient)

## 1. Database Setup (Supabase)
TriagePlus uses Supabase for authentication and database storage.

1. Create a new project in [Supabase](https://supabase.com/).
2. Navigate to the SQL Editor in your Supabase dashboard.
3. Copy the contents of `backend/schema.sql` and run it in the SQL Editor. This will set up the necessary tables (`doctor`, `patient`, `clinician_slot`, `appointment`, `triage_session`).
4. In Supabase Authentication settings, ensure **Email Providers** are enabled.

## 2. Backend Setup
The backend runs on FastAPI and powers the LangGraph and WebSocket chat interfaces.

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Mac/Linux: source venv/bin/activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```
5. Edit `.env` and fill in your Supabase URL and Anon Key (found in your Supabase Project Settings -> API).
6. Start the backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   *The backend will now be running on `http://localhost:8000`.*

## 3. Knowledge Bases & ML Models (Optional)
**Note**: The XGBoost triage model and necessary datasets (DDXPlus, MedQuAD, MedDialog) are already included in this repository. The model (`triage_xgb.json`) is pre-trained.

If you wish to re-train the models or build the FAISS RAG indices from scratch:
1. Open a new terminal.
2. Navigate to the ML training directory:
   ```bash
   cd ai_engine/ml_training
   ```
3. Install ML dependencies:
   ```bash
   pip install xgboost scikit-learn pandas sentence-transformers langchain-community faiss-cpu
   ```
4. **To Re-train XGBoost Model**:
   ```bash
   python train_triage_model.py
   ```
5. **To Build FAISS RAG Indices**:
   ```bash
   python build_knowledge_bases.py
   ```
   *(This step takes several minutes as it embeds tens of thousands of medical texts).*

## 4. Frontend Setup
The frontend is a React application built with Vite and TailwindCSS.

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the Node dependencies:
   ```bash
   npm install
   ```
3. Copy the environment variables template:
   ```bash
   cp .env.example .env.local
   ```
4. Edit `.env.local` to match your Supabase credentials (VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY).
5. Start the frontend development server:
   ```bash
   npm run dev
   ```
   *The frontend will now be running, typically at `http://localhost:5173`.*

## 5. Usage
- **Patient Portal**: Go to `http://localhost:5173/` to start a new triage chat.
- **Doctor Portal**: Go to `http://localhost:5173/doctor/login` (You can create a doctor account directly through Supabase Auth dashboard).
- **RAG Diagnostics**: Go to `http://localhost:5173/diagnostics` to view live RAG metadata and LangGraph events as you chat.
