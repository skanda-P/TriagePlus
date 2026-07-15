# TriagePlus v2

TriagePlus is an AI-powered medical triage platform featuring:
- A LangGraph-powered conversational AI engine.
- RAG using FAISS and HuggingFace for retrieving similar clinical conversations and MedQuAD information.
- XGBoost-based symptom classification (trained on DDXPlus).
- A real-time Doctor Portal and diagnostics monitor.

## Setup Instructions

### 1. Database Setup
1. Create a Supabase project.
2. Run the SQL script located in `supabase/migrations/0001_init.sql` in the Supabase SQL editor to create all necessary tables and functions.
3. Obtain your `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `SUPABASE_ANON_KEY`.

### 2. Backend Setup
The backend requires Python 3.9+.

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note on CUDA & FAISS on Windows:**
If you are on Windows, `faiss-gpu` is not available via standard `pip`. 
To ensure FAISS uses GPU acceleration on Windows, install `faiss-gpu` via Conda instead of pip:
```bash
conda install -c pytorch faiss-gpu
```

Configure your `.env` file in the `backend/` directory:
```env
SUPABASE_URL=<your-supabase-url>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
DEVELOPER_PASSWORD=devpass
```

### 3. Training the XGBoost Model
The XGBoost model classifies the patient's symptoms into a pathology using the DDXPlus dataset.

```bash
python backend/scripts/train_xgboost.py
```
This script runs with `device='cuda'` for GPU acceleration. Ensure your CUDA toolkit is properly configured.

### 4. Generating FAISS Indices (RAG Embeddings)
The semantic search indices for the RAG pipeline are generated separately. If you are generating the embeddings locally, follow these steps:

1. Create a script or use an existing script to process the source data (e.g., MedQuAD XML files, clinical conversation transcripts).
2. Configure HuggingFace embeddings inside your script, ensuring that you use GPU acceleration for speed:
   ```python
   from langchain_community.embeddings import HuggingFaceEmbeddings
   embeddings = HuggingFaceEmbeddings(
       model_name="NeuML/pubmedbert-base-embeddings",
       model_kwargs={'device': 'cuda'}  # Ensure 'cuda' is set
   )
   ```
3. Initialize a FAISS vector store with your loaded documents and the embedding model.
4. Save the generated indices to their designated folders so the backend can load them on startup:
   - MedQuAD Index: Save to `backend/model/faiss/medquad`
   - Conversational Index: Save to `backend/model/faiss/conversations`
   
*(Note: If you have a dedicated script like `backend/scripts/generate_embeddings.py`, run it using your virtual environment where CUDA and FAISS are correctly installed.)*

### 5. Frontend Setup
The frontend is a React + Vite application using Tailwind CSS.

```bash
cd frontend
npm install
```

Configure your `.env` file in the `frontend/` directory:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
VITE_SUPABASE_URL=<your-supabase-url>
VITE_SUPABASE_ANON_KEY=<your-anon-key>
```

### 6. Running the Application

**Start the Backend:**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Frontend:**
```bash
cd frontend
npm run dev
```
