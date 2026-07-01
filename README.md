# TriagePlus

TriagePlus is an AI-powered medical triage assistant. It provides fast intake, accurate specialty guidance, and instant appointment booking, removing the wait time for patients seeking medical help.

## Features

- **Smart Triage:** Powered by Gemini 2.5 Flash, it accurately infers the required hospital department based on patient symptoms.
- **Zero Wait Booking:** Bypasses phone queues to let patients book a confirmed slot instantly.
- **Privacy-First:** Secure, local session-based interaction that doesn't persist sensitive health data inappropriately.
- **Real-Time Chat:** WebSocket-based conversational interface for fluid communication.

## Project Structure

- `frontend/`: React + Vite web application built with Tailwind CSS.
- `backend/`: FastAPI application managing WebSockets, chat states, and integrations.
- `RAG/`: Machine learning scripts, FAISS indexes, and logic for AI model interactions (including Gemini).

## Setup & Installation

### 1. Backend Setup

Prerequisites: Python 3.10+

1. Navigate to the project root and create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Gemini API key in `backend/.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Run the FastAPI server:
   You can either run the `run.py` script from the project root:
   ```bash
   python run.py
   ```
   Or navigate to the backend directory and use uvicorn:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

### 2. Frontend Setup

Prerequisites: Node.js 18+

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

### Usage

Once both servers are running:
- Open the frontend at `http://localhost:5173`.
- Enter your name to start the triage process.
- The backend API and WebSockets are accessible at `http://localhost:8000`.

## Design System

The frontend follows a dedicated design system specified in `DESIGN.md`. It features a Canopy Green hero section, Coral call-to-actions, and Pastel feature cards, styled using Tailwind CSS and DM Sans.
