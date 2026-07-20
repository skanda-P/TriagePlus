import json
import sqlite3
import os
import asyncio
from typing import Optional
from .db.supabase_client import get_supabase

# Simple SQLite store for fast intake FSM state tracking
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sessions.db"))


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intake_sessions (
            session_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            patient_id TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_session_state(session_id: str):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT state, patient_id FROM intake_sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"state": row[0], "patient_id": row[1]}
    return None


def update_fsm_session(session_id: str, state: str, patient_id: Optional[str] = None):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO intake_sessions (session_id, state, patient_id)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET 
            state = excluded.state,
            patient_id = coalesce(excluded.patient_id, intake_sessions.patient_id)
    ''', (session_id, state, patient_id))
    conn.commit()
    conn.close()


async def complete_intake(
    session_id: str, 
    name: str, 
    age: int, 
    gender: str, 
    contact: str
) -> Optional[str]:
    """Runs once, right after the miniform is submitted. Returns patient_id."""
    
    async def _inner():
        supabase = get_supabase()
        
        # Check if patient exists
        existing = supabase.table("patient").select("id").eq("contact", contact)\
            .order("created_at", desc=True).limit(1).execute()
            
        if existing.data:
            patient_id = existing.data[0]["id"]
        else:
            created = supabase.table("patient").insert({
                "name": name, "age": age, "gender": gender, "contact": contact
            }).execute()
            # Guard against empty data from RLS or other issues
            if not created.data:
                raise RuntimeError("Failed to create patient record")
            patient_id = created.data[0]["id"]

        # Create chat_session if it doesn't exist
        session_check = supabase.table("chat_session").select("id").eq("session_id", session_id).execute()
        if not session_check.data:
            supabase.table("chat_session").insert({
                "session_id": session_id, "patient_id": patient_id, "status": "in_progress"
            }).execute()

        # Move to INITIAL_SYMPTOM
        update_fsm_session(session_id, "INITIAL_SYMPTOM", patient_id=patient_id)
        return patient_id

    return await asyncio.to_thread(_inner)


# Initialize DB on import
init_db()