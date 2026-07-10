from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid, asyncio, json
import sys, os
import sqlite3

from ...core.triage_graph import graph_builder, db_path, load_rag_models
from ...core.db import get_supabase
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def check_emergency_keywords(content: str) -> bool:
    emergency_keywords = ["chest pain", "heart attack", "stroke", "can't breathe", "breathing difficulty", "severe bleeding", "unconscious", "suicide", "kill myself"]
    return any(kw in content.lower() for kw in emergency_keywords)

router = APIRouter()

# ── Diagnostics Event Bus ──────────────────────────────────────────────────────
_diagnostic_clients: list[WebSocket] = []

import logging
logger = logging.getLogger("chat_api")

async def broadcast_diagnostic(data: dict):
    logger.info(f"Broadcasting diagnostic to {len(_diagnostic_clients)} clients.")
    for client in _diagnostic_clients.copy():
        try:
            await client.send_json(data)
        except Exception as e:
            logger.error(f"Failed to broadcast diagnostic: {e}")
            try:
                _diagnostic_clients.remove(client)
            except ValueError:
                pass

def _is_authorized_doctor(token: str) -> bool:
    supabase = get_supabase()
    if not supabase or not token:
        return False
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user if user_response else None
        if not user:
            return False
        doctor_lookup = supabase.table("doctor").select("id").eq("auth_user_id", user.id).limit(1).execute()
        return bool(doctor_lookup.data)
    except Exception as exc:
        logger.error("Diagnostics auth failed: %s", exc)
        return False

@router.websocket("/ws/diagnostics")
async def diagnostics_ws(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token or not await asyncio.to_thread(_is_authorized_doctor, token):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _diagnostic_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _diagnostic_clients.remove(websocket)
        except ValueError:
            pass



# ── SQLite session store ───────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT)")

_init_db()

def _get_session_blocking(session_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT data FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    return None

async def _get_session(session_id: str) -> dict | None:
    return await asyncio.to_thread(_get_session_blocking, session_id)

def _save_session_blocking(session_id: str, data: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions (id, data) VALUES (?, ?)", (session_id, json.dumps(data)))

async def _save_session(session_id: str, data: dict):
    await asyncio.to_thread(_save_session_blocking, session_id, data)

# ── WebSocket endpoint (primary) ───────────────────────────────────────────────
@router.websocket("/ws/chat/{session_id}")
async def patient_ws(websocket: WebSocket, session_id: str):
    try:
        uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    state = await _get_session(session_id)
    is_new = state is None

    if is_new:
        state = {
            "fsm_state": "NAME_ENTRY",
            "history": [], # Stores dicts like {"role": "user", "content": "..."}
            "turn_count": 0,
            "slots": None,
        }
        await _save_session(session_id, state)
        asyncio.create_task(asyncio.to_thread(load_rag_models))
        try:
            await websocket.send_json({
                "type": "message",
                "content": "Welcome to TriagePlus! 👋 I'm your AI triage assistant. What's your full name?",
                "state": "NAME_ENTRY",
            })
        except Exception:
            return
    else:
        if state.get("fsm_state") != "NAME_ENTRY":
            try:
                await websocket.send_json({
                    "type": "sync_history",
                    "history": state.get("history", []),
                    "state": state.get("fsm_state")
                })
            except Exception as e:
                logger.error(f"Failed to sync history on reconnect: {e}")

    async def ping_loop():
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping_task = asyncio.create_task(ping_loop())

    try:
        async for data in websocket.iter_json():
            if data.get("type") == "pong":
                continue

            content = data.get("content", "").strip()
            if not content:
                continue

            fsm = state.get("fsm_state", "NAME_ENTRY")

            # Gate emergency check to only run during symptom phases
            if fsm in ["INITIAL_SYMPTOM", "LLM_CONVERSATION"]:
                if await check_emergency_keywords(content):
                    await websocket.send_json({
                        "type": "emergency",
                        "content": "⚠️ This sounds like a medical emergency. Please call 112 immediately or go to your nearest ER.",
                        "state": "EMERGENCY",
                    })
                    await websocket.close()
                    return

            if fsm == "NAME_ENTRY":
                if len(content) < 2 or content.isdigit():
                    await websocket.send_json({"type": "error", "content": "Please enter a valid name."})
                else:
                    state["patient_name"] = content
                    state["fsm_state"] = "AGE_ENTRY"
                    await websocket.send_json({"type": "message", "content": f"Hi {content}! What is your age?", "state": "AGE_ENTRY"})

            elif fsm == "AGE_ENTRY":
                if not content.isdigit() or not (0 < int(content) < 120):
                    await websocket.send_json({"type": "error", "content": "Please enter a valid age between 1 and 120."})
                else:
                    state["age"] = int(content)
                    state["fsm_state"] = "GENDER_ENTRY"
                    await websocket.send_json({"type": "message", "content": "Got it. What is your gender?", "state": "GENDER_ENTRY"})

            elif fsm == "GENDER_ENTRY":
                if len(content) < 3 or content.isdigit():
                    await websocket.send_json({"type": "error", "content": "Please enter a valid gender (e.g., Male, Female, Other)."})
                else:
                    state["gender"] = content
                    state["fsm_state"] = "PHONE_ENTRY"
                    await websocket.send_json({"type": "message", "content": "Got it. What is your phone number?", "state": "PHONE_ENTRY"})

            elif fsm == "PHONE_ENTRY":
                cleaned = ''.join(filter(str.isdigit, content))
                if len(cleaned) < 7:
                    await websocket.send_json({"type": "error", "content": "Please enter a valid phone number (at least 7 digits)."})
                else:
                    state["phone"] = cleaned
                    state["fsm_state"] = "INITIAL_SYMPTOM"
                    await websocket.send_json({"type": "message", "content": "Thank you. What brings you in today? Please describe your symptoms.", "state": "INITIAL_SYMPTOM"})

            elif fsm in ["INITIAL_SYMPTOM", "LLM_CONVERSATION", "RECOMMENDING", "BOOKING"]:
                state["history"].append({"role": "user", "content": content})
                state["turn_count"] = state.get("turn_count", 0) + 1
                
                await websocket.send_json({"type": "typing", "content": True})
                
                try:
                    # Initialize AsyncSqliteSaver
                    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
                        triage_app = graph_builder.compile(checkpointer=checkpointer)
                        
                        # Invoke LangGraph
                        config = {"configurable": {"thread_id": session_id}}
                        
                        # Ensure initial state is populated
                        if state["turn_count"] == 1:
                            initial_state = {
                                "age": state.get("age", 30),
                                "gender": "M" if str(state.get("gender", "")).lower().startswith("m") else "F",
                                "session_id": session_id,
                                "patient_id": None,
                                "present_symptoms": [],
                                "absent_symptoms": [],
                                "question_count": 0,
                                "is_emergency": False,
                                "messages": [HumanMessage(content=content)]
                            }
                            await triage_app.aupdate_state(config, initial_state)
                        else:
                            await triage_app.aupdate_state(config, {"messages": [HumanMessage(content=content)]})
                            
                        # Stream graph updates
                        async for event in triage_app.astream(None, config, stream_mode="updates"):
                            for node_name, node_state in event.items():
                                # Send Diagnostic Event
                                diagnostic_data = {
                                    "type": "diagnostic_update",
                                    "node": node_name,
                                    "state": {k: v for k, v in node_state.items() if k != "messages"} # Exclude full message history to save bandwidth
                                }
                                await broadcast_diagnostic(diagnostic_data)

                                if "messages" in node_state and node_state["messages"]:
                                    last_msg = node_state["messages"][-1].content
                                    await websocket.send_json({"type": "typing", "content": False})
                                    await websocket.send_json({"type": "message", "content": last_msg, "state": "LLM_CONVERSATION"})
                                    state["fsm_state"] = "LLM_CONVERSATION"
                                    state["history"].append({"role": "assistant", "content": last_msg})
                                    
                                if node_name == "process_payment" and node_state.get("payment_status") == "succeeded":
                                    await websocket.send_json({"type": "typing", "content": False})
                                    await websocket.send_json({
                                        "type": "triage_complete",
                                        "summary": "Appointment Booked and Paid",
                                    })
                except Exception as exc:
                    logger.error(f"LangGraph failed: {exc}", exc_info=True)
                    await websocket.send_json({"type": "typing", "content": False})
                    await websocket.send_json({"type": "error", "content": "I encountered an error. Could you try rephrasing?"})
                    
                await _save_session(session_id, state)

            else:
                await websocket.send_json({
                    "type": "message",
                    "content": "Thank you for using TriagePlus. Stay healthy! 💚",
                    "state": fsm,
                })

            await _save_session(session_id, state)

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
