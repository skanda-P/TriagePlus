from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid, asyncio, json
import sys, os
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../RAG")))
from ml_training.gemini_inference import infer_department_interactive, infer_department_final, DEPARTMENTS, _get_rag_components, check_emergency_llm

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

@router.websocket("/ws/diagnostics")
async def diagnostics_ws(websocket: WebSocket):
    await websocket.accept()
    _diagnostic_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _diagnostic_clients.remove(websocket)

# ── Mock Doctor Endpoints ──────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/auth/doctor/login")
async def doctor_login(req: LoginRequest):
    return {"access_token": "dummy_token"}

@router.get("/doctors/me/queue")
async def get_doctor_queue():
    return []

# ── SQLite session store ───────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT)")

_init_db()

def _get_session(session_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT data FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    return None

def _save_session(session_id: str, data: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions (id, data) VALUES (?, ?)", (session_id, json.dumps(data)))

# ── WebSocket endpoint (primary) ───────────────────────────────────────────────
@router.websocket("/ws/chat/{session_id}")
async def patient_ws(websocket: WebSocket, session_id: str):
    try:
        uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    state = _get_session(session_id)
    is_new = state is None

    if is_new:
        state = {
            "fsm_state": "NAME_ENTRY",
            "history": [], # Stores dicts like {"role": "user", "content": "..."}
            "turn_count": 0,
            "slots": None,
        }
        _save_session(session_id, state)
        asyncio.create_task(asyncio.to_thread(_get_rag_components))
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
            if fsm in ["INITIAL_SYMPTOM", "GEMINI_CONVERSATION"]:
                if await check_emergency_llm(content):
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

            elif fsm in ["INITIAL_SYMPTOM", "GEMINI_CONVERSATION"]:
                state["history"].append({"role": "user", "content": content})
                state["turn_count"] = state.get("turn_count", 0) + 1
                
                await websocket.send_json({"type": "typing", "content": True})
                
                try:
                    gemini_res = None
                    async for payload in infer_department_interactive(
                        state["history"], session_id,
                        turn_count=state["turn_count"],
                        known_slots=state.get("slots"),
                        patient_info={"gender": state.get("gender"), "age": state.get("age")}
                    ):
                        if payload["type"] == "stream_start":
                            await websocket.send_json({"type": "stream_start"})
                        elif payload["type"] == "stream_chunk":
                            await websocket.send_json({"type": "stream_chunk", "content": payload["content"]})
                        elif payload["type"] == "result":
                            gemini_res = payload["data"]

                    if not gemini_res:
                        raise Exception("No result from inference generator")

                    state["slots"] = gemini_res.get("slots", state.get("slots"))
                    
                    action = gemini_res.get("action", "ask")
                    
                    if action == "ask":
                        reply = gemini_res.get("reply", "")
                        state["history"].append({"role": "assistant", "content": reply})
                        state["fsm_state"] = "GEMINI_CONVERSATION"
                        
                        diag = {
                            "type": "diagnostic",
                            "session_id": session_id,
                            "query": content,
                            "top_k_a": gemini_res.get("top_k_a", []),
                            "top_k_b": [],
                            "prompt": gemini_res.get("prompt", ""),
                            "raw_response": gemini_res.get("raw_response", ""),
                            "department": "Interactive Turn",
                            "confidence": 1.0,
                            "latencies": gemini_res.get("latencies", {"embed": 0, "faiss": 0, "llm": 0, "total": 0})
                        }
                        asyncio.create_task(broadcast_diagnostic(diag))
                        
                        await websocket.send_json({"type": "typing", "content": False, "state": "GEMINI_CONVERSATION"})
                        
                    elif action == "complete":
                        summary = gemini_res.get("summary", content)
                        
                        dept, conf, urgency, diag = await asyncio.to_thread(
                            infer_department_final, summary, session_id,
                            patient_info={"gender": state.get("gender"), "age": state.get("age")}
                        )
                        asyncio.create_task(broadcast_diagnostic(diag))
                        
                        state["fsm_state"] = "RECOMMENDING"
                        state["department"] = dept
                        state["confidence"] = conf
                        state["urgency_score"] = urgency
                        
                        confidence_pct = int(conf * 100)
                        
                        if urgency >= 8:
                            color = "red"
                        elif urgency >= 5:
                            color = "orange"
                        elif urgency >= 3:
                            color = "yellow"
                        else:
                            color = "green"
                            
                        response = (
                            f"Thank you for sharing. Based on your symptoms, I recommend seeing our **{dept}** department.\n\n"
                            f"**Confidence:** {confidence_pct}%\n"
                            f"**Urgency Score:** {urgency}/10\n\n"
                            "⚠️ *This information is general in nature and does not constitute a medical diagnosis.*\n\n"
                            "Would you like to book an appointment? Just say **'book'**."
                        )
                        
                        await websocket.send_json({"type": "typing", "content": False})
                        await websocket.send_json({
                            "type": "message",
                            "content": response,
                            "state": "RECOMMENDING",
                            "meta": {
                                "specialty": dept,
                                "confidence": conf,
                                "confidence_label": f"{confidence_pct}%",
                                "urgency": urgency,
                                "triage_color": color,
                            },
                        })
                        
                except Exception as exc:
                    await websocket.send_json({"type": "typing", "content": False})
                    await websocket.send_json({
                        "type": "error",
                        "content": "I encountered an error analyzing your symptoms. Could you try rephrasing?",
                    })

            elif fsm == "RECOMMENDING":
                lower = content.lower()
                if any(w in lower for w in ["book", "yes", "appointment", "schedule"]):
                    state["fsm_state"] = "BOOKING"
                    await websocket.send_json({
                        "type": "message",
                        "content": f"Great! I'll help you book an appointment with {state.get('department', 'a specialist')}. Please visit our appointment booking portal.",
                        "state": "BOOKING",
                    })
                else:
                    await websocket.send_json({
                        "type": "message",
                        "content": "I understand. Let me know if you would like to book an appointment.",
                        "state": "RECOMMENDING",
                    })

            else:
                await websocket.send_json({
                    "type": "message",
                    "content": "Thank you for using TriagePlus. Stay healthy! 💚",
                    "state": fsm,
                })

            _save_session(session_id, state)

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
