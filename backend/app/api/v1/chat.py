from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid, asyncio, json
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../RAG")))
from ml_training.gemini_inference import infer_department_interactive, infer_department_final, DEPARTMENTS

router = APIRouter()

# ── Diagnostics Event Bus ──────────────────────────────────────────────────────
_diagnostic_clients: list[WebSocket] = []

async def broadcast_diagnostic(data: dict):
    for client in _diagnostic_clients.copy():
        try:
            await client.send_json(data)
        except Exception:
            _diagnostic_clients.remove(client)

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

# ── In-memory session store ────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}

_EMERGENCY_PHRASES = [
    "heart attack", "chest pain", "can't breathe", "cannot breathe",
    "stroke", "unconscious", "not breathing", "severe bleeding",
    "overdose", "suicide", "dying", "call 112", "ambulance",
]

def check_emergency(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _EMERGENCY_PHRASES)

# ── WebSocket endpoint (primary) ───────────────────────────────────────────────
@router.websocket("/ws/chat/{session_id}")
async def patient_ws(websocket: WebSocket, session_id: str):
    try:
        uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    if session_id not in _sessions:
        _sessions[session_id] = {
            "fsm_state": "NAME_ENTRY",
            "history": [] # Stores dicts like {"role": "user", "content": "..."}
        }

    state = _sessions[session_id]

    if state.get("fsm_state") == "NAME_ENTRY":
        try:
            await websocket.send_json({
                "type": "message",
                "content": "Welcome to TriagePlus! 👋 I'm your AI triage assistant. What's your full name?",
                "state": "NAME_ENTRY",
            })
        except Exception:
            return

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

            if check_emergency(content):
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
                    state["fsm_state"] = "PHONE_ENTRY"
                    await websocket.send_json({"type": "message", "content": f"Hi {content}! What is your phone number?", "state": "PHONE_ENTRY"})

            elif fsm == "PHONE_ENTRY":
                # Very basic validation: only digits and lengths typical of phone numbers
                cleaned = ''.join(filter(str.isdigit, content))
                if len(cleaned) < 7:
                    await websocket.send_json({"type": "error", "content": "Please enter a valid phone number (at least 7 digits)."})
                else:
                    state["phone"] = cleaned
                    state["fsm_state"] = "AGE_ENTRY"
                    await websocket.send_json({"type": "message", "content": "Got it. And what is your age?", "state": "AGE_ENTRY"})

            elif fsm == "AGE_ENTRY":
                if not content.isdigit() or not (0 < int(content) < 120):
                    await websocket.send_json({"type": "error", "content": "Please enter a valid age between 1 and 120."})
                else:
                    state["age"] = int(content)
                    state["fsm_state"] = "INITIAL_SYMPTOM"
                    await websocket.send_json({"type": "message", "content": "Thank you. What brings you in today? Please describe your symptoms.", "state": "INITIAL_SYMPTOM"})

            elif fsm in ["INITIAL_SYMPTOM", "GEMINI_CONVERSATION"]:
                state["history"].append({"role": "user", "content": content})
                
                # Tell frontend we are typing
                await websocket.send_json({"type": "typing", "content": True})
                
                try:
                    # Run conversational inference
                    gemini_res = await asyncio.to_thread(infer_department_interactive, state["history"], session_id)
                    
                    action = gemini_res.get("action", "ask")
                    
                    if action == "ask":
                        reply = gemini_res.get("reply", "Can you tell me more?")
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
                        
                        await websocket.send_json({"type": "typing", "content": False})
                        await websocket.send_json({"type": "message", "content": reply, "state": "GEMINI_CONVERSATION"})
                        
                    elif action == "complete":
                        summary = gemini_res.get("summary", content)
                        
                        # Run final triage inference
                        dept, conf, urgency, diag = await asyncio.to_thread(infer_department_final, summary, session_id)
                        asyncio.create_task(broadcast_diagnostic(diag))
                        
                        state["fsm_state"] = "RECOMMENDING"
                        state["department"] = dept
                        state["confidence"] = conf
                        state["urgency_score"] = urgency
                        
                        confidence_pct = int(conf * 100)
                        
                        # Set color based on urgency
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

            _sessions[session_id] = state

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
