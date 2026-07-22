import asyncio
import hmac
import json
import logging
import os
from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Header
from ..intake_fsm import complete_intake, get_session_state
from typing import Optional

router = APIRouter()

# Global set of diagnostic clients with lock for thread-safe access
_diagnostic_clients = set()
_diagnostic_clients_lock = asyncio.Lock()
_MAX_DIAGNOSTIC_CLIENTS = 10


class IntakeFormPayload(BaseModel):
    type: str = "intake_form"
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(M|F|other)$")
    contact: str = Field(..., min_length=5, max_length=50)
    
    @field_validator("contact")
    @classmethod
    def validate_contact(cls, v: str) -> str:
        # Basic email or phone validation
        if "@" in v:
            if not v.count("@") == 1 or "." not in v.split("@")[1]:
                raise ValueError("Invalid email format")
        else:
            # Phone - basic check for digits
            digits = "".join(c for c in v if c.isdigit())
            if len(digits) < 7:
                raise ValueError("Invalid phone number")
        return v


class ChatMessagePayload(BaseModel):
    type: str = "message"
    content: str = Field(..., min_length=1, max_length=5000)


def verify_dev_password(token: str) -> bool:
    """Verify developer password using constant-time comparison."""
    dev_password = os.getenv("DEVELOPER_PASSWORD")
    if not dev_password:
        logging.warning("DEVELOPER_PASSWORD not set - diagnostics access disabled")
        return False
    return hmac.compare_digest(token, dev_password)


@router.websocket("/api/v1/ws/diagnostics")
async def diagnostics_websocket(
    websocket: WebSocket,
    authorization: str = Header(None, description="Bearer token for diagnostics access")
):
    """WebSocket endpoint for diagnostic dashboard to receive real-time graph updates."""
    # Verify token before accepting - use Authorization header instead of query param
    if not authorization or not authorization.startswith("Bearer "):
        await websocket.close(code=1008, reason="Missing or invalid authorization header")
        return
    
    token = authorization.split(" ", 1)[1]
    try:
        if not verify_dev_password(token):
            await websocket.close(code=1008, reason="Invalid token")
            return
    except RuntimeError as e:
        await websocket.close(code=1011, reason=str(e))
        return
        
    # Enforce max concurrent diagnostic connections
    async with _diagnostic_clients_lock:
        if len(_diagnostic_clients) >= _MAX_DIAGNOSTIC_CLIENTS:
            await websocket.close(code=1013, reason="Too many diagnostic connections")
            return
        _diagnostic_clients.add(websocket)

    await websocket.accept()
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Diagnostics connected"
        }))
        
        # Keep connection alive, handle incoming messages (ping/pong)
        while True:
            try:
                data = await websocket.receive_text()
                payload = json.loads(data)
                if payload.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.warning(f"Diagnostics WS error: {e}")
    finally:
        async with _diagnostic_clients_lock:
            _diagnostic_clients.discard(websocket)


async def ping_task(websocket: WebSocket):
    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_text('{"type": "ping"}')
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

# Prefixes whose text after "PREFIX: " is real patient-facing content —
# the prefix just tags which node emitted it. Strip it, still send it.
CONTENT_SENTINEL_PREFIXES = {
    "QUESTION", "DIAGNOSIS_EXPLANATION", "PROMPT_DEPARTMENT",
    "PROMPT_DEPARTMENT_RETRY", "PROMPT_BOOKING", "PAYMENT_SUCCESS",
}

# Pure internal markers with no patient-facing payload — always suppressed.
SILENT_SENTINELS = {"SLOTS_OFFERED", "SYSTEM_FALLBACK", "SLOT_CONFIRMED"}


def _visible_content(msg: str) -> Optional[str]:
    """Text to send to the patient for a graph message, or None to suppress it."""
    if not isinstance(msg, str):
        return None
    prefix, sep, rest = msg.partition(":")
    prefix = prefix.strip()
    if sep and prefix in CONTENT_SENTINEL_PREFIXES:
        return rest.strip()
    if prefix in SILENT_SENTINELS:
        return None
    return msg

async def _broadcast_diagnostics(event: dict):
    """Fire-and-forget broadcast to all diagnostic clients with a short timeout."""
    async with _diagnostic_clients_lock:
        clients = list(_diagnostic_clients)
    if not clients:
        return
    # Send concurrently; individual failures are swallowed.
    async def _send(ws):
        try:
            await asyncio.wait_for(
                ws.send_text(json.dumps({
                    "type": "diagnostic_update",
                    "node": list(event.keys())[0] if event else "unknown",
                    "state": event[list(event.keys())[0]] if event else {}
                })),
                timeout=0.5,
            )
        except Exception:
            pass
    await asyncio.gather(*[_send(ws) for ws in clients], return_exceptions=True)


async def _process_graph_event(websocket: WebSocket, event: dict, msg: str):
    """Process a single graph event and send appropriate responses to the client."""
    # Broadcast to diagnostics (non-blocking)
    asyncio.create_task(_broadcast_diagnostics(event))

    for node_name, node_state in event.items():
        if node_state.get("is_emergency"):
            await websocket.send_text(json.dumps({
                "type": "emergency",
                "content": node_state["messages"][-1] if node_state.get("messages") else "Emergency detected",
            }))
            return
        
        # Send metadata if available
        if node_state.get("department"):
            await websocket.send_text(json.dumps({
                "type": "message",
                "state": node_state.get("triage_level"),
                "meta": {
                    "specialty": node_state.get("department"),
                    "confidence": node_state.get("confidence"),
                    "confidence_label": f"{int(node_state.get('confidence', 0) * 100)}%" if node_state.get("confidence") else None,
                    "urgency": node_state.get("triage_level"),
                    "triage_level": node_state.get("triage_level"),
                    "triage_color": "red" if node_state.get("triage_level") == 1 else "orange" if node_state.get("triage_level") in [2, 3] else "green"
                }
            }))
        
        # if "messages" in node_state and len(node_state["messages"]) > 0:
        #     # Send latest message, but filter out internal sentinels
        #     last_msg = node_state["messages"][-1]
        #     if last_msg != msg and last_msg.strip() and not _is_sentinel(last_msg):
        #         await websocket.send_text(json.dumps({
        #             "type": "message",
        #             "content": last_msg
        #         }))

        if "messages" in node_state and len(node_state["messages"]) > 0:
            last_msg = node_state["messages"][-1]
            if last_msg != msg:
                visible = _visible_content(last_msg)
                if visible:
                    await websocket.send_text(json.dumps({
                        "type": "message",
                        "content": visible
                    }))
                    
        if node_state.get("available_slots"):
            await websocket.send_text(json.dumps({
                "type": "message",
                "content": "Available appointment slots:"
            }))
            for slot in node_state["available_slots"]:
                await websocket.send_text(json.dumps({
                    "type": "message",
                    "content": f"Slot: {slot.get('start_time', 'TBD')}"
                }))


# Graph reference set by main.py lifespan
_graph = None


def set_graph(graph):
    global _graph
    _graph = graph


@router.websocket("/api/v1/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    ping = asyncio.create_task(ping_task(websocket))
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle each message independently - don't let one bad message kill the connection
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "content": "Invalid JSON payload"}))
                continue
            
            # Handle intake form
            if payload.get("type") == "intake_form":
                try:
                    intake_data = IntakeFormPayload(**payload)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "content": f"Invalid intake form: {str(e)}"}))
                    continue
                    
                try:
                    await complete_intake(
                        session_id,
                        intake_data.name,
                        intake_data.age,
                        intake_data.gender,
                        intake_data.contact
                    )
                except Exception as e:
                    logging.error(f"Intake completion error: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "content": "Failed to complete intake. Please try again."}))
                    continue
                    
                await websocket.send_text(json.dumps({
                    "type": "message", 
                    "content": "Thank you. What brings you in today? (Or you can choose an option below)",
                    "chips": ["Describe my symptoms", "Book by department", "Book with a specific doctor"]
                }))
                continue
            
            # Message
            if payload.get("type") == "message":
                try:
                    msg_data = ChatMessagePayload(**payload)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "content": f"Invalid message format: {str(e)}"}))
                    continue
                
                msg = msg_data.content
                
                # Check FSM state - wrap sync call in thread to avoid blocking event loop
                state = await asyncio.to_thread(get_session_state, session_id)
                if not state or state["state"] != "INITIAL_SYMPTOM":
                    await websocket.send_text(json.dumps({"type": "error", "content": "Please complete intake first."}))
                    continue
                    
                await websocket.send_text(json.dumps({"type": "typing"}))
                
                # Invoke LangGraph - checkpointer restores persisted state on subsequent messages
                config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
                
                # Provide required fields for first message; checkpointer merges with persisted state on subsequent messages
                input_state = {
                    "session_id": session_id,
                    "patient_id": state["patient_id"],
                    "messages": [msg],
                    "is_emergency": False,
                }
                
                try:
                    # Send stream_start BEFORE streaming
                    await websocket.send_text(json.dumps({"type": "stream_start"}))
                    
                    async for event in _graph.astream(input_state, config=config, stream_mode="updates"):
                        await _process_graph_event(websocket, event, msg)
                except Exception as e:
                    logging.error(f"Graph streaming error: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "content": f"Processing error: {str(e)}"}))
                
                # Mark typing as complete
                await websocket.send_text(json.dumps({"type": "stream_end"}))
                            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"Chat WS error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": "An error occurred."}))
        except Exception:
            pass
    finally:
        ping.cancel()
        try:
            await ping
        except asyncio.CancelledError:
            pass
        except Exception:
            pass