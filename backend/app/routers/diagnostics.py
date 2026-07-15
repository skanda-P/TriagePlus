import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .chat import _diagnostic_clients

router = APIRouter()

DEVELOPER_PASSWORD = os.getenv("DEVELOPER_PASSWORD", "devpass")

@router.websocket("/api/v1/ws/diagnostics")
async def diagnostics_websocket(websocket: WebSocket, token: str = None):
    if token != DEVELOPER_PASSWORD:
        await websocket.close(code=1008, reason="Policy Violation")
        return
        
    await websocket.accept()
    _diagnostic_clients.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _diagnostic_clients.remove(websocket)
    except Exception:
        if websocket in _diagnostic_clients:
            _diagnostic_clients.remove(websocket)
