from fastapi import APIRouter, Depends
from ..deps import get_doctor_user
from ...core.db import get_supabase
from supabase import Client
import asyncio
import logging

logger = logging.getLogger("doctors_api")

router = APIRouter(tags=["doctors"])

@router.get("/me/queue")
async def get_doctor_queue(doctor=Depends(get_doctor_user)):
    supabase: Client = get_supabase()
    if not supabase:
        return []
        
    try:
        # Fetching completed chat sessions. 
        res = await asyncio.to_thread(
            supabase.table("chat_session")
            .select("*, patient(*)")
            .eq("status", "completed")
            .order("created_at", desc=False)
            .limit(50)
            .execute
        )
        data = res.data or []
        queue = []
        for i, row in enumerate(data):
            patient_name = "Unknown Patient"
            if row.get("patient") and isinstance(row["patient"], dict):
                patient_name = row["patient"].get("name", "Unknown Patient")
            
            queue.append({
                "position": i + 1,
                "patient_name": patient_name,
                "specialty": "General", 
                "ai_brief": row.get("summary", ""),
                "wait_time_minutes": (i + 1) * 10
            })
        return queue
    except Exception as e:
        logger.error(f"Failed to fetch doctor queue: {e}")
        return []
