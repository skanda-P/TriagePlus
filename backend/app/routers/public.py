from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
import asyncio
from ..db.supabase_client import get_supabase

router = APIRouter(prefix="/api/v1", tags=["public"])


class FeedbackPayload(BaseModel):
    doctor_id: str
    appointment_id: Optional[str] = None
    stars: int
    comment: Optional[str] = None


@router.get("/specialties")
async def get_specialties():
    supabase = get_supabase()
    res = await asyncio.to_thread(
        supabase.table("specialty").select("id, name").execute
    )
    return res.data


@router.get("/doctors")
async def get_doctors(
    specialty_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    supabase = get_supabase()
    q = supabase.table("doctor").select("id, name, rating, avg_consult_min").order("rating", desc=True)
    if specialty_id:
        q = q.eq("specialty_id", specialty_id)
    q = q.range(offset, offset + limit - 1)
    res = await asyncio.to_thread(q.execute)
    return res.data


@router.post("/patient/feedback")
async def submit_feedback(payload: FeedbackPayload):
    supabase = get_supabase()
    
    # Validate appointment exists if provided
    if payload.appointment_id:
        appt_check = await asyncio.to_thread(
            supabase.table("appointment").select("id").eq("id", payload.appointment_id).execute
        )
        if not appt_check.data:
            return {"status": "error", "message": "Appointment not found"}
    
    res = await asyncio.to_thread(
        supabase.table("feedback").insert({
            "doctor_id": payload.doctor_id,
            "appointment_id": payload.appointment_id,
            "stars": payload.stars,
            "comment": payload.comment
        }).execute
    )
    return {"status": "success", "data": res.data}