from fastapi import APIRouter
from ..db.supabase_client import get_supabase

router = APIRouter(prefix="/api/v1", tags=["public"])

@router.get("/specialties")
async def get_specialties():
    supabase = get_supabase()
    res = supabase.table("specialty").select("id, name").execute()
    return res.data

@router.get("/doctors")
async def get_doctors(specialty_id: str = None):
    supabase = get_supabase()
    q = supabase.table("doctor").select("id, name, rating, avg_consult_min").order("rating", desc=True)
    if specialty_id:
        q = q.eq("specialty_id", specialty_id)
    res = q.execute()
    return res.data

@router.post("/patient/feedback")
async def submit_feedback(payload: dict):
    supabase = get_supabase()
    supabase.table("feedback").insert({
        "doctor_id": payload.get("doctor_id"),
        "appointment_id": payload.get("appointment_id"),
        "stars": payload.get("stars"),
        "comment": payload.get("comment")
    }).execute()
    return {"status": "success"}
