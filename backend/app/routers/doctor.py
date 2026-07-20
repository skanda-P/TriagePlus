from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from typing import Optional
import asyncio
from datetime import datetime
from ..db.supabase_client import get_supabase

router = APIRouter(prefix="/api/v1/doctor", tags=["doctor"])

async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    
    supabase = get_supabase()
    try:
        user_res = supabase.auth.get_user(token)
        if not user_res or not user_res.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = user_res.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
        
    res = supabase.table("doctor").select("*").eq("auth_user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=403, detail="Doctor profile not found for this user")
    return res.data[0]


class UpdateAppointmentStatus(BaseModel):
    status: str


async def _verify_appointment_ownership(appointment_id: str, doctor_id: str) -> bool:
    supabase = get_supabase()
    res = await asyncio.to_thread(
        lambda: supabase.table("queue_entry")\
            .select("id").eq("appointment_id", appointment_id).eq("doctor_id", doctor_id).execute()
    )
    return bool(res.data)


@router.get("/me")
async def get_me(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    spec = supabase.table("specialty").select("name").eq("id", doctor["specialty_id"]).execute()
    doctor["specialty_name"] = spec.data[0]["name"] if spec.data else None
    return doctor


@router.get("/dashboard")
async def get_dashboard(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    # Patients waiting in queue for this doctor
    queue_res = supabase.table("queue_entry").select("id", count="exact").eq("doctor_id", doctor["id"]).execute()
    
    # Critical cases (triage_level 1-2) for this doctor's queue
    crit_res = supabase.table("queue_entry")\
        .select("id", count="exact").eq("doctor_id", doctor["id"])\
        .lte("appointment.triage_level", 2).execute()
        
    # Appointments today from queue
    today_str = datetime.now().strftime("%Y-%m-%d")
    appt_today_res = supabase.table("queue_entry")\
        .select("id", count="exact").eq("doctor_id", doctor["id"])\
        .eq("appointment_date", today_str).execute()
        
    # Average wait time
    wait_res = supabase.table("queue_entry").select("est_wait_min").eq("doctor_id", doctor["id"]).execute()
    avg_wait = sum(e["est_wait_min"] for e in wait_res.data) / len(wait_res.data) if wait_res.data else 15

    return {
        "patients_waiting": queue_res.count if queue_res.count else 0,
        "critical_cases": crit_res.count if crit_res.count else 0,
        "appointments_today": appt_today_res.count if appt_today_res.count else 0,
        "avg_wait_time": int(avg_wait)
    }


@router.get("/appointments")
async def get_appointments(
    date: Optional[str] = Query(None), 
    doctor: dict = Depends(verify_token)
):
    supabase = get_supabase()
    # Get appointments via doctor's queue entries
    q = supabase.table("queue_entry")\
        .select("appointment_id, position, appointment!inner(id, status, triage_level, department, patient(name, age, gender))")\
        .eq("doctor_id", doctor["id"])
    if date:
        q = q.eq("appointment_date", date)
    res = q.order("position").execute()
    return [row.get("appointment") for row in res.data if row.get("appointment")]


@router.patch("/appointments/{id}")
async def update_appointment(
    id: str, 
    payload: UpdateAppointmentStatus, 
    doctor: dict = Depends(verify_token)
):
    supabase = get_supabase()
    if not await _verify_appointment_ownership(id, doctor["id"]):
        raise HTTPException(status_code=403, detail="Appointment not found or access denied")
    supabase.table("appointment").update({"status": payload.status}).eq("id", id).execute()
    return {"status": "success"}


@router.delete("/appointments/{id}")
async def delete_appointment(id: str, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    if not await _verify_appointment_ownership(id, doctor["id"]):
        raise HTTPException(status_code=403, detail="Appointment not found or access denied")
    supabase.rpc('cancel_appointment', {'p_appointment_id': id}).execute()
    return {"status": "success"}


@router.get("/patients")
async def get_patients(
    search: Optional[str] = Query(None), 
    doctor: dict = Depends(verify_token)
):
    supabase = get_supabase()
    # Only return patients who have queue entries for this doctor
    q = supabase.table("patient")\
        .select("id, name, age, contact, queue_entry!inner(appointment_id, doctor_id)")\
        .eq("queue_entry.doctor_id", doctor["id"])
    if search:
        q = q.ilike("name", f"%{search}%")
    res = q.execute()
    # Deduplicate
    seen = set()
    unique_patients = []
    for p in res.data:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_patients.append(p)
    return unique_patients


@router.get("/patients/{id}")
async def get_patient_detail(id: str, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    # Verify this patient has a queue entry for this doctor
    appt_check = supabase.table("queue_entry")\
        .select("id")\
        .eq("doctor_id", doctor["id"])\
        .eq("appointment.patient_id", id)\
        .execute()
    if not appt_check.data:
        raise HTTPException(status_code=403, detail="Patient not found or access denied")
        
    res = supabase.table("patient").select("*, medical_history(*)").eq("id", id).execute()
    return res.data[0] if res.data else None


# Slot management
@router.get("/slots")
async def get_doctor_slots(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    res = supabase.table("clinician_slot")\
        .select("id, start_time, status")\
        .eq("doctor_id", doctor["id"])\
        .order("start_time")\
        .execute()
    return res.data


@router.post("/slots")
async def create_slot(
    start_time: str,
    doctor: dict = Depends(verify_token)
):
    supabase = get_supabase()
    res = supabase.table("clinician_slot").insert({
        "doctor_id": doctor["id"],
        "start_time": start_time,
        "status": "open"
    }).execute()
    return res.data[0]


@router.delete("/slots/{slot_id}")
async def cancel_slot(slot_id: str, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    res = supabase.table("clinician_slot").select("doctor_id").eq("id", slot_id).execute()
    if not res.data or res.data[0]["doctor_id"] != doctor["id"]:
        raise HTTPException(status_code=403, detail="Slot not found or access denied")
    supabase.table("clinician_slot").update({"status": "cancelled"}).eq("id", slot_id).execute()
    return {"status": "success"}


# Queue endpoint
@router.get("/queue")
async def get_queue(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    res = supabase.table("queue_entry")\
        .select("id, position, est_wait_min, appointment_date, appointment(id, status, triage_level, department, patient(name, age, gender))")\
        .eq("doctor_id", doctor["id"])\
        .order("position")\
        .execute()
    return res.data