from fastapi import APIRouter, Depends, HTTPException, Header
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

@router.get("/me")
async def get_me(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    spec = supabase.table("specialty").select("name").eq("id", doctor["specialty_id"]).execute()
    doctor["specialty_name"] = spec.data[0]["name"] if spec.data else None
    return doctor

@router.get("/dashboard")
async def get_dashboard(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    # Aggregate KPIs
    # patients waiting:
    queue_res = supabase.table("queue_entry").select("id", count="exact").eq("doctor_id", doctor["id"]).execute()
    
    # critical count (triage_level 1-2):
    crit_res = supabase.table("appointment")\
        .select("id", count="exact")\
        .eq("status", "pending_slot")\
        .lte("triage_level", 2).execute()
        
    return {
        "patients_waiting": queue_res.count if queue_res else 0,
        "critical_cases": crit_res.count if crit_res else 0,
        "appointments_today": 0,
        "avg_wait_time": 15
    }

@router.get("/queue")
async def get_queue(doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    # Join appointment + patient + chat_session
    res = supabase.table("appointment")\
        .select("id, status, triage_level, department, chat_session_id, patient(name, age, gender)")\
        .eq("status", "pending_slot")\
        .order("triage_level", desc=False)\
        .execute()
    return res.data

@router.get("/appointments")
async def get_appointments(date: str = None, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    res = supabase.table("clinician_slot")\
        .select("id, start_time, status, appointment(id, status, patient(name))")\
        .eq("doctor_id", doctor["id"])\
        .execute()
    return res.data

@router.patch("/appointments/{id}")
async def update_appointment(id: str, payload: dict, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    supabase.table("appointment").update({"status": payload.get("status")}).eq("id", id).execute()
    return {"status": "success"}

@router.delete("/appointments/{id}")
async def delete_appointment(id: str, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    supabase.rpc('cancel_appointment', {'p_appointment_id': id}).execute()
    return {"status": "success"}

@router.get("/patients")
async def get_patients(search: str = None, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    q = supabase.table("patient").select("id, name, age, contact")
    if search:
        q = q.ilike("name", f"%{search}%")
    res = q.execute()
    return res.data

@router.get("/patients/{id}")
async def get_patient_detail(id: str, doctor: dict = Depends(verify_token)):
    supabase = get_supabase()
    res = supabase.table("patient").select("*, medical_history(*)").eq("id", id).execute()
    return res.data[0] if res.data else None
