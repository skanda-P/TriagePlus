"""
Booking APIs for patient appointment scheduling

Endpoints:
- GET /api/doctors - Get doctors by department with search
- GET /api/doctors/search - Live search doctors
- GET /api/slots - Get available slots for doctor
- POST /api/appointments - Create appointment
- GET /api/appointments/{id} - Get appointment details
"""

from fastapi import APIRouter, HTTPException, Query
from supabase import create_client
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

router = APIRouter(prefix="/api", tags=["booking"])

def get_supabase():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

# --- Doctor Discovery ---

@router.get("/doctors")
async def get_doctors_by_department(
    department: str,
    skip: int = Query(0),
    limit: int = Query(10)
) -> List[Dict[str, Any]]:
    """Get doctors in a specific department"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("doctors").select(
            "id, name, specialization, rating, bio, image_url"
        ).eq("specialization", department).range(skip, skip + limit - 1).execute()
        
        # Add available slots count
        doctors = response.data
        for doctor in doctors:
            slots_response = supabase.table("doctor_slots").select("id").eq(
                "doctor_id", doctor["id"]
            ).eq("is_booked", False).execute()
            
            doctor["available_slots_count"] = len(slots_response.data)
        
        return doctors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/doctors/search")
async def search_doctors(q: str = Query(..., min_length=2)) -> List[Dict[str, Any]]:
    """Live search for doctors by name"""
    supabase = get_supabase()
    
    try:
        # Use Postgres full-text search or simple ILIKE
        response = supabase.table("doctors").select(
            "id, name, specialization, rating, bio, image_url"
        ).ilike("name", f"%{q}%").execute()
        
        # Add available slots count
        doctors = response.data
        for doctor in doctors:
            slots_response = supabase.table("doctor_slots").select("id").eq(
                "doctor_id", doctor["id"]
            ).eq("is_booked", False).execute()
            
            doctor["available_slots_count"] = len(slots_response.data)
        
        return doctors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Slots ---

@router.get("/slots")
async def get_available_slots(
    doctor_id: str,
    month: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get available slots for a doctor in a month
    
    month format: "2025-01"
    """
    supabase = get_supabase()
    
    try:
        if not month:
            # Default to current month
            month = datetime.utcnow().strftime("%Y-%m")
        
        # Get slots for the month
        year, month_num = month.split("-")
        
        response = supabase.table("doctor_slots").select(
            "id, doctor_id, date, start_time, end_time, is_booked"
        ).eq("doctor_id", doctor_id).eq("is_booked", False).gte(
            "date", f"{year}-{month_num}-01"
        ).lt("date", f"{year}-{int(month_num) + 1:02d}-01").execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Appointments ---

@router.post("/appointments")
async def create_appointment(
    patient_id: str,
    doctor_id: str,
    slot_id: str,
    symptoms: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new appointment"""
    supabase = get_supabase()
    
    try:
        # Get slot details
        slot_response = supabase.table("doctor_slots").select("*").eq("id", slot_id).execute()
        if not slot_response.data:
            raise HTTPException(status_code=404, detail="Slot not found")
        
        slot = slot_response.data[0]
        
        # Create appointment
        appointment_data = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "slot_id": slot_id,
            "scheduled_date": slot["date"],
            "start_time": slot["start_time"],
            "end_time": slot["end_time"],
            "symptoms": symptoms,
            "notes": notes,
            "status": "scheduled",
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = supabase.table("appointment").insert(appointment_data).execute()
        appointment = response.data[0]
        
        # Mark slot as booked
        supabase.table("doctor_slots").update({"is_booked": True}).eq("id", slot_id).execute()
        
        return appointment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/appointments/{appointment_id}")
async def get_appointment(appointment_id: str) -> Dict[str, Any]:
    """Get appointment details"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("appointment").select("*").eq("id", appointment_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        appointment = response.data[0]
        
        # Add doctor and patient info
        doctor_response = supabase.table("doctors").select("*").eq("id", appointment["doctor_id"]).execute()
        patient_response = supabase.table("patients").select("*").eq("id", appointment["patient_id"]).execute()
        
        if doctor_response.data:
            appointment["doctor"] = doctor_response.data[0]
        if patient_response.data:
            appointment["patient"] = patient_response.data[0]
        
        return appointment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/appointments/{appointment_id}/cancel")
async def cancel_appointment(appointment_id: str, reason: str) -> Dict[str, str]:
    """Cancel an appointment"""
    supabase = get_supabase()
    
    try:
        # Get appointment
        appointment_response = supabase.table("appointment").select("*").eq("id", appointment_id).execute()
        if not appointment_response.data:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        appointment = appointment_response.data[0]
        
        # Cancel appointment
        supabase.table("appointment").update({
            "status": "cancelled_by_patient",
            "cancellation_reason": reason,
            "cancelled_at": datetime.utcnow().isoformat()
        }).eq("id", appointment_id).execute()
        
        # Free up slot
        supabase.table("doctor_slots").update({"is_booked": False}).eq("id", appointment["slot_id"]).execute()
        
        # TODO: Send notification emails
        
        return {"status": "success", "appointment_cancelled": appointment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
