"""
Doctor Portal APIs for slot management and availability

Endpoints:
- GET /api/doctor/templates - List doctor's templates
- POST /api/doctor/templates - Create new template
- GET /api/doctor/overrides - List availability overrides
- POST /api/doctor/apply-template - Apply template to dates
- POST /api/doctor/cancel-slot - Cancel a slot
"""

from fastapi import APIRouter, HTTPException, Depends
from supabase import create_client
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

router = APIRouter(prefix="/api/doctor", tags=["doctor_portal"])

def get_supabase():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

# --- Template Management ---

@router.get("/templates")
async def get_templates(doctor_id: str) -> List[Dict[str, Any]]:
    """Get all templates for a doctor"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("doctor_slot_templates").select("*").eq("doctor_id", doctor_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/templates")
async def create_template(doctor_id: str, name: str, slots: List[Dict]) -> Dict[str, Any]:
    """
    Create a new slot template
    
    slots format:
    [
        {"day_of_week": 1, "start_time": "09:00", "end_time": "17:00"},
        {"day_of_week": 2, "start_time": "09:00", "end_time": "17:00"}
    ]
    """
    supabase = get_supabase()
    
    try:
        template_data = {
            "doctor_id": doctor_id,
            "name": name,
            "slots": slots,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        response = supabase.table("doctor_slot_templates").insert(template_data).execute()
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Apply Template ---

@router.post("/apply-template")
async def apply_template(doctor_id: str, template_id: str, dates: List[str]) -> Dict[str, str]:
    """
    Apply a template to selected dates
    
    dates format: ["2025-01-15", "2025-01-16", ...]
    """
    supabase = get_supabase()
    
    try:
        # Get template
        template_response = supabase.table("doctor_slot_templates").select("*").eq("id", template_id).execute()
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = template_response.data[0]
        slots = template["slots"]
        
        # Generate slots for each date
        created_slots = []
        for date_str in dates:
            date_obj = datetime.fromisoformat(date_str)
            day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
            
            # Find matching slot for this day of week
            matching_slots = [s for s in slots if s["day_of_week"] == day_of_week or s["day_of_week"] == -1]  # -1 = all days
            
            for slot in matching_slots:
                slot_data = {
                    "doctor_id": doctor_id,
                    "date": date_str,
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"],
                    "is_booked": False,
                    "created_at": datetime.utcnow().isoformat()
                }
                created_slots.append(slot_data)
        
        # Bulk insert
        if created_slots:
            supabase.table("doctor_slots").insert(created_slots).execute()
        
        return {"status": "success", "slots_created": len(created_slots)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Overrides ---

@router.get("/overrides")
async def get_overrides(doctor_id: str) -> List[Dict[str, Any]]:
    """Get availability overrides (cancelled/limited slots)"""
    supabase = get_supabase()
    
    try:
        response = supabase.table("doctor_slot_overrides").select("*").eq("doctor_id", doctor_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Cancel Slot ---

@router.post("/cancel-slot")
async def cancel_slot(doctor_id: str, date: str, reason: str) -> Dict[str, str]:
    """
    Cancel a slot and notify patients with appointments
    """
    supabase = get_supabase()
    
    try:
        # Mark slot as unavailable
        supabase.table("doctor_slot_overrides").insert({
            "doctor_id": doctor_id,
            "date": date,
            "status": "unavailable",
            "reason": reason,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        # Find appointments on this date
        appointments = supabase.table("appointment").select("*").eq("doctor_id", doctor_id).eq("scheduled_date", date).execute()
        
        if appointments.data:
            # Notify patients
            for appointment in appointments.data:
                # TODO: Send email notification to patient
                # TODO: Send notification to admin
                
                # Update appointment status to cancelled_by_doctor
                supabase.table("appointment").update({
                    "status": "cancelled_by_doctor",
                    "cancellation_reason": reason,
                    "cancelled_at": datetime.utcnow().isoformat()
                }).eq("id", appointment["id"]).execute()
        
        return {"status": "success", "appointments_cancelled": len(appointments.data) if appointments.data else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
