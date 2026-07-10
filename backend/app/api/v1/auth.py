from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ...core.db import get_supabase
from supabase import Client

router = APIRouter(tags=["auth"])

class DoctorLoginRequest(BaseModel):
    email: str
    password: str

class PatientMagicLinkRequest(BaseModel):
    email: str

@router.post("/doctor/login")
def doctor_login(request: DoctorLoginRequest):
    """
    Login endpoint for doctors using email and password.
    Returns the Supabase auth session (which includes the JWT access_token).
    """
    supabase: Client = get_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    
    try:
        res = supabase.auth.sign_in_with_password({
            "email": request.email, 
            "password": request.password
        })
        # Optionally, verify they are in the doctor table here too
        # doctor_check = supabase.table("doctor").select("id").eq("auth_user_id", res.user.id).execute()
        # if not doctor_check.data:
        #     raise Exception("User is not registered as a doctor")
        
        return res.session
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@router.post("/patient/magiclink")
def patient_magiclink(request: PatientMagicLinkRequest):
    """
    Endpoint for patients to request a magic link (OTP) to their email.
    """
    supabase: Client = get_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Database client not initialized")
        
    try:
        res = supabase.auth.sign_in_with_otp({
            "email": request.email
        })
        return {"message": "Magic link / OTP sent successfully. Please check your email."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send magic link: {str(e)}")

@router.post("/patient/verify-otp")
def patient_verify_otp(email: str, token: str):
    """
    Endpoint for patients to verify the OTP they received.
    Returns the Supabase auth session.
    """
    supabase: Client = get_supabase()
    try:
        res = supabase.auth.verify_otp({
            "email": email,
            "token": token,
            "type": "magiclink"
        })
        return res.session
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"OTP Verification failed: {str(e)}")
