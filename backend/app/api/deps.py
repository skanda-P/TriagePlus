from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..core.db import get_supabase
from supabase import Client

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to verify a JWT token from Supabase.
    It calls Supabase's `get_user` using the provided token.
    If valid, returns the user object (which contains id, email, role, etc.).
    """
    token = credentials.credentials
    supabase: Client = get_supabase()
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        # Supabase Python client's get_user will verify the JWT token
        # If it's invalid, expired, or tampered with, it throws an exception or returns None.
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_response.user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_doctor_user(current_user = Depends(get_current_user)):
    """
    Dependency to ensure the current authenticated user is a doctor.
    Can be expanded to check database roles if you assign specific roles or
    check the doctor table.
    """
    # For now, simply verify they are authenticated and exist in the doctor table.
    # We will just verify they have an entry in `doctor` where auth_user_id = current_user.id
    supabase: Client = get_supabase()
    res = supabase.table("doctor").select("id, specialty_id").eq("auth_user_id", current_user.id).execute()
    
    if not res.data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not registered as a doctor",
        )
    return {"auth_user": current_user, "doctor_profile": res.data[0]}

def get_patient_user(current_user = Depends(get_current_user)):
    """
    Dependency to ensure the current authenticated user is a patient.
    """
    supabase: Client = get_supabase()
    res = supabase.table("patient").select("id").eq("auth_user_id", current_user.id).execute()
    
    if not res.data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not registered as a patient",
        )
    return {"auth_user": current_user, "patient_profile": res.data[0]}
