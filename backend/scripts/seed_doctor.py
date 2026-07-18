import os
import asyncio
from dotenv import load_dotenv

# Ensure we're in the backend dir to load the correct .env
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
load_dotenv(os.path.join(backend_dir, ".env"))

from app.db.supabase_client import get_supabase

async def seed():
    supabase = get_supabase()
    email = "doctor@hospital.com"
    password = "password123"
    
    print(f"Creating Auth User: {email}")
    try:
        # Check if user already exists
        auth_response = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        user_id = auth_response.user.id
        print(f"User created with ID: {user_id}")
    except Exception as e:
        print(f"Auth user might already exist or creation failed: {e}")
        # Try to log in to get the user ID
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            user_id = res.user.id
            print(f"User already existed. ID: {user_id}")
        except Exception as login_e:
            print(f"Failed to log in as well: {login_e}")
            return
            
    # Now check if doctor profile exists
    res = supabase.table("doctor").select("*").eq("auth_user_id", user_id).execute()
    if res.data:
        print("Doctor profile already exists in public.doctor table.")
        return
        
    print("Creating doctor profile in public.doctor table...")
    # Get a specialty to assign
    spec_res = supabase.table("specialty").select("id").limit(1).execute()
    if not spec_res.data:
        print("No specialties found! Please run your database migrations first.")
        return
        
    specialty_id = spec_res.data[0]["id"]
    
    doc_res = supabase.table("doctor").insert({
        "name": "Dr. Sarah Jenkins",
        "specialty_id": specialty_id,
        "rating": 4.9,
        "avg_consult_min": 15,
        "auth_user_id": user_id
    }).execute()
    
    print(f"Doctor profile created successfully! ID: {doc_res.data[0]['id']}")
    print("Seed complete.")

if __name__ == "__main__":
    asyncio.run(seed())
