import os
from supabase import create_client, Client
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

_supabase_client = None

def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise ValueError("Supabase URL and Service Role Key must be provided in .env")
        _supabase_client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _supabase_client
