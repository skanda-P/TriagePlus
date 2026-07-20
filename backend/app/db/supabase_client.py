import os
import threading
from typing import Optional
from supabase import create_client, Client
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_anon_key: str = ""


_settings = Settings()
_supabase_client_lock = threading.Lock()
_anon_client_lock = threading.Lock()
_supabase_client: Optional[Client] = None
_anon_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get the service-role client (bypasses RLS). Use only for trusted/internal operations."""
    global _supabase_client
    if _supabase_client is None:
        with _supabase_client_lock:
            if _supabase_client is None:
                if not _settings.supabase_url or not _settings.supabase_service_role_key:
                    raise ValueError("Supabase URL and Service Role Key must be provided in .env")
                _supabase_client = create_client(_settings.supabase_url, _settings.supabase_service_role_key)
    return _supabase_client


def get_anon_supabase() -> Client:
    """Get the anon-key client (enforces RLS). Use for patient-facing operations."""
    global _anon_client
    if _anon_client is None:
        with _anon_client_lock:
            if _anon_client is None:
                if not _settings.supabase_url or not _settings.supabase_anon_key:
                    raise ValueError("Supabase URL and Anon Key must be provided in .env")
                _anon_client = create_client(_settings.supabase_url, _settings.supabase_anon_key)
    return _anon_client