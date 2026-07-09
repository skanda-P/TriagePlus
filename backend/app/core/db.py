import os
from supabase import create_client, Client

# The env vars should be loaded by main.py (load_dotenv) before this is imported
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

# Initialize the Supabase client using the service role key to bypass RLS for backend operations
# (Frontend will use anon key)
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase: Client = None

def get_supabase() -> Client:
    """Dependency to get the Supabase client"""
    return supabase
