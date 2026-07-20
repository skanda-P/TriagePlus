"""Pytest configuration: keeps the package importable and isolates runtime state.

These tests are designed to run in CI WITHOUT a live Supabase project, Ollama
server, or generated model artifacts. Anything that would touch those external
systems is patched via monkeypatch / unittest.mock.
"""
import os
import sys
from pathlib import Path

# Make `backend/app` importable regardless of pytest invocation cwd.
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Force the intake FSM SQLite DB to a temp path so tests never touch the real
# sessions.db in the repo root.
os.environ.setdefault("INTAKE_SESSIONS_DB_PATH", ":memory:")

# Minimal env stubs so importing app.db.supabase_client doesn't raise on
# Settings validation (we patch get_supabase / get_anon_supabase per-test).
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
