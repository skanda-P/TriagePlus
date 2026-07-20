"""Regression test for code-review issue #1.

`complete_intake` previously declared `_inner` as `async def` and wrapped it
with `asyncio.to_thread(_inner)` — which returned the coroutine object unawait
without ever running the body. This test asserts that a real string/UUID
`patient_id` comes back and that `_inner` actually executed.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_complete_intake_returns_real_patient_id_string(tmp_path, monkeypatch):
    # Point the FSM SQLite DB at an isolated temp file for this test.
    monkeypatch.setenv("INTAKE_SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))

    # Re-import so the DB_PATH module-level constant picks up the env override.
    import importlib
    import app.intake_fsm as intake_fsm
    importlib.reload(intake_fsm)

    fake_supabase = MagicMock()

    # Track call ordering so we can return distinct data for the patient
    # insert (call #1) vs the chat_session insert (call #2 later).
    insert_call_count = {"n": 0}

    def fake_insert(*args, **kwargs):
        insert_call_count["n"] += 1
        # insert({...}) returns an object whose .execute() yields the data
        res = MagicMock()
        if insert_call_count["n"] == 1:
            # Patient insert -> new patient id
            res.execute.return_value.data = [{"id": "patient-uuid-123"}]
        else:
            # chat_session insert -> session id (value irrelevant to the test)
            res.execute.return_value.data = [{"id": "session-uuid-456"}]
        return res

    fake_supabase.table.return_value.insert.side_effect = fake_insert

    # Patient lookup by contact -> empty (no existing patient)
    patient_lookup = MagicMock()
    patient_lookup.data = []
    fake_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = patient_lookup

    # chat_session existence check -> empty (no existing session)
    session_check = MagicMock()
    session_check.data = []
    fake_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = session_check

    with patch.object(intake_fsm, "get_anon_supabase", return_value=fake_supabase):
        result = await intake_fsm.complete_intake(
            session_id="sess-1",
            name="John",
            age=30,
            gender="male",
            contact="john@example.com",
        )

    # Critical assertion: result is a real UUID-like string, NOT a coroutine object.
    # The original bug returned the coroutine object unawaited.
    assert isinstance(result, str), f"Expected str patient_id, got {type(result)}: {result!r}"
    assert result == "patient-uuid-123", (
        f"Expected patient_id from insert, got {result!r} - the mock may have "
        f"been wired incorrectly or complete_intake returned the wrong row."
    )

    # Verify the FSM state was actually persisted to the local SQLite DB
    # (this is what would have been skipped if _inner were still an
    # un-awaited async coroutine - review issue #1).
    state = intake_fsm.get_session_state("sess-1")
    assert state is not None, "FSM state was not persisted - _inner never ran"
    assert state["state"] == "INITIAL_SYMPTOM"
    assert state["patient_id"] == "patient-uuid-123"

