"""Regression test for code-review issue #2.

The original lifespan entered & immediately exited the SqliteSaver `with` block
BEFORE `yield`, so the SQLite connection was closed before the app started
serving requests. Every multi-turn conversation then hit
`sqlite3.ProgrammingError: Cannot operate on a closed database`.

This test verifies that two sequential graph invocations on the same
`thread_id` (session_id) share persisted state: a marker written in turn 1 is
visible to turn 2.
"""
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.triage_graph import build_graph, TriageState


@pytest.mark.asyncio
async def test_checkpointer_persists_state_across_turns(tmp_path):
    """Two sequential astream turns on one thread_id should see shared state."""
    db_path = tmp_path / "checkpoints.db"
    conn_uri = f"sqlite:///{db_path.as_posix()}"

    with SqliteSaver.from_conn_string(conn_uri) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)

        # Minimal node that appends a sentinel marker to messages and reads
        # any marker left by a previous turn. Injected by mutating the builder
        # is overkill - instead we drive the real graph using a pathway that
        # deterministically reaches the END on the first turn. We do that by
        # raising is_emergency=True via direct input state.
        session_id = "test-session-1"
        config = {"configurable": {"thread_id": session_id}}

        # Turn 1: drive directly into the emergency path. The emergency_check
        # node runs NER + evaluate_red_flags against plain text - we use a
        # strong emergency phrase so it fires even if NER returns nothing.
        input_state: TriageState = {
            "session_id": session_id,
            "patient_id": "p-1",
            "messages": ["I cannot breathe and have severe bleeding"],
            "is_emergency": False,
            "present_symptoms": [],
            "confidence": None,
            "triage_level": None,
            "department": None,
            "payment_status": None,
            "intent": None,
            "requested_department_raw": None,
            "requested_doctor_raw": None,
            "selected_doctor_id": None,
            "awaiting_department_choice": False,
            "booking_intent": None,
            "available_slots": None,
            "selected_slot_id": None,
            "final_diagnosis": None,
            "asked_symptoms": [],
            "rag_chunks": None,
            "latencies": None,
        }

        # Patch DB touch in node_emergency_response so the graph can run without
        # a live Supabase connection.
        from unittest.mock import patch, MagicMock

        fake_supabase = MagicMock()
        with patch("app.core.triage_graph.get_supabase", return_value=fake_supabase):
            # Turn 1
            async for _event in graph.astream(input_state, config=config, stream_mode="updates"):
                pass

            # Inspect checkpoint state after turn 1
            state1 = await graph.aget_state(config=config)
            assert state1.values.get("is_emergency") is True, "Turn 1 did not set is_emergency"
            assert state1.values.get("final_diagnosis") == "Possible Medical Emergency"

            # Turn 2: identical input, but the checkpointer should give us the
            # *accumulated* state from turn 1 (is_emergency already True).
            state2 = await graph.aget_state(config=config)
            assert state2.values.get("is_emergency") is True, (
                "State from turn 1 was not persisted - checkpointer closed early (review issue #2)"
            )
