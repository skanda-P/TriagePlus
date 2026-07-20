"""Regression test for code-review issue #5.

Doctor dashboard queries originally filtered on `appointment.triage_level` /
`appointment.patient_id` WITHOUT embedding the `appointment` relation in
`select(...)`, which PostgREST silently ignores — leading to potential IDOR
(cross-tenant data exposure).

This test exercises the Supabase query builder shape used by
`get_patient_detail` against a mocked client: it asserts the select string
EMBEDS `appointment!inner(patient_id)` (the fix), and that a 403 is raised when
the ownership check returns no rows.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.routers.doctor import get_patient_detail


def _make_async_supabase_mock(*, appt_check_data):
    """Returns a MagicMock supabase client whose `.table().select().eq().execute()`
    chain for the ownership check returns `appt_check_data`.

    Captures args on the chain so the test can assert on `select()` call args.
    """
    fake = MagicMock()

    # Build the chain used by get_patient_detail's ownership check:
    # supabase.table("queue_entry").select("id, appointment!inner(patient_id)")
    #   .eq("doctor_id", ...).eq("appointment.patient_id", ...).execute()
    table_mock = MagicMock()
    select_mock = MagicMock()
    eq1 = MagicMock()
    eq2 = MagicMock()
    exec_mock = MagicMock()

    exec_mock.data = appt_check_data

    fake.table.return_value = table_mock
    table_mock.select.return_value = select_mock
    # First eq("doctor_id"), second eq("appointment.patient_id") - chained in
    # either order; both chain to the next mock that ends at execute().
    select_mock.eq.return_value = eq1
    eq1.eq.return_value = eq2
    eq2.execute.return_value = exec_mock
    # In case the test path returns early with a single eq:
    select_mock.eq.return_value.execute.return_value = exec_mock

    # Make asyncio.to_thread(lambda: <call>) work: capture the call expression
    # and resolve it through the mock chain. We do this by making each chained
    # method return itself; the lambda in the route then calls .execute().
    return fake


@pytest.mark.asyncio
async def test_get_patient_detail_denies_access_when_no_queue_entry():
    """Doctor A querying a patient they have NO queue entry for -> 403."""
    fake_supabase = _make_async_supabase_mock(appt_check_data=[])

    with patch("app.routers.doctor.get_supabase", return_value=fake_supabase):
        with pytest.raises(HTTPException) as exc_info:
            await get_patient_detail(
                id="patient-belonging-to-doctor-b",
                doctor={"id": "doctor-A-id", "specialty_id": "x"},
            )

    assert exc_info.value.status_code == 403, (
        "Doctor isolation check returned wrong status code - regression of code-review issue #5."
    )


@pytest.mark.asyncio
async def test_get_patient_detail_uses_inner_join_on_appointment():
    """Assert the ownership select string embeds `appointment!inner(patient_id)`
    so PostgREST actually filters on the related column."""
    fake_supabase = _make_async_supabase_mock(appt_check_data=[])

    with patch("app.routers.doctor.get_supabase", return_value=fake_supabase):
        try:
            await get_patient_detail(
                id="any-patient", doctor={"id": "doctor-A-id", "specialty_id": "x"}
            )
        except HTTPException:
            pass  # We expect 403; we just want to capture the select() call.

    # Inspect the .select() call that was made for the ownership check.
    select_call_args = fake_supabase.table.return_value.select.call_args
    assert select_call_args is not None, "select() was never called - mock chain is broken"
    select_str = select_call_args.args[0]
    assert "appointment!inner(patient_id)" in select_str, (
        f"Ownership check select() must embed `appointment!inner(patient_id)` "
        f"or PostgREST will silently ignore the join filter (review issue #5). "
        f"Got: {select_str!r}"
    )
