"""Regression test for the new KG `_kg_condition_name_to_id` field-name bug.

Previously `_kg_condition_name_to_id` was built with `cinfo.get("name", "")`
but the loaded DDXPlus conditions dict uses the field `condition_name`, so the
mapping collapsed every entry onto the empty-string key, and
`_get_kg_department` always fell through to the keyword fallback. This test
asserts that a real DDXPlus condition name now maps to a non-None department.
"""
import pytest
from unittest.mock import patch

from app.core.triage_graph import _get_kg_department


def _fake_kg_with_ddxplus_schema():
    """Build a minimal KG singleton that mirrors the real DDXPlus JSON schema."""
    fake_kg = type("FakeKG", (), {})()
    # Mirror release_conditions.json: keys are condition ids, values use
    # `condition_name` (NOT `name`).
    fake_kg.conditions = {
        "C_001": {"condition_name": "Acute COPD exacerbation / infection", "severity": 3},
        "C_002": {"condition_name": "STEMI", "severity": 1},
        "C_003": {"name": "Legacy entry that uses 'name' instead", "severity": 2},
    }

    def get_condition_specialty(condition_id):
        # Mirror the real KG's mapping using condition_name lookup.
        cond = fake_kg.conditions.get(condition_id, {})
        name = (cond.get("condition_name") or cond.get("name") or "").lower()
        if "copd" in name:
            return "Pulmonology"
        if "stemi" in name:
            return "Cardiology"
        return "General Medicine"

    fake_kg.get_condition_specialty = get_condition_specialty
    return fake_kg


@pytest.mark.asyncio
async def test_kg_department_lookup_returns_non_none_for_ddxplus_condition_name():
    """A real DDXPlus condition name must map to a department, not silently None."""
    fake_kg = _fake_kg_with_ddxplus_schema()

    # Reset the module-level cache so the test doesn't see a stale mapping
    # from a prior test / import.
    import app.core.triage_graph as tg
    old_cache = tg._kg_condition_name_to_id
    tg._kg_condition_name_to_id = None

    try:
        with patch("app.core.triage_graph.get_kg", return_value=fake_kg):
            dept = await _get_kg_department("Acute COPD exacerbation / infection")
            assert dept == "Pulmonology", (
                "KG-based department lookup returned None or wrong value - the "
                "`condition_name` field fix in _get_kg_department has regressed."
            )

            # Also assert a name that needs case-insensitive / whitespace-stripped match.
            dept2 = await _get_kg_department("  STEMI  ")
            assert dept2 == "Cardiology"

            # Legacy KG entries using the `name` key should still resolve.
            dept3 = await _get_kg_department("Legacy entry that uses 'name' instead")
            assert dept3 == "General Medicine"
    finally:
        tg._kg_condition_name_to_id = old_cache
