# TriagePlus — Backend Architecture Build Prompt

**Project:** TriagePlus · IIT Dharwad Summer of Innovation · Team "Hardly Human"
**Depends on:** `01_Database_Schema_Build_Prompt.md` (table/RPC names are fixed contracts), `03_AI_Engine_Build_Prompt.md` (LangGraph node behavior)

## 1. Overview

The TriagePlus backend is an asynchronous, performance-oriented Python server (FastAPI + Uvicorn) that handles long-running, stateful WebSocket connections. It bridges a lightweight finite-state machine for patient intake with a LangGraph state machine for AI-driven triage, and exposes a REST API for the doctor-facing portal — alongside integrations with Supabase (PostgreSQL) and real-time developer diagnostics.

Two conventions are binding end-to-end and must be respected in every layer this document touches:
- **`triage_level`**: integer 1–5, ESI convention (1 = most critical). This backend never inverts, rescales, or renames this field.
- **`confidence`**: float in `[0, 1]`. Never multiplied by 100 in this layer — that happens only in frontend display code.

## 2. Tech Stack & Core Libraries

- **Framework:** FastAPI, Python 3.10+
- **Database:** `supabase-py` (service-role key), server-side only
- **State Machine:** LangGraph (`langgraph>=0.2.0`, `langgraph-checkpoint-sqlite`)
- **Real-Time:** `websockets`, `asyncio`
- **Auth verification:** Supabase JWT validation for doctor-facing REST routes

## 3. Application Initialization & Resource Management

FastAPI lifecycle hooks (`@app.on_event("startup")` in `main.py`) manage heavy resources so they never block the event loop during live requests:

- **Graph Compilation:** the LangGraph app (`graph_builder`) is compiled exactly once, globally, at startup.
- **Persistent Checkpointer:** `AsyncSqliteSaver` connected to `triage_checkpoints.sqlite`, with `PRAGMA journal_mode=WAL` for concurrent access.
- **RAG & ML Warmup:** `load_rag_models()` runs in a background thread to preload FAISS indices and the embedding model, so the first chat session isn't stalled by model weight loading.
- **Stale-hold reaper:** an `asyncio.create_task` loop calls `supabase.rpc('release_stale_holds', {'p_max_age_minutes': 10})` every 60 seconds and logs the count of slots released.

## 4. Session Identity

`session_id` is the single identifier used consistently across the WebSocket path param, the intake FSM store, the LangGraph `thread_id`, and `chat_session.session_id` — nothing generates its own id for the same conversation.

### 4.1 Intake FSM

A hand-rolled SQLite store (`sessions.db`) tracks the deterministic pre-triage intake flow, bypassing LangGraph entirely for instant, zero-latency form-filling:

`NAME_ENTRY → AGE_ENTRY → GENDER_ENTRY → PHONE_ENTRY → INITIAL_SYMPTOM`

The instant `PHONE_ENTRY` completes — **before** control passes to LangGraph — the backend resolves the patient identity and creates the triage session record:

```python
async def complete_intake(session_id: str, name: str, age: int, gender: str, contact: str) -> str:
    """Runs once, right after PHONE_ENTRY. Returns patient_id."""
    existing = await supabase.table("patient").select("id").eq("contact", contact) \
        .order("created_at", desc=True).limit(1).execute()
    if existing.data:
        patient_id = existing.data[0]["id"]
    else:
        created = await supabase.table("patient").insert({
            "name": name, "age": age, "gender": gender, "contact": contact
        }).execute()
        patient_id = created.data[0]["id"]

    await supabase.table("chat_session").insert({
        "session_id": session_id, "patient_id": patient_id, "status": "in_progress"
    }).execute()

    await update_fsm_session(session_id, patient_id=patient_id)
    return patient_id
```

`patient_id` must be included in the initial `TriageState` the very first time the graph is invoked for `INITIAL_SYMPTOM`.

### 4.2 LangGraph State

Once the user reaches `INITIAL_SYMPTOM`, control passes to LangGraph. State is persisted via `thread_id` (mapped 1:1 to the WebSocket `session_id`), tracking a `TriageState` TypedDict with `messages`, `present_symptoms`, `confidence`, `triage_level`, `department`, `payment_status`, `is_emergency`, `intent`, `requested_department_raw`, `requested_doctor_raw`, `selected_doctor_id`, `awaiting_department_choice`, and live `latencies`. The last five fields drive the direct-booking flow — see the AI engine prompt §4.

## 5. WebSocket Endpoints

### `ws/chat/{session_id}` (Patient Chat)
- 20-second background ping loop (`ping_task`) to keep connections alive through proxies.
- Streams via `triage_app.astream(stream_mode="updates")`, yielding graph state changes continuously.
- Emits JSON payloads to the frontend:
  - `{"type": "message"}` — chat content
  - `{"type": "typing"}` — typing indicator
  - `{"type": "emergency", "message": ..., "department": "Emergency Medicine"}` — emitted the instant `TriageState.is_emergency` becomes `True`. Takes priority over any other queued message on the frontend. Also writes an `audit_log` row (`event: "emergency_flagged"`, `metadata: {session_id, patient_id}`).

### `ws/diagnostics` (Developer Monitor)
- Protected by a query string `?token=...` matched against `DEVELOPER_PASSWORD`. Mismatch → close with code `1008` (Policy Violation).
- On success, the connection joins the global `_diagnostic_clients` list.
- As LangGraph iterates in `chat.py`, it broadcasts `diagnostic_update` events containing the current `node_name` and sanitized `node_state` (full message history stripped to save bandwidth) to all connected clients.
- This is a developer-only tool, not a doctor-facing product surface — keep the auth model to the shared token; no Supabase Auth or role checks belong here.

## 6. REST API

### 6.1 Public Endpoints (no auth — used by patient-side chat UI)

These back the department/doctor quick-reply chips in the chat frontend so it never has to hardcode specialty or doctor names, and so its list always matches what `node_detect_intent` in the AI engine can actually resolve.

| Method & Path | Purpose |
|---|---|
| `GET /api/v1/specialties` | All rows from `specialty`, `id` + `name` — renders as department chips |
| `GET /api/v1/doctors?specialty_id=` | Doctors in a specialty (`id`, `name`, `rating`, `avg_consult_min`), ordered by `rating DESC` — renders as a "choose your doctor" list |

### 6.2 Doctor Portal (authenticated)

All routes below require `Authorization: Bearer <supabase_jwt>`; a shared FastAPI dependency verifies the token and resolves it to a `doctor.id` via `doctor.auth_user_id`. Return `401` on missing/invalid token, `403` if the JWT is valid but has no matching `doctor` row.

| Method & Path | Purpose |
|---|---|
| `GET /api/v1/doctor/me` | Current doctor profile, joined with `specialty` |
| `GET /api/v1/doctor/dashboard` | KPI aggregates: patients waiting, critical count (`triage_level` 1–2), today's appointment count, average wait time |
| `GET /api/v1/doctor/queue` | Triage queue for this doctor's specialty — joins `appointment` + `patient` + `chat_session`, sortable by wait time, filterable by `triage_level` |
| `GET /api/v1/doctor/appointments?date=YYYY-MM-DD` | Agenda/calendar view for a given day |
| `PATCH /api/v1/doctor/appointments/{id}` | Update status (`in_consult`, `completed`), attach notes |
| `DELETE /api/v1/doctor/appointments/{id}` | Cancel — calls `supabase.rpc('cancel_appointment', {'p_appointment_id': id})` |
| `GET /api/v1/doctor/patients?search=...` | Patient directory search by name or contact |
| `GET /api/v1/doctor/patients/{id}` | Full patient detail: profile, `medical_history`, past `chat_session` rows, past `appointment` rows |
| `POST /api/v1/patient/feedback` | Patient submits post-consult rating (`stars`, `comment`) — no auth required, rate-limited by `appointment_id` |

## 7. Database & Persistence Integration

- **Access:** through the `get_supabase()` singleton (service-role key), never a per-request client.
- **Session finalization:** `node_explain` (in `triage_graph.py`) writes the completed triage outcome to `chat_session` — setting `status='completed'`, `completed_at=now()`, `final_diagnosis`, `department`, `triage_level`, `confidence`, `triage_summary`. On failure, falls back to an `audit_log` event so no clinical record is lost.
- **Slot booking:** `node_confirm_slot` calls `supabase.rpc('book_slot', {...})`. On a `SLOT_NOT_AVAILABLE` exception, catch it, tell the patient the slot was just taken, and route back to `node_fetch_slots` for a fresh offer.
- **Payments:** simulated — `asyncio.sleep(1.5)`, generate a fake `stripe_intent`, write `amount_paisa` (integer, INR × 100) to `payment`, set `status='succeeded'`.

## 8. Acceptance Tests

- WebSocket ping/pong keeps `/ws/chat/{session_id}` alive past 20 seconds through a proxy that would otherwise time out idle connections.
- After `PHONE_ENTRY` completes, `TriageState.patient_id` is non-null on the very first LangGraph invocation for that session.
- A test transcript that trips the emergency rule set produces a `{"type": "emergency"}` WS message within the same turn, plus an `audit_log` row.
- Firing two `node_confirm_slot` calls concurrently against the same `slot_id` results in exactly one successful booking and one graceful "slot just taken" message to the loser.
- `GET /api/v1/doctor/queue` without a bearer token returns `401`; with a token that doesn't map to any `doctor` row returns `403`.
- `/ws/diagnostics` with a wrong token closes with code `1008` and never enters `_diagnostic_clients`.
- Killing FAISS index files at startup does not crash `main.py` — RAG health tracking degrades gracefully.
- `GET /api/v1/specialties` and `GET /api/v1/doctors?specialty_id=` require no auth header and return `200` even for an anonymous patient session.
