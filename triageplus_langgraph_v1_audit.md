# TriagePlus — `langgraph_v1` Audit

Repo: `skanda-P/TriagePlus`, branch `langgraph_v1`, commit `5a3b527`.

---

## CRITICAL

### 1. Doctor Dashboard is completely dead — no matching backend route
`frontend/src/pages/DoctorDashboard.tsx:18` fetches `GET /api/v1/doctors/me/queue`. There is no `doctors` router anywhere in `backend/app/api/` — only `auth.py` and `chat.py` are registered in `main.py:26-27`. Every dashboard load 404s and shows "Could not load queue."

**Fix:** add a `doctors.py` router with a `/me/queue` endpoint gated by `get_doctor_user` (already defined in `deps.py`), querying `chat_session`/`appointment` by the doctor's specialty. Register it in `main.py`.

### 2. Payment step is unreachable — `route_entry()` has a routing gap
`backend/app/core/triage_graph.py:521-536`

```python
def route_entry(state: TriageState) -> str:
    if state.get("payment_status") == "pending":
        return "process_payment"
    if state.get("booking_intent") is True and not state.get("selected_slot_id") and state.get("available_slots"):
        return "confirm_slot"
    if state.get("booking_intent") is None and state.get("department"):
        return "handle_booking"
    return "extract_symptoms"
```

After `node_confirm_slot` runs (`triage_graph.py:472-502`), state has `selected_slot_id` set but `payment_status` is still unset (`None`) — it's only ever set inside `node_process_payment` itself. So on the user's very next message (e.g. "PAY"):
- `payment_status == "pending"` → False (never set yet)
- `booking_intent True and not selected_slot_id` → False (`selected_slot_id` is now set)
- `booking_intent is None` → False (it's `True`)
- falls through to `"extract_symptoms"` — the user's "PAY" reply gets fed into symptom extraction instead of the payment node.

Once inside `process_payment` this self-corrects (payment_status="pending" is then caught by the first branch), but the *first* entry is always missed. Booking a slot and paying is currently broken end-to-end.

**Fix:** set `payment_status: "pending"` in `node_confirm_slot`'s return dict (`triage_graph.py:502`), or add a route_entry branch: `if state.get("selected_slot_id") and state.get("payment_status") != "succeeded": return "process_payment"`.

### 3. Triage result card never renders — backend never sends `meta`
Frontend gates the whole "Recommended specialty / confidence / urgency" card on `sessionMeta.specialty`, which is only set via `setSessionMeta`, which only fires when a websocket frame has `data.meta`:

- `frontend/src/hooks/useWebSocket.ts:39-48` — reads `data.meta.{specialty,confidence,confidence_label,urgency,triage_level,triage_color}`
- `frontend/src/components/chat/ChatWindow.tsx:78` — `{sessionMeta.specialty && (...)}`

`backend/app/api/v1/chat.py:239-261` streams LangGraph node output but only ever sends `{"type": "message", "content": ...}` and a bare `triage_complete` event. `node_explain`'s return dict (department, confidence, urgency — `triage_graph.py:326-331`) is discarded except for the message text. **No code path ever emits a `meta` key.** The result card is 100% dead on the current backend regardless of how well the classifier performs.

**Fix:** in the `chat.py` stream loop, when `node_name == "explain"`, pull `department`/`confidence`/`urgency` out of `node_state` and send them as `{"type": "message", ..., "meta": {...}}`.

---

## HIGH

### 4. Confidence will be double-scaled once #3 is fixed
`triage_graph.py:329` already converts to a percentage: `"confidence": round(confidence * 100, 1)` (e.g. `87.3`).
`ChatWindow.tsx:85` assumes a 0–1 fraction and re-multiplies: `` `${Math.round((sessionMeta.confidence ?? 0) * 100)}%` `` → would render `8730%`.

**Fix:** pick one convention. Simplest: send the raw 0–1 fraction from the backend (`confidence` not `confidence*100`), or drop the `*100` on the frontend.

### 5. Urgency scale mismatch
Backend: `urgency_score = 6 - severity` where `severity` is 1–5 → `urgency_score` range is **1–5** (`triage_graph.py:319-322`).
Frontend: `ChatWindow.tsx:87` — `Urgency: {sessionMeta.urgency}/10` (assumes 0–10).

**Fix:** either scale on backend (`urgency_score * 2`) or fix the frontend label to `/5`.

---

## MEDIUM

### 6. Session DB with patient PII is committed to git
`backend/app/api/v1/sessions.db` and `backend/triage_checkpoints.sqlite` are both tracked in git (confirmed via `git ls-files`), and `.gitignore` has no `*.db`/`*.sqlite` rule. `sessions.db` stores `patient_name`, `age`, `gender`, `phone` in plaintext JSON blobs (`chat.py:115-120`, `178-203`). Currently empty, but any local dev run will produce real chat/PII data that's one `git add .` away from being committed and pushed.

**Fix:** add `*.sqlite`, `*.db` to `.gitignore`, `git rm --cached` both files.

### 7. `doctor_login` doesn't check doctor-table membership
`backend/app/api/v1/auth.py:30-33` — the doctor-table check is written but commented out. Any valid Supabase auth account (including patients) gets a "successful" doctor login and a session token, then immediately 403s on every dashboard call via `get_doctor_user` (`deps.py:38-54`, which *does* check the table correctly). Confusing UX, and technically returns a real JWT to non-doctors.

**Fix:** uncomment the check and raise 403 in `doctor_login` itself.

### 8. Inconsistent emergency handling between two code paths
`chat.py:12-14` and `triage_graph.py:213` keep **separate, duplicated** emergency keyword lists. `chat.py:164-172` hard-closes the socket on a raw keyword match before LangGraph ever runs. But `node_emergency_check` (`triage_graph.py:206-226`) *also* flags emergencies via KG severity (symptoms mapped to severity-1/2 conditions) — a path `chat.py` never sees. When that path fires, `node_decide_next` routes to `explain` (`triage_graph.py:228-233`), which emits the ER warning, but the edge `explain → prompt_booking` is unconditional (`triage_graph.py:578`), so the bot then asks "Would you like me to find an available doctor in the Emergency Medicine department?" instead of stopping. Two different emergency-detection paths, two different behaviors.

**Fix:** share one keyword list; add a conditional edge so `is_emergency=True` routes `explain → END` instead of `explain → prompt_booking`.

---

## LOW / latent

### 9. `patient_verify_otp` will 422 on first use
`auth.py:56-57` — `def patient_verify_otp(email: str, token: str)` uses bare scalar params, which FastAPI treats as **query params**, not a JSON body, unlike the sibling endpoints in the same file that use Pydantic models. Not currently called anywhere in the frontend (no patient-login UI exists yet), so it's latent — but will break the moment someone POSTs JSON to it expecting the same convention as `/doctor/login`.

**Fix:** wrap in a `PatientVerifyOtpRequest(BaseModel)` like the other two endpoints.

### 10. Doctor JWT passed as a WS query param
`chat.py:53` — `/ws/diagnostics?token=...`. Tokens in query strings land in server/proxy access logs. Minor hardening item, not urgent.

---

## Priority order to fix
1. #2 (payment routing) and #3 (missing `meta`) — both silently break the two core product flows (booking+payment, and showing the AI's actual recommendation).
2. #1 (missing doctors route) — dashboard is unusable for the actual clinician-facing side.
3. #4/#5 — cosmetic but will actively mislead once #3 is fixed.
4. #6/#7 — hygiene/security, cheap to fix.
5. #8/#9/#10 — lower urgency, fix when touching those areas.

---
---

# Addendum: "chat is slow to load" + "fever → emergency" + architecture review

## 11. Why even the hardcoded intake questions (name/age/gender/phone) feel slow

None of `NAME_ENTRY` / `AGE_ENTRY` / `GENDER_ENTRY` / `PHONE_ENTRY` (`chat.py:174-206`) touch LangGraph, the KG, RAG, or the ML models — they're pure `if`/`send_json`. So the lag isn't in that code path itself; it's contention from something else running on the same process at the same time. Two concrete causes, both real:

**a) RAG model warmup fires eagerly on every *new* session, and it's CPU-bound work sharing the GIL with your event loop.**
`chat.py:114-122`:
```python
if is_new:
    ...
    asyncio.create_task(asyncio.to_thread(load_rag_models))   # <-- fires immediately on connect
    await websocket.send_json({... welcome message ...})
```
`load_rag_models()` (`triage_graph.py:68-129`) loads `HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")` — a real PyTorch/transformers model — plus two FAISS indices, all via `asyncio.to_thread`. `to_thread` moves the call off the event-loop thread, but it does **not** give you true parallelism for CPU-bound Python: the GIL is still shared, and `torch`/`transformers` model loading holds it for meaningful stretches. While that's loading (can be several seconds, worse on first run if weights aren't cached), the main thread — the one answering "what's your name?" — gets starved and every hardcoded reply stalls too.

It's also pointless to warm this up before the user has even given their name: RAG is never touched until `INITIAL_SYMPTOM`/`LLM_CONVERSATION`. There's no reason to pay this cost during intake at all.

A global `_rag_load_attempted` flag means this only *truly* loads once per process — but during active dev (frequent restarts), that "once" lands on whatever session happens to connect right after each restart, which is exactly the session where this is being noticed.

**Fix:** move the warmup to a FastAPI startup hook (`@app.on_event("startup")` / lifespan handler in `main.py`) so it runs once, deterministically, before the server accepts any websocket connections — not racing a live user's intake flow.

**b) Per-message graph recompilation + fresh SQLite connection (applies once past intake, into every AI turn).**
`chat.py:215-216`, run on *every single message* once in `INITIAL_SYMPTOM`/`LLM_CONVERSATION`/etc.:
```python
async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
    triage_app = graph_builder.compile(checkpointer=checkpointer)
```
This opens a new aiosqlite connection and rebuilds/recompiles the entire `StateGraph` (10 nodes, ~10 edges) from scratch on every turn, instead of once. Doesn't explain the *hardcoded*-question lag (those never reach this code), but it's a real, avoidable latency tax on every AI turn and worth fixing alongside (a).

**Fix:** compile `triage_app` once at import/startup with a long-lived checkpointer (or a connection pool), and reuse it across requests — recreate the config's `thread_id` per session, not the whole graph.

*(Not a bug, but a related, expected cost:* `await llm.ainvoke(...)` in `node_next_question`/`node_explain` is a real synchronous round-trip to local Ollama — that latency is inherent to local LLM inference, not something to "fix" here.)

## 12. Why reporting "fever" routes straight to the emergency message

`"fever"` is not in the hardcoded keyword list (`chat.py:13`, `triage_graph.py:213`), so this isn't the keyword path — it's the KG severity path, and it's a genuine logic bug in the matching rule itself, independent of what's actually in the DDXPlus JSON:

`triage_graph.py:217-224`:
```python
if not is_emerg:
    for cond_id, cond_data in kg.conditions.items():
        if cond_data.get("severity") in [1, 2]:
            cond_symps = cond_data.get("symptoms", {})
            if any(s in cond_symps for s in state["present_symptoms"]):
                is_emerg = True
                break
```
This says: *if even one of the patient's reported symptoms happens to appear anywhere in the symptom list of any severity-1/2 condition in the whole DDXPlus corpus, declare an emergency.* There's no minimum-match count, no weighting by how defining/specific that symptom is to the condition, and no requirement to have gathered more than one data point first — `node_decide_next` (`triage_graph.py:228-233`) checks `is_emergency` *before* checking `question_count`, so this can fire on the very first message.

Fever is about as non-specific as a symptom gets — it's a listed symptom of dozens of conditions across every severity band, including rare severe ones (sepsis-adjacent, PE, meningitis-type presentations, etc. tend to exist somewhere in a 40+ condition severity-1/2 set). Because the rule only needs *one* overlapping evidence code with *any one* of those conditions, reporting a single common, non-specific symptom is essentially guaranteed to trip it. (I couldn't pull the actual DDXPlus JSON contents to name the exact condition — this repo's data files are Git LFS pointers and the LFS media host rejected an anonymous fetch — but the bug is in the matching rule itself, not a property of the specific dataset row, so this holds regardless.)

This is also a safety problem, not just an annoyance: over-triggering "this is an emergency, go to the ER" erodes trust and causes alert fatigue — the opposite of what an emergency flag is for.

**Fix options (combine 1+2):**
1. Require a minimum overlap, e.g. ≥50% of a condition's *listed* symptoms present, not just one.
2. Maintain a small, curated "true red-flag" evidence-code allowlist (chest pain, loss of consciousness, severe hemorrhage, etc.) for immediate escalation, and keep the broad KG severity-matching only as an input to *classification/urgency scoring*, not an instant hard stop.
3. Exclude generic, low-specificity evidence codes (fever, fatigue, generic pain/cough) from ever being sufficient on their own to set `is_emergency`.
4. Gate the KG-based path (unlike the keyword path, which should stay instant) behind having asked at least 1–2 clarifying questions, so a single ambiguous word can't end the conversation.

## 13. LangGraph + RAG + DB structure — review and suggested corrections

**LangGraph**
- *State is split across two independent state machines that can drift.* Intake (`NAME_ENTRY`→`PHONE_ENTRY`) is tracked by a hand-rolled `fsm_state` string in `sessions.db`; everything after that is tracked by LangGraph's own checkpointed `TriageState` in `triage_checkpoints.sqlite`. Two systems, two transcripts (`state["history"]` in `sessions.db` vs. `state["messages"]` in the LangGraph checkpoint) that nothing guarantees stay in sync. **Suggestion:** fold intake into the graph itself as 4 more nodes at the front (`collect_name → collect_age → collect_gender → collect_phone → extract_symptoms`) so there's one state machine, one checkpointer, one transcript, for the life of a session.
- *`route_entry()` reverse-engineers "where are we" from a combination of optional fields* (`payment_status`, `booking_intent`, `selected_slot_id`, `available_slots`, `department`) rather than reading one explicit field. This is exactly how bug #2 (payment unreachable) happened — an implicit combination the entry function forgot to handle. **Suggestion:** add one explicit `flow_stage: Literal["triage","booking_prompt","slot_selection","payment","done"]` field to `TriageState`, set it explicitly at the end of every node that changes stage, and make `route_entry` a plain switch on that field. New stages become impossible to "forget," because adding one forces a new `match` arm instead of a new implicit-flag combination.
- *Emergency detection logic is duplicated* between `chat.py` (raw keyword list, hard socket close) and `triage_graph.py` (same keyword list + KG severity check, soft in-graph message). **Suggestion:** pull into one `core/emergency.py` used by both call sites, and apply the fix from §12 there.
- *`explain → prompt_booking` edge is unconditional* even when `is_emergency=True` (`triage_graph.py:578`), so an ER warning is immediately followed by "want me to book a doctor?" **Suggestion:** branch this edge on `is_emergency` and route straight to `END` for the emergency case.

**RAG**
- Model warmup should be a startup-lifecycle event, not something dispatched per-connection (§11a).
- `_medquad_index`/`_conversations_index` FAISS lookups happen on every turn via `asyncio.to_thread`; fine functionally, but there's no caching of repeated/near-duplicate queries (e.g. the same `base_question` text recurring across sessions). A small LRU on `(index_name, query)` would cut a meaningful chunk of latency for common intake questions.
- RAG "health" is tracked well (`_rag_health` dict, surfaced to the user as "degraded mode" text) — that pattern is good; consider surfacing `_ner_health` the same way, since NER silently failing means `present_symptoms` silently stops growing from free text with no user-visible signal at all.

**Database**
- *Three separate persistence layers for one conversation*: hand-rolled `sessions.db` (schemaless — one `data TEXT` blob per row), LangGraph's `triage_checkpoints.sqlite`, and Supabase (`chat_session`, `audit_log`, `appointment`, `clinician_slot`). Folding intake into the graph (above) removes one of the three; consider whether `sessions.db` is needed at all once that's done.
- `node_explain`'s Supabase writes (`triage_graph.py:386-413`) are wrapped in try/except that only logs on failure — the user is never told their result wasn't persisted, so a doctor may never see this patient anywhere. Combined with critical bug #1 (no `/doctors/me/queue` route at all), there's currently no verified path from "triage completed" to "a doctor can see it." **Suggestion:** surface persistence failures to the user (even a soft "your session may not have saved — please note down your recommendation" is better than silence), and add an alerting/retry path for the audit_log fallback failure case.
- `node_confirm_slot` (`triage_graph.py:489-497`) does two non-atomic Supabase calls — `update slot status=held` then `insert appointment` — with no rollback if the second fails. That leaks a permanently-held slot with no appointment behind it. **Suggestion:** wrap both in a single Postgres RPC/transaction (Supabase supports Postgres functions callable via `.rpc()`), or add a reconciliation job that releases slots held longer than N minutes with no matching appointment row.
