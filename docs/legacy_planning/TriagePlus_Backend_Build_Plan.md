# TriagePlus — Backend Build Plan (Clean Rebuild, Supabase/Postgres)

This is a from-scratch build plan, not a patch list. Every fix identified in `TriagePlus_Issues_and_Fix_Instructions.md` and every Tiger in `TriagePlus_PreMortem_Risk_Analysis.md` is designed *into* the architecture below rather than bolted on after. Section 10 is a traceability table mapping every issue ID to the section that resolves it — check it before considering this done.

**Stack:** FastAPI · Supabase (Postgres) · Redis · Ollama (`llama3.2`, local) · FAISS · XGBoost · NetworkX

---

## 0. The One Structural Decision That Fixes the Most

TriagePlus's current `infer_department_final` asks the LLM to decide the department, fed by RAG context — which is exactly why issue 1.2 (fever/headache misrouted) happened: hand a small model specific, confident-sounding disease text and ask it to pick *something*, and it picks something. Per your locked architecture decision, **this rebuild replaces LLM-decides with classifier-decides, LLM-explains:**

- **XGBoost classifies** the department from a structured feature vector (symptoms, severity, duration, age, sex). It's not swayed by which FAISS chunks happened to retrieve well for a given query — it doesn't see the chunks at all.
- **The LLM only explains** the classifier's decision, using retrieved facts you hand it, under a prompt that explicitly forbids overriding the department.
- A **deterministic keyword pre-check** for common, non-specific complaints (fever, headache, fatigue) still runs *ahead of* the classifier, as a fast, auditable, zero-inference-cost safety net for the exact cases issue 1.2 reported.

This one change closes 1.2 at the architecture level instead of the prompt-engineering level, and it removes the RAG-contamination attack surface (1.1) from the decision path entirely — contamination can now only affect phrasing/explanation quality, never the actual department assigned.

---

## 1. Database Layer — Supabase Postgres

### 1.1 Schema (faithful to your attached ER diagram)

```sql
create extension if not exists pgcrypto;

create table specialty (
  id uuid primary key default gen_random_uuid(),
  name text not null unique
);

create table doctor (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  specialty_id uuid not null references specialty(id),
  rating float not null default 0 check (rating between 0 and 5),
  avg_consult_min float not null default 15,
  auth_user_id uuid references auth.users(id),   -- links to Supabase Auth, see §2
  created_at timestamptz not null default now()
);
create index idx_doctor_specialty on doctor(specialty_id);

create table patient (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  age int not null check (age between 0 and 130),
  gender text not null,
  contact text not null,
  language text not null default 'en',
  created_at timestamptz not null default now()
);
create index idx_patient_contact on patient(contact);

create table clinician_slot (
  id uuid primary key default gen_random_uuid(),
  doctor_id uuid not null references doctor(id) on delete cascade,
  start_time timestamptz not null,
  status text not null default 'open' check (status in ('open','held','booked','cancelled')),
  created_at timestamptz not null default now()
);
create index idx_slot_doctor_time on clinician_slot(doctor_id, start_time);
create index idx_slot_status on clinician_slot(status);

create table medical_history (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references patient(id) on delete cascade,
  conditions text,
  medications text,
  allergies text,
  immunocompromised boolean not null default false,
  updated_at timestamptz not null default now()
);
create index idx_history_patient on medical_history(patient_id);

create table appointment (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references patient(id) on delete cascade,
  slot_id uuid references clinician_slot(id),
  department text not null,
  triage_level int not null check (triage_level between 1 and 5),
  risk_score float not null default 0,
  confidence float,                     -- classifier confidence; feeds the eval set (T3 fix)
  status text not null default 'pending_slot'
    check (status in ('pending_slot','scheduled','in_queue','in_consult','completed','cancelled')),
  created_at timestamptz not null default now()
);
create index idx_appt_patient on appointment(patient_id);
create index idx_appt_slot on appointment(slot_id);
-- a slot can't be double-booked (not enforced anywhere in the original schema):
create unique index uq_appt_active_slot on appointment(slot_id)
  where status in ('scheduled','in_queue','in_consult');

create table feedback (
  id uuid primary key default gen_random_uuid(),
  doctor_id uuid not null references doctor(id) on delete cascade,
  stars int not null check (stars between 1 and 5),
  comment text,
  created_at timestamptz not null default now()
);
create index idx_feedback_doctor on feedback(doctor_id);

create table payment (
  id uuid primary key default gen_random_uuid(),
  appointment_id uuid not null references appointment(id) on delete cascade,
  stripe_intent text,
  status text not null default 'pending' check (status in ('pending','succeeded','failed','refunded')),
  amount float not null,
  created_at timestamptz not null default now()
);
create index idx_payment_appt on payment(appointment_id);

create table queue_entry (
  id uuid primary key default gen_random_uuid(),
  appointment_id uuid not null references appointment(id) on delete cascade,
  position int not null,
  est_wait_min int not null default 0,
  created_at timestamptz not null default now()
);
create unique index uq_queue_appt on queue_entry(appointment_id);

-- generic, append-only. Events reference other entities via metadata, not FK columns
-- (e.g. {"appointment_id": "...", "event": "department_assigned", "confidence": 0.82})
create table audit_log (
  id uuid primary key default gen_random_uuid(),
  event text not null,
  ts timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);
create index idx_audit_event_ts on audit_log(event, ts desc);
```

### 1.2 Tables added beyond the diagram — required to fix T8/T11/T12

Your diagram has no table for in-progress chat state, which is exactly what caused the in-memory-dict problems (T8: unbounded growth, T11: blocks horizontal scaling, T12: refresh desyncs frontend/backend). Add:

```sql
create table chat_session (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patient(id),          -- null until NAME_ENTRY completes
  fsm_state text not null default 'INITIAL_SYMPTOM',
  symptom_summary jsonb not null default '{}'::jsonb,
  department_suggested text,
  urgency_score int,
  classifier_confidence float,
  status text not null default 'active' check (status in ('active','completed','abandoned')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now()
);
create index idx_session_patient on chat_session(patient_id);
create index idx_session_last_seen on chat_session(last_seen_at);

create table chat_message (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references chat_session(id) on delete cascade,
  role text not null check (role in ('patient','assistant','system')),
  content text not null,
  created_at timestamptz not null default now()
);
create index idx_message_session on chat_message(session_id, created_at);
```

This directly replaces `_sessions: dict[str, dict]`. Reconnect logic becomes: look up `chat_session` by `session_id`, replay `chat_message` history to the client, resume from `fsm_state` — no more blank chat on refresh (T12 fixed structurally, not patched).

### 1.3 Row-Level Security

Enable RLS on every table (`alter table X enable row level security;`) even though the backend talks to Supabase exclusively via the **service role key** server-side. The service role bypasses RLS, so this doesn't affect normal operation — it's defense-in-depth: if the anon/public key ever leaks or gets used somewhere it shouldn't, a default-deny RLS policy means no data is exposed instead of everything being. Do not use the anon key from the frontend for direct table access anywhere; all reads/writes go through FastAPI, which enforces authorization explicitly (§2).

### 1.4 Migrations

Keep the DDL above as `supabase/migrations/0001_init.sql`, applied via `supabase db push` (Supabase CLI) or the SQL editor for the first pass. Every future schema change is a new numbered migration file, checked into git — this is what makes the schema reproducible instead of something that only exists in one person's Supabase project.

---

## 2. Auth & Security

**Doctor auth (fixes T6 — currently fully mocked):**
- Use **Supabase Auth** for doctor accounts — don't hand-roll password hashing/JWT issuance. Create doctor accounts via `supabase.auth.admin.create_user()`, link `doctor.auth_user_id` to the resulting `auth.users.id`.
- Login: frontend calls Supabase Auth directly (or through a thin backend proxy) to get a JWT; every subsequent doctor-portal request sends that JWT as a Bearer token.
- Backend verifies the JWT against Supabase's JWT secret (or JWKS endpoint) on every protected route — **actually verify it**, don't just check for the header's presence like the current `dummy_token` implementation.
- `get_doctor_queue()` and every other doctor route must derive `doctor_id` from the verified token's `sub` claim (joined against `doctor.auth_user_id`), never trust a client-supplied doctor ID.
- Treat this as a hard blocker before wiring any real queue data to these routes, exactly as T6 recommends — build the auth path first, then connect data to it, not the other way around.

**Patient session identity:** patients don't need full accounts for triage intake — a `chat_session` row plus phone-based contact capture (already in the `patient` table) is enough. Don't over-build auth here; the actual sensitive surface is the doctor/diagnostics side.

**Diagnostics/monitoring feed (fixes T5 — the single biggest issue in the pre-mortem):**
- Require the same doctor/staff JWT auth as the dashboard. No exceptions, no "internal tool so it's fine" reasoning — T5 is exactly this reasoning going wrong.
- Redesign what it broadcasts, not just who can see it: strip patient name/contact/exact age, and don't push raw symptom text at all. Send aggregated/de-identified signals — department distribution, average confidence, per-turn latency, error rates. If you genuinely need to inspect a raw conversation for debugging, that's a doctor pulling up one `chat_session` by ID through an authenticated, audited route — not a live broadcast to anyone who finds the URL.
- Route the frontend page (`/diagnostics` equivalent) behind the same auth guard as the rest of the doctor portal in `App.tsx` — the backend fix alone isn't sufficient if the frontend route is still open.

**Rate limiting (fixes T7):**
- REST endpoints: `slowapi` (FastAPI-native, built on the `limits` library) — per-IP limits on auth and booking endpoints especially.
- WebSocket (`/ws/chat/{session_id}`): `slowapi` doesn't cover WS natively — implement a Redis-backed counter (`INCR` + `EXPIRE` per IP per minute) checked on connect and on each inbound message; close the connection or reject messages past the threshold. This also gives you a natural place to cap concurrent sessions per IP.

---

## 3. Session & Conversation State (fixes T8, T11, T12)

- **Source of truth:** `chat_session` + `chat_message` in Postgres (§1.2).
- **Hot path cache:** Redis, keyed `session:{id}`, holding the current `fsm_state` + `symptom_summary` for fast reads during an active conversation — write-through to Postgres on every turn so Redis is never the only copy.
- **TTL eviction:** Redis key TTL of ~30 min idle; a scheduled job (or simple check-on-access) marks `chat_session.status = 'abandoned'` for sessions past that window in Postgres, so nothing grows unbounded even without Redis.
- **Reconnect:** on WebSocket connect, look up `session_id` → if `chat_session` exists and `status = 'active'`, replay `chat_message` history to the client (oldest first) before accepting new input, and resume from the stored `fsm_state`. This is the structural fix for T12 — the frontend is never left guessing what the backend thinks the state is.
- **Horizontal scaling (T11):** because state lives in Postgres/Redis, not process memory, you can run more than one backend worker safely — this was previously impossible.

---

## 4. RAG Knowledge Base

Three indices, each with a single, non-overlapping job — and every one of the concrete corpus/embedding bugs from Section 1 of the issues doc fixed at build time, not patched later.

### 4.1 Index 1 — Clinical Facts (FAISS)
**Source:** MedQuAD only. (Symptom2Disease moves to classifier training, §5 — see rationale below.)
**Fixes 1.5 (silent truncation):** split `question` from `answer`; embed `question` alone as the query-shaped record; chunk `answer` with sentence-aware splitting (~180 tokens, ~20% overlap). Add a build-time assertion that no `embed_text` exceeds your embedding model's token limit — fail the build loudly instead of silently truncating.
**Fixes part of 1.2 (no benign-presentation content):** MedQuAD is disease-topic-only by construction. Supplement it with general-practice-oriented content (general symptom-triage guidance covering "when is a fever/headache/fatigue not concerning") so retrieval has something to point to *besides* a specific disease when the presentation is genuinely non-specific.
**Runtime injection cap:** top-3 chunks, ~800 character hard cap total, tagged with `specialty` so the explanation node (§6) can filter to the classifier's predicted department specifically.

### 4.2 Index 2 — Requirement Graph (NetworkX)
This is new — TriagePlus doesn't currently have a principled next-best-question mechanism (the FSM's `already_collected`/`still_needed` logic implies something ad hoc). Build a proper graph: `Symptom → requires_info → Question Template`.
- **If DDXPlus is downloaded:** build from its evidence/pathology structure directly — it already encodes symptom→evidence dependencies, no LLM extraction needed for the graph's skeleton.
- **If not:** extract `(symptom, requires_info, question_asked)` triples from MedDialog + your own transcripts via LLM-assisted extraction, with a validation gate — whitelist relation types, sample 5–10% for manual review before trusting it in production. This is the single riskiest data-prep step; budget real time for it, and don't skip the validation sample even under deadline pressure.
- **Store in NetworkX** — no Neo4j at this scale, same reasoning as before.
- This graph is what drives the FSM's slot-filling logic going forward, replacing whatever hardcoded `still_needed` list currently exists with something that generalizes across symptom categories.

### 4.3 Index 3 — Phrasing Templates (FAISS)
**Sources:** `conversations/*.txt` (your existing synthetic corpus) + MedDialog + your own doctor-patient transcripts.
**Fixes 1.3 (MedDialog never actually embedded):** embed each MedDialog consultation's `description` field as the query-shaped record; store the first ~8 turns of `utterances` as payload (small-to-big retrieval); additionally sliding-window `utterances` (3-turn window, 1-turn overlap) as a second record type for mid-conversation phrasing. Exact-dedup on normalized `description`; stratified-subsample to ~3,000–5,000 entries so it doesn't drown out the synthetic corpus.
**Your own transcripts:** split to turn pairs, keep doctor question-turns, tag by symptom category (keyword match against canonical vocabulary) — no labels needed for this use, unlike classifier training.
**Fixes 1.4 (severe class imbalance — 76.8% Respiratory, 19% Musculoskeletal, 6 of 17 departments represented at all):** cap chunks per specialty at index-build time (reservoir sampling, e.g. max 150/specialty from `conversations/*.txt`), and add or expand a `General/` folder of synthetic conversations specifically covering fever, headache, fatigue, and common cold in the same `D:`/`P:` script format, so General Medicine has real exemplars instead of 19 chunks' worth.
**Fixes 1.1 (RAG contamination) at retrieval time, not just prompt time:** filter each retrieved chunk against the relevance threshold *individually* (`score < 0.35` → drop that chunk, regardless of whether hit #1 cleared the bar) — a strong top hit must never drag in weak, irrelevant hits #2/#3.

### 4.4 Shared retrieval/prompt-injection helper (fixes 1.1 everywhere it applies)
Both Index 1 (facts) and Index 3 (phrasing) get injected into LLM prompts — build **one** shared function both nodes call, so the contamination fix lives in one place instead of being reimplemented per call site:

```python
def build_rag_block(chunks: list[dict], label: str, char_cap: int = 400) -> str:
    """chunks already individually threshold-filtered by the caller."""
    if not chunks:
        return ""
    body = "\n".join(f"- {c['text'][:250]}" for c in chunks)[:char_cap]
    return (
        f"\n{label} (unrelated prior cases, shown only for phrasing/style — "
        f"do NOT state, imply, or ask about any specific fact, diagnosis, or "
        f"detail from these unless the current patient has said it themselves):\n{body}"
    )
```
Every prompt that injects retrieved text uses this, plus a matching negative instruction in the system prompt (as in the original fix instructions, Step 1.3) — no call site is allowed to inject raw `cases_text` directly.

---

## 5. Classification (XGBoost, decides — fixes 1.2 structurally, per §0)

**Training data:** Symptom2Disease (primary — this is exactly the labeled `symptom_text → disease` pairs a classifier needs) + DDXPlus if available (broader disease coverage). This is a better use of Symptom2Disease than embedding it into a RAG index, since the classifier — not retrieval quality — now makes the department decision.

**Feature schema:** multi-hot vector over the canonical symptom vocabulary (from Index 2's graph) + severity (ordinal 1–5) + duration (bucketed: <24h / 1–3d / >3d / chronic) + age bucket + sex.

**Disease → department mapping:** build this table by hand once (e.g., "Migraine" → Neurology, "Eczema" → Dermatology) against your 17-department taxonomy — neither Symptom2Disease nor DDXPlus ships this mapping natively.

**Deterministic pre-check, still ahead of the classifier (fixes 1.2's specific reported cases directly):**
```python
GENERAL_MEDICINE_DEFAULTS = {
    "fever", "headache", "fatigue", "tiredness", "sore throat",
    "common cold", "mild cough", "body ache", "general weakness",
}
SPECIALTY_OVERRIDE_TERMS = {
    "worst headache", "neck stiffness", "chest pain", "vision loss",
    "seizure", "severe", "unconscious", "difficulty breathing",
    "blood in", "pregnant", "rash spreading",
}

def rule_based_department(summary: str) -> str | None:
    s = summary.lower()
    if any(t in s for t in SPECIALTY_OVERRIDE_TERMS):
        return None  # let the classifier decide — this needs real triage judgment
    if any(t in s for t in GENERAL_MEDICINE_DEFAULTS) and not any(
        t in s for t in SPECIALTY_OVERRIDE_TERMS
    ):
        return "General Medicine"
    return None
```
Run this first; only fall through to the classifier if it returns `None`. Cheap, fast, auditable, and doesn't depend on training data quality for the common case.

**Urgency score safety margin (fixes T2 — currently no equivalent to the department fallback):** when classifier confidence or RAG match score is low, **floor** the urgency score upward (e.g., minimum "moderate") rather than trusting a raw low number, or route to an explicit "insufficient information — please seek in-person evaluation" path. A confidently-wrong department is embarrassing; a confidently-low urgency on something serious is the failure mode that actually hurts someone.

**Evaluation (fixes T3 — no accuracy eval anywhere):** build a small labeled eval set (symptom text → correct department + urgency) before trusting the `0.6`/`0.35`/`0.3`-style thresholds with real patients. Report precision/recall/F1 per department, and specifically re-test the exact fever/headache cases from 1.2 as regression cases.

---

## 6. LLM Orchestration (Ollama, `llama3.2`)

**Explanation generation (replaces the department-deciding call, per §0):** classifier output → Index 1 retrieval filtered to the predicted department → constrained prompt:
```
System: You are explaining a triage recommendation to a patient. Use ONLY the
facts provided below. Do not diagnose. Do not suggest a different department
than the one given. If the facts are insufficient, say so and recommend an
in-person evaluation. Keep the response under 120 words.

Recommended department: {department}
Patient's reported symptoms: {symptom_list}
Supporting facts: {build_rag_block(index1_chunks, "SUPPORTING FACTS")}
```
Low temperature (~0.3), `num_predict` capped (~150).

**Async correctness (fixes T4-adjacent/1.6 — blocking call freezes every concurrent session):**
```python
top_chunks, max_score = await asyncio.to_thread(_search_faiss, query, index, meta, top_k=3)
```
Applied everywhere `embedder.encode` or FAISS search is called from an `async def` — the interactive path was the one place this was missing in the original code; make it a rule for every call site in the rebuild, not a per-instance fix.

**Timeouts (fixes T8/1.8 — no timeouts anywhere):** wrap every Ollama call in `asyncio.wait_for(..., timeout=N)` with a graceful fallback message on timeout, not a silent hang.

**Sequential-call reduction (partially addresses 1.8):** gate emergency checking to symptom-related FSM states only (`INITIAL_SYMPTOM`, `GEMINI_CONVERSATION`-equivalent) — skip it entirely during `NAME_ENTRY`/`AGE_ENTRY`/`GENDER_ENTRY`/`PHONE_ENTRY`, since those turns can't possibly describe a symptom. See §7 for why this matters less than it used to.

**Documentation honesty (fixes T9/1.10):** this plan assumes Ollama/`llama3.2` as the only inference path from day one — there's no Gemini fallback to reconcile because it's never introduced. `requirements.txt` never lists `google-generativeai`; the README describes Ollama setup from the first line.

---

## 7. Emergency Detection — Deterministic-First (fixes T1, structurally)

The original fix instructions (issues doc Step 6) add a keyword net that runs *alongside* the LLM check, but its own sample code still returns `False` on LLM exception — it narrows the fail-open gap without closing it. The pre-mortem (T1) correctly calls this out: **failing closed is the only correct default here.** This rebuild resolves it more simply than patching the exception handler — **make the deterministic check authoritative on its own, and the LLM check purely additive:**

```python
EMERGENCY_KEYWORDS = [
    "can't breathe", "cannot breathe", "chest pain", "unconscious",
    "severe bleeding", "not breathing", "stroke", "heart attack",
    "suicidal", "want to die", "overdose",
]

def keyword_emergency_check(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)

async def check_emergency(text: str) -> bool:
    if keyword_emergency_check(text):
        return True                      # authoritative — no LLM call even needed
    try:
        return await asyncio.wait_for(_llm_emergency_check(text), timeout=5)
    except Exception as e:
        logger.error(f"LLM emergency check failed: {e}")
        return False                     # fine — deterministic check already ran and said no
```
Because the deterministic check runs first and is sufficient on its own, an LLM/Ollama failure can never cause a missed emergency that the keyword net would have caught — the LLM only ever *adds* coverage for phrasing the keyword list misses, never subtracts from it. This is a stronger guarantee than "OR the two checks together and hope the exception handler is right," and it doesn't depend on Ollama being up at all for the safety-critical path. Enrich the keyword list using real phrasing mined from your own transcripts (§4.3) — patients don't always say "chest pain," they say what actually hurts, in their own words.

---

## 8. Doctor Portal, Appointments, Scheduling

- **Doctor queue** (`/doctors/me/queue`): derives `doctor_id` from the verified JWT (§2), returns `appointment` rows joined to `queue_entry` where `slot.doctor_id = doctor_id`, ordered by `queue_entry.position`.
- **Scheduling:** no ML needed — constraint-based slot allocation: filter `clinician_slot` by `specialty_id` matching the classifier's department, sort by doctor rating, pick soonest available slot, weighted by `appointment.triage_level` for priority patients. Insert with `status='held'` momentarily to prevent race conditions on the unique active-slot index (§1.1), then confirm to `'scheduled'`.
- **Feedback, payment:** standard CRUD against the tables in §1.1, no AI involvement — explicitly out of scope for the RAG/LLM layer, worth stating so it's not mistaken for an oversight.
- **Patient brief for the doctor portal:** a single Gemini/Llama summarization call over the completed `chat_session.symptom_summary` + classifier output, generated once per completed intake — not a new subsystem, just a read of data you already have.

---

## 9. Observability, Testing, Deployment, Repo Hygiene

**Observability (fixes E3 — nobody currently watches this in production):** wire `audit_log` inserts at the key decision points — emergency detection triggered, department assigned (with confidence), LLM call failed/timed out, auth failure. This is cheap now and is what turns "the team finds out from a complaint" into "the team finds out from a dashboard."

**Testing (fixes T13 — zero tests anywhere):** start with the two scenarios that matter most in this domain — a regression here means a missed emergency, not a broken button:
- FSM transition tests (state machine correctness).
- Emergency fallback test: simulate Ollama being down, assert the keyword net still triggers the banner (§7's whole point).
- RAG contamination test: inject a case with a distinct diagnosis into Index 3, assert an unrelated query's response doesn't leak it.
- General Medicine regression test: fever-only and headache-only inputs resolve to General Medicine (the exact 1.2 cases).
Add a GitHub Actions workflow running `pytest` on every push — even minimal CI beats zero CI.

**Deployment (fixes T10 — no coherent path beyond localhost):** Ollama + FAISS-in-memory is a stateful, resource-heavy service — it does not fit a serverless/static host. Deploy the backend as a container (Dockerfile, `0.0.0.0` bind, not `127.0.0.1`) to a persistent-process host with enough RAM for the `llama3.2` model plus both FAISS indices loaded (budget ~4–8GB). Frontend stays a static SPA (Vercel, as already planned). Confirm the full stack actually runs on the target host before calling it launch-ready — a laptop demo passing isn't the same claim.

**Repo hygiene (fixes T14, T15, T16):** `.gitignore` covers `*.faiss`, `index_*_meta.json`, and dataset caches consistently (not just some extensions); regenerable index artifacts are rebuilt via a repeatable pipeline script, not committed; the recovery script takes a path argument or reads from env instead of a hardcoded Windows path; delete `.bak` files and `frontend/legacy/`; add a `LICENSE`; keep `backend/.env.example` in sync with `frontend/.env.example`.

---

## 10. Traceability — Every Issue, Where It's Fixed

| ID | Issue | Fixed in |
|---|---|---|
| 1.1 | RAG contamination from unframed retrieved chunks | §4.4 shared `build_rag_block()`, applied at every injection site |
| 1.2 | Fever/headache misrouted to specialties | §0 architecture change + §5 deterministic pre-check + classifier |
| 1.3 | MedDialog/Symptom2Disease downloaded but never embedded | §4.3 (MedDialog → Index 3), §5 (Symptom2Disease → classifier) |
| 1.4 | Index A severely class-imbalanced (76.8% Respiratory) | §4.3 per-specialty cap + General/ folder expansion |
| 1.5 | MedQuAD embeddings silently truncated | §4.1 question/answer split + sentence-aware chunking + build-time assertion |
| 1.6 | Blocking CPU-bound call freezes event loop | §6 `asyncio.to_thread` as a universal rule |
| 1.7 / T1 | Emergency detection fails open | §7 deterministic-first design |
| 1.8 / T8 | No timeouts, wasted calls on non-symptom turns | §6 timeouts + FSM-state gating |
| 1.9 / T8,T11,T12 | Unpersisted session state | §1.2, §3 |
| 1.10 / T9 | Docs/deps describe a Gemini system that doesn't exist | §6 — never introduced in this rebuild |
| T2 | No confidence-based urgency safety margin | §5 urgency floor |
| T3 | No accuracy evaluation | §5 eval set requirement |
| T5 | Unauthenticated broadcast of patient data | §2 diagnostics redesign |
| T6 | Doctor auth fully mocked | §2 Supabase Auth |
| T7 | No rate limiting | §2 `slowapi` + Redis WS throttling |
| T10 | No deployment path beyond localhost | §9 |
| T13 | Zero tests, zero CI | §9 |
| T14, T15, T16 | Repo hygiene, hardcoded paths, dead code | §9 |
| E3 | No production observability | §9 `audit_log` wiring |
| E4 | No clinician in the loop | Not a code fix — flag explicitly: get the red-flag keyword list (§7) and `GENERAL_MEDICINE_DEFAULTS`/`SPECIALTY_OVERRIDE_TERMS` tables (§5) reviewed by a clinician before real patients rely on them |

---

## 11. Build Order

1. Supabase project + schema (§1) — everything else depends on this existing first.
2. Auth (§2) — build and test doctor login/JWT verification before any route uses it.
3. Session persistence (§3) + FSM skeleton, no LLM yet — get the deterministic core working and tested first.
4. RAG indices (§4) — this is the slowest phase; DDXPlus/no-DDXPlus decision gates the Index 2 approach.
5. Classifier (§5) — train, evaluate, integrate the pre-check + XGBoost + urgency floor.
6. LLM orchestration (§6) + emergency detection (§7) — wire Ollama last, since everything above it should work (in a degraded, LLM-free mode) without it.
7. Doctor portal, scheduling, payments (§8).
8. Observability, tests, deployment, repo cleanup (§9) — don't leave this for "later," it's what the pre-mortem's Elephants are actually about.
