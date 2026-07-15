create extension if not exists pgcrypto;

-- 2.1 specialty ---------------------------------------------------------
create table specialty (
  id   uuid primary key default gen_random_uuid(),
  name text not null unique
);

-- 2.2 doctor --------------------------------------------------------------
create table doctor (
  id             uuid primary key default gen_random_uuid(),
  name           text not null,
  specialty_id   uuid not null references specialty(id),
  rating         float not null default 4.5 check (rating between 0 and 5),
  avg_consult_min float not null default 15,
  auth_user_id   uuid references auth.users(id),   -- Supabase Auth identity for doctor login
  created_at     timestamptz not null default now()
);
create index idx_doctor_specialty on doctor(specialty_id);

-- 2.3 patient ---------------------------------------------------------------
create table patient (
  id           uuid primary key default gen_random_uuid(),
  name         text not null,
  age          int  not null check (age between 0 and 130),
  gender       text not null,
  contact      text not null,
  language     text not null default 'en',
  auth_user_id uuid references auth.users(id),      -- null for anonymous patients (default case)
  created_at   timestamptz not null default now()
);
create index idx_patient_contact on patient(contact);  -- lookup key for get_or_create_patient logic

-- 2.4 medical_history ---------------------------------------------------
create table medical_history (
  id                uuid primary key default gen_random_uuid(),
  patient_id        uuid not null references patient(id) on delete cascade,
  conditions        text,
  medications       text,
  allergies         text,
  immunocompromised boolean not null default false,
  updated_at        timestamptz not null default now()
);
create index idx_history_patient on medical_history(patient_id);

-- 2.5 clinician_slot ------------------------------------------------------
create table clinician_slot (
  id         uuid primary key default gen_random_uuid(),
  doctor_id  uuid not null references doctor(id) on delete cascade,
  start_time timestamptz not null,
  status     text not null default 'open' check (status in ('open','held','booked','cancelled')),
  held_at    timestamptz,   -- set when status -> 'held'; used by release_stale_holds()
  created_at timestamptz not null default now()
);
create index idx_slot_doctor_time on clinician_slot(doctor_id, start_time);
create index idx_slot_status on clinician_slot(status);

-- 2.6 chat_session ---------------------------------------------------------
-- The persistent record of one AI triage conversation, independent of whether it
-- results in a booking. Keyed by the same session_id used by the WebSocket and LangGraph.
create table chat_session (
  id              uuid primary key default gen_random_uuid(),
  session_id      text not null,                -- == WebSocket path param == LangGraph thread_id
  patient_id      uuid references patient(id),  -- set once intake collects name/age/gender/contact
  status          text not null default 'in_progress' check (status in ('in_progress','completed','abandoned')),
  is_emergency    boolean not null default false,
  final_diagnosis text,
  department      text,
  triage_level    int check (triage_level between 1 and 5),
  confidence      float check (confidence between 0 and 1),
  triage_summary  text,
  created_at      timestamptz not null default now(),
  completed_at    timestamptz
);
create unique index idx_chat_session_session_id on chat_session(session_id);
create index idx_chat_session_patient on chat_session(patient_id);

-- 2.7 appointment -----------------------------------------------------------
create table appointment (
  id               uuid primary key default gen_random_uuid(),
  patient_id       uuid not null references patient(id) on delete cascade,
  chat_session_id  uuid references chat_session(id),   -- traces the booking back to its triage conversation
  slot_id          uuid references clinician_slot(id),
  department       text not null,
  triage_level     int not null check (triage_level between 1 and 5),
  risk_score       float not null default 0,
  confidence       float check (confidence between 0 and 1),
  status           text not null default 'pending_slot'
                     check (status in ('pending_slot','scheduled','in_queue','in_consult','completed','cancelled')),
  created_at       timestamptz not null default now()
);
create index idx_appt_patient on appointment(patient_id);
create index idx_appt_slot on appointment(slot_id);
create unique index uq_appt_active_slot on appointment(slot_id)
  where status not in ('cancelled','completed');

-- 2.8 feedback ------------------------------------------------------------
create table feedback (
  id             uuid primary key default gen_random_uuid(),
  doctor_id      uuid not null references doctor(id) on delete cascade,
  appointment_id uuid references appointment(id),
  stars          int not null check (stars between 1 and 5),
  comment        text,
  created_at     timestamptz not null default now()
);
create index idx_feedback_doctor on feedback(doctor_id);

-- 2.9 payment ---------------------------------------------------------------
create table payment (
  id             uuid primary key default gen_random_uuid(),
  appointment_id uuid not null references appointment(id) on delete cascade,
  stripe_intent  text unique,        -- simulated Stripe PaymentIntent id; unique = idempotency key
  status         text not null default 'pending' check (status in ('pending','succeeded','failed','refunded')),
  amount_paisa   int not null,       -- INR, integer paisa (₹ × 100) — never a float
  created_at     timestamptz not null default now()
);
create index idx_payment_appt on payment(appointment_id);

-- 2.10 queue_entry ------------------------------------------------------
create table queue_entry (
  id                uuid primary key default gen_random_uuid(),
  appointment_id    uuid not null references appointment(id) on delete cascade,
  doctor_id         uuid not null references doctor(id),
  position          int not null,
  est_wait_min      int not null default 0,
  appointment_date  date not null,
  created_at        timestamptz not null default now()
);
create unique index uq_queue_appt on queue_entry(appointment_id);
create index idx_queue_doctor_date on queue_entry(doctor_id, appointment_date);

-- 2.11 audit_log ----------------------------------------------------------
create table audit_log (
  id       uuid primary key default gen_random_uuid(),
  event    text not null,
  ts       timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);
create index idx_audit_event_ts on audit_log(event, ts desc);

-- 3.1 book_slot — call via supabase.rpc('book_slot', {...}) when the patient confirms a slot
create or replace function book_slot(
  p_slot_id         uuid,
  p_patient_id      uuid,
  p_chat_session_id uuid,
  p_department      text,
  p_triage_level    int,
  p_confidence      float
) returns uuid
language plpgsql
as $$
declare
  v_appointment_id uuid;
begin
  -- Serializes concurrent booking attempts on this slot; auto-released at transaction end.
  perform pg_advisory_xact_lock(hashtext(p_slot_id::text));

  if not exists (select 1 from clinician_slot where id = p_slot_id and status = 'open') then
    raise exception 'SLOT_NOT_AVAILABLE';
  end if;

  update clinician_slot set status = 'held', held_at = now() where id = p_slot_id;

  insert into appointment (patient_id, chat_session_id, slot_id, department, triage_level, confidence, status)
  values (p_patient_id, p_chat_session_id, p_slot_id, p_department, p_triage_level, p_confidence, 'pending_slot')
  returning id into v_appointment_id;

  return v_appointment_id;
end;
$$;

-- 3.2 release_stale_holds — call every 60s from a background asyncio task in the backend
create or replace function release_stale_holds(p_max_age_minutes int default 10)
returns int
language plpgsql
as $$
declare
  v_count int;
begin
  -- Global lock so this stays safe even if the backend ever scales to multiple instances.
  perform pg_advisory_xact_lock(hashtext('release_stale_holds'));

  update clinician_slot
    set status = 'open', held_at = null
    where status = 'held' and held_at < now() - (p_max_age_minutes || ' minutes')::interval;

  get diagnostics v_count = row_count;
  return v_count;
end;
$$;

-- 3.3 cancel_appointment — atomic cancel + queue reflow
create or replace function cancel_appointment(p_appointment_id uuid)
returns void
language plpgsql
as $$
declare
  v_slot_id   uuid;
  v_doctor_id uuid;
  v_position  int;
  v_appt_date date;
begin
  select slot_id into v_slot_id from appointment where id = p_appointment_id;

  select doctor_id, position, appointment_date
    into v_doctor_id, v_position, v_appt_date
    from queue_entry where appointment_id = p_appointment_id;

  if v_doctor_id is not null then
    update queue_entry
      set position = position - 1
      where doctor_id = v_doctor_id
        and appointment_date = v_appt_date
        and position > v_position;

    delete from queue_entry where appointment_id = p_appointment_id;
  end if;

  update clinician_slot set status = 'cancelled' where id = v_slot_id;
  update appointment set status = 'cancelled' where id = p_appointment_id;
end;
$$;

alter table patient enable row level security;
alter table appointment enable row level security;
alter table chat_session enable row level security;
alter table medical_history enable row level security;

create policy deny_all_patient on patient for all using (false);
create policy deny_all_appointment on appointment for all using (false);
create policy deny_all_chat_session on chat_session for all using (false);
create policy deny_all_medical_history on medical_history for all using (false);

insert into specialty (name) values
  ('Cardiology'), ('Dermatology'), ('Orthopedics'), ('Gastroenterology'),
  ('Neurology'), ('Pediatrics'), ('Psychiatry'), ('Respiratory'),
  ('General Medicine / Internal Medicine')
on conflict (name) do nothing;
