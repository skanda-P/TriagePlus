drop table if exists audit_log cascade;
drop table if exists queue_entry cascade;
drop table if exists payment cascade;
drop table if exists feedback cascade;
drop table if exists appointment cascade;
drop table if exists medical_history cascade;
drop table if exists clinician_slot cascade;
drop table if exists patient cascade;
drop table if exists doctor cascade;
drop table if exists specialty cascade;

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
  auth_user_id uuid references auth.users(id),   -- links to Supabase Auth
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
  auth_user_id uuid references auth.users(id),  -- links to Supabase Auth
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
  confidence float,                     -- classifier confidence
  status text not null default 'pending_slot'
    check (status in ('pending_slot','scheduled','in_queue','in_consult','completed','cancelled')),
  created_at timestamptz not null default now()
);
create index idx_appt_patient on appointment(patient_id);
create index idx_appt_slot on appointment(slot_id);
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

create table audit_log (
  id uuid primary key default gen_random_uuid(),
  event text not null,
  ts timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);
create index idx_audit_event_ts on audit_log(event, ts desc);
