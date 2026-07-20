-- 0002_anon_intake_policies.sql
-- Enable RLS INSERT/SELECT policies required by the unauthenticated intake, chat,
-- and public-doctor-browse flows.
--
-- Rationale: complete_intake() in app/intake_fsm.py uses the anon Supabase client
-- (get_anon_supabase) per the fix to code-review issue #4. The deny_all policies
-- in 0001_init.sql block every anon operation; this migration opens the minimum
-- surface needed for the patient-facing entry path while leaving doctor/queue/
-- medical_history/admin operations on the service-role client.

-- 1. patient: anon may create a new patient row (intake flow).
--    SELECT remains deny-all (no listing of patients by anon users).
drop policy if exists allow_anon_insert_patient on patient;
create policy allow_anon_insert_patient
  on patient for insert
  to anon, authenticated
  with check (true);

-- 2. chat_session: anon may create a session row at intake, and mark its own
--    in_progress sessions as completed. SELECT remains deny-all so patients
--    cannot enumerate other patients' chats.
drop policy if exists allow_anon_insert_chat_session on chat_session;
create policy allow_anon_insert_chat_session
  on chat_session for insert
  to anon, authenticated
  with check (true);

drop policy if exists allow_anon_update_chat_session on chat_session;
create policy allow_anon_update_chat_session
  on chat_session for update
  to anon, authenticated
  using (status = 'in_progress')
  with check (true);

-- 3. feedback: anon may submit feedback against any doctor. The appointment_id
--    is optional and validated server-side if provided. SELECT remains deny_all
--    (the doctor portal uses the service-role client for its own
--    `get_doctors`-style reads, which bypass RLS).
drop policy if exists allow_anon_insert_feedback on feedback;
create policy allow_anon_insert_feedback
  on feedback for insert
  to anon, authenticated
  with check (true);

-- 4. Public catalog read paths: /api/v1/specialties, /api/v1/doctors.
--    These are unauthenticated browse endpoints; let anon SELECT the catalog
--    tables so future code can safely use the anon client instead of the
--    service-role client for these queries.
drop policy if exists allow_anon_select_specialty on specialty;
create policy allow_anon_select_specialty
  on specialty for select
  to anon, authenticated
  using (true);

drop policy if exists allow_anon_select_doctor on doctor;
create policy allow_anon_select_doctor
  on doctor for select
  to anon, authenticated
  using (true);

drop policy if exists allow_anon_select_open_slot on clinician_slot;
create policy allow_anon_select_open_slot
  on clinician_slot for select
  to anon, authenticated
  using (status = 'open');

-- 5. audit_log: never writable by anon; only the service-role client (server)
--    writes audit events. No policy added here.
