-- 0003_book_slot_enqueue.sql
--
-- Fix for code-review issue #3: book_slot inserts an appointment in
-- 'pending_slot' status but never enqueues it. The doctor portal reads
-- exclusively from queue_entry, so booked patients were invisible to doctors.
--
-- Fix for code-review issue #11: cancel_appointment unconditionally marked
-- every slot as 'cancelled' — including held-but-unconfirmed slots, which
-- permanently removed them from circulation. Now we release a held/open slot
-- of an un-confirmed appointment back to 'open'.
--
-- This migration ADDS:
--   1. confirm_appointment(p_appointment_id) — atomically moves an appointment
--      from 'pending_slot' into the doctor's queue. Called by
--      app/core/triage_graph.py::node_process_payment on successful payment.
--   2. Replaces book_slot with the same signature + an extra status flip
--      on the slot ('held' stays — confirm_appointment promotes it to 'booked').
--   3. Replaces cancel_appointment so released slots go back to 'open'
--      when the appointment had not yet been enqueued.

-- 3.1 (revised) book_slot — unchanged signature; just recreates for clarity.
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

-- 3.4 confirm_appointment — promote a 'pending_slot' appointment into the
-- doctor's queue. Called after the payment succeeds so the doctor portal
-- (which reads from queue_entry) immediately sees the booked patient.
create or replace function confirm_appointment(p_appointment_id uuid)
returns void
language plpgsql
as $$
declare
  v_slot_id       uuid;
  v_doctor_id     uuid;
  v_appt_date     date;
  v_start_time    timestamptz;
  v_triage_level  int;
  v_position      int;
  v_max_pos       int;
  v_est_wait      int;
  v_status        text;
  v_avg_consult   float;
begin
  perform pg_advisory_xact_lock(hashtext('confirm:' || p_appointment_id::text));

  -- Load appointment + slot info. Lock the appointment row for the txn.
  select a.slot_id, a.triage_level, a.status, s.doctor_id, s.start_time
    into v_slot_id, v_triage_level, v_status, v_doctor_id, v_start_time
    from appointment a
    join clinician_slot s on s.id = a.slot_id
    where a.id = p_appointment_id
    for update of a;

  if v_appointment_id is null and p_appointment_id is null then
    raise exception 'APPOINTMENT_NOT_FOUND';
  end if;

  if v_status is null then
    raise exception 'APPOINTMENT_NOT_FOUND';
  end if;

  -- Idempotent: if already in_queue, do nothing. Prevents double-enqueue on
  -- payment retry.
  if v_status = 'in_queue' then
    return;
  end if;

  if v_status <> 'pending_slot' then
    raise exception 'APPOINTMENT_NOT_CONFIRMABLE';
  end if;

  -- Reject if the slot was lost (released by release_stale_holds) between
  -- book_slot and confirm_appointment.
  if not exists (select 1 from clinician_slot where id = v_slot_id and status in ('held', 'booked')) then
    raise exception 'SLOT_NOT_AVAILABLE';
  end if;

  v_appt_date := v_start_time::date;

  -- Position = current max + 1 for this doctor on this date (or 1 if empty).
  select coalesce(max(position), 0) into v_max_pos
    from queue_entry
    where doctor_id = v_doctor_id and appointment_date = v_appt_date;
  v_position := v_max_pos + 1;

  -- Estimated wait = position * avg_consult_min for the doctor (caps the floor
  -- at 15 min if doctor has no history).
  select avg_consult_min into v_avg_consult from doctor where id = v_doctor_id;
  v_est_wait := v_position * coalesce(v_avg_consult, 15);

  -- Promote the slot to 'booked' and the appointment into the queue.
  update clinician_slot set status = 'booked', held_at = null where id = v_slot_id;
  update appointment set status = 'in_queue' where id = p_appointment_id;

  insert into queue_entry (appointment_id, doctor_id, position, est_wait_min, appointment_date)
  values (p_appointment_id, v_doctor_id, v_position, v_est_wait, v_appt_date);
end;
$$;

-- 3.3 (revised) cancel_appointment — release a held/open slot back to 'open'
-- if the appointment had not yet been enqueued. Confirmed (in_queue) bookings
-- cancel their slot as before (the patient had been promised a time and the
-- doctor's calendar reflects it).
create or replace function cancel_appointment(p_appointment_id uuid)
returns void
language plpgsql
as $$
declare
  v_slot_id        uuid;
  v_status         text;
  v_doctor_id      uuid;
  v_position       int;
  v_appt_date      date;
  v_in_queue       boolean;
begin
  -- Lock the appointment row to prevent a race with confirm_appointment.
  select slot_id, status
    into v_slot_id, v_status
    from appointment
    where id = p_appointment_id
    for update;

  if v_slot_id is null then
    return;  -- Nothing to cancel (already deleted or never existed).
  end if;

  v_in_queue := (v_status = 'in_queue');

  if v_in_queue then
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

    -- Was confirmed → slot was 'booked' → mark cancelled.
    update clinician_slot set status = 'cancelled' where id = v_slot_id;
  else
    -- Was only held (pending_slot, never confirmed) → release the slot back
    -- to 'open' so another patient can take it.
    update clinician_slot set status = 'open', held_at = null
      where id = v_slot_id and status = 'held';
  end if;

  update appointment set status = 'cancelled' where id = p_appointment_id;
end;
$$;
