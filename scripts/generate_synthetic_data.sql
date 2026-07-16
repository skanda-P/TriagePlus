-- TriagePlus Synthetic Data Generator
-- Run this in Supabase SQL Editor to populate doctors, patients, and appointment slots

-- Step 1: Clear existing data (optional - comment out if you want to keep old data)
-- DELETE FROM clinician_slot WHERE doctor_id IN (SELECT id FROM doctor WHERE created_at > NOW() - INTERVAL '1 hour');
-- DELETE FROM doctor WHERE created_at > NOW() - INTERVAL '1 hour';
-- DELETE FROM appointment WHERE created_at > NOW() - INTERVAL '1 hour';
-- DELETE FROM patient WHERE created_at > NOW() - INTERVAL '1 hour';

-- Step 2: Insert 15 synthetic doctors
INSERT INTO doctor (name, specialty_id, rating, avg_consult_min) VALUES
  ('Dr. Rajesh Kumar', (SELECT id FROM specialty WHERE name = 'Cardiology'), 4.8, 30),
  ('Dr. Priya Sharma', (SELECT id FROM specialty WHERE name = 'Dermatology'), 4.6, 20),
  ('Dr. Amit Patel', (SELECT id FROM specialty WHERE name = 'Orthopedics'), 4.7, 25),
  ('Dr. Neha Singh', (SELECT id FROM specialty WHERE name = 'Gastroenterology'), 4.5, 30),
  ('Dr. Vikram Gupta', (SELECT id FROM specialty WHERE name = 'Neurology'), 4.9, 45),
  ('Dr. Anjali Verma', (SELECT id FROM specialty WHERE name = 'Pediatrics'), 4.8, 20),
  ('Dr. Sanjay Reddy', (SELECT id FROM specialty WHERE name = 'Psychiatry'), 4.4, 45),
  ('Dr. Pooja Desai', (SELECT id FROM specialty WHERE name = 'Respiratory'), 4.7, 25),
  ('Dr. Arjun Nair', (SELECT id FROM specialty WHERE name = 'General Medicine / Internal Medicine'), 4.6, 20),
  ('Dr. Sneha Menon', (SELECT id FROM specialty WHERE name = 'Cardiology'), 4.9, 30),
  ('Dr. Rohan Sharma', (SELECT id FROM specialty WHERE name = 'Orthopedics'), 4.5, 25),
  ('Dr. Divya Iyer', (SELECT id FROM specialty WHERE name = 'Dermatology'), 4.8, 20),
  ('Dr. Arun Pillai', (SELECT id FROM specialty WHERE name = 'Neurology'), 4.7, 45),
  ('Dr. Ravi Chandran', (SELECT id FROM specialty WHERE name = 'General Medicine / Internal Medicine'), 4.4, 20),
  ('Dr. Meera Chopra', (SELECT id FROM specialty WHERE name = 'Pediatrics'), 4.6, 20)
ON CONFLICT DO NOTHING;

-- Step 3: Create appointment slots for each doctor (7 days, 8am-6pm)
-- This creates ~100+ slots per doctor
DO $$
DECLARE
  v_doctor_id UUID;
  v_slot_time TIMESTAMPTZ;
  v_day_offset INT;
  v_hour INT;
  v_minute INT;
  doctor_cursor CURSOR FOR SELECT id FROM doctor ORDER BY created_at DESC LIMIT 15;
BEGIN
  OPEN doctor_cursor;
  LOOP
    FETCH doctor_cursor INTO v_doctor_id;
    EXIT WHEN v_doctor_id IS NULL;

    -- Create slots for next 7 days
    FOR v_day_offset IN 1..7 LOOP
      -- Skip weekends
      IF EXTRACT(DOW FROM NOW() + (v_day_offset || ' days')::INTERVAL) NOT IN (0, 6) THEN
        -- Create slots at 9am, 9:30am, 10am, 10:30am... until 5pm
        FOR v_hour IN 9..16 LOOP
          FOR v_minute IN 0..30 BY 30 LOOP
            v_slot_time := NOW() + (v_day_offset || ' days')::INTERVAL;
            v_slot_time := v_slot_time + (v_hour || ' hours')::INTERVAL + (v_minute || ' minutes')::INTERVAL;
            
            INSERT INTO clinician_slot (doctor_id, start_time, status)
            VALUES (v_doctor_id, v_slot_time, 'open')
            ON CONFLICT DO NOTHING;
          END LOOP;
        END LOOP;
      END IF;
    END LOOP;
  END LOOP;
  CLOSE doctor_cursor;
END $$;

-- Step 4: Insert 10 synthetic patients
INSERT INTO patient (name, age, gender, contact, language) VALUES
  ('Rajesh Kumar', 45, 'Male', '+919876543210', 'en'),
  ('Priya Singh', 32, 'Female', '+918765432109', 'hi'),
  ('Amit Patel', 58, 'Male', '+917654321098', 'en'),
  ('Neha Desai', 28, 'Female', '+916543210987', 'ta'),
  ('Vikram Nair', 52, 'Male', '+915432109876', 'en'),
  ('Anjali Iyer', 35, 'Female', '+914321098765', 'te'),
  ('Sanjay Reddy', 41, 'Male', '+913210987654', 'en'),
  ('Pooja Sharma', 29, 'Female', '+912109876543', 'hi'),
  ('Arjun Menon', 50, 'Male', '+911098765432', 'en'),
  ('Sneha Pillai', 27, 'Female', '+919988776655', 'ta')
ON CONFLICT (contact) DO NOTHING;

-- Step 5: Create medical histories for patients
INSERT INTO medical_history (patient_id, conditions, medications, allergies, immunocompromised) 
SELECT 
  id,
  CASE (ROW_NUMBER() OVER (ORDER BY RANDOM())) % 5
    WHEN 1 THEN 'Diabetes, Hypertension'
    WHEN 2 THEN 'Asthma'
    WHEN 3 THEN 'Arthritis'
    WHEN 4 THEN 'Migraine'
    ELSE NULL
  END,
  CASE (ROW_NUMBER() OVER (ORDER BY RANDOM())) % 4
    WHEN 1 THEN 'Metformin, Lisinopril'
    WHEN 2 THEN 'Albuterol inhaler'
    WHEN 3 THEN 'Ibuprofen PRN'
    ELSE NULL
  END,
  CASE (ROW_NUMBER() OVER (ORDER BY RANDOM())) % 3
    WHEN 1 THEN 'Penicillin'
    WHEN 2 THEN 'Shellfish'
    ELSE NULL
  END,
  (ROW_NUMBER() OVER (ORDER BY RANDOM())) % 4 = 0
FROM patient
ON CONFLICT (patient_id) DO NOTHING;

-- Step 6: Verify the data
SELECT 'Doctors' as entity, COUNT(*) as count FROM doctor
UNION ALL
SELECT 'Appointment Slots', COUNT(*) FROM clinician_slot
UNION ALL
SELECT 'Patients', COUNT(*) FROM patient
UNION ALL
SELECT 'Medical Histories', COUNT(*) FROM medical_history;

-- Show doctors with their specialties and slot count
SELECT 
  d.name,
  s.name as specialty,
  d.rating,
  d.avg_consult_min,
  COUNT(cs.id) as available_slots
FROM doctor d
JOIN specialty s ON d.specialty_id = s.id
LEFT JOIN clinician_slot cs ON d.id = cs.doctor_id AND cs.status = 'open'
GROUP BY d.id, d.name, s.name, d.rating, d.avg_consult_min
ORDER BY d.created_at DESC;
