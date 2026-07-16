# TriagePlus Synthetic Data Generation Guide

This guide explains how to generate and populate synthetic doctor and patient data into your Supabase database.

## Quick Start (Recommended - SQL Method)

The fastest and easiest way to populate your database with synthetic data is using the SQL script directly in Supabase.

### Steps:

1. **Open Supabase Console**
   - Go to https://supabase.com → Your Project → SQL Editor

2. **Copy and Run the SQL Script**
   - Open `/scripts/generate_synthetic_data.sql` in your text editor
   - Copy all the SQL code
   - Paste it into the Supabase SQL Editor
   - Click "Run" button

3. **Wait for completion**
   - The script will create:
     - **15 doctors** with different specialties
     - **~1,500 appointment slots** (100+ per doctor across 7 days)
     - **10 patients** with realistic data
     - **10 medical histories** with conditions and medications

4. **Verify the data**
   - The script shows a summary table at the end
   - You should see:
     ```
     Doctors: 15
     Appointment Slots: 1500+
     Patients: 10
     Medical Histories: 10
     ```

## Alternative Method - Python Script

If you prefer to generate data programmatically:

### Requirements:
- Python 3.8+
- `supabase` package (already in your requirements)

### Steps:

1. **Install dependencies** (if needed)
   ```bash
   cd backend
   pip install faker  # or pip install --system faker
   ```

2. **Run the Python script**
   ```bash
   cd backend
   cd ..  # back to root
   python scripts/generate_synthetic_doctors.py
   ```

3. **Script output**
   - Creates 15 doctors with various specialties
   - Generates appointment slots for each doctor
   - Creates 10 patients with medical histories
   - Shows verification counts

## Data Details

### Doctors (15 total)
- Names: Indian doctor names (Dr. Rajesh Kumar, Dr. Priya Sharma, etc.)
- Specialties: 
  - Cardiology, Dermatology, Orthopedics, Gastroenterology
  - Neurology, Pediatrics, Psychiatry, Respiratory
  - General Medicine / Internal Medicine
- Ratings: 4.4-4.9 (out of 5.0)
- Consultation duration: 15-45 minutes

### Appointment Slots
- **Per doctor:** ~100 slots
- **Time range:** 9:00 AM - 5:00 PM
- **Interval:** 30 minutes
- **Days:** Next 7 days (excluding weekends)
- **Status:** All start as "open"

### Patients (10 total)
- Names: Realistic Indian names
- Age: 27-58 years
- Gender: Mix of Male, Female, Other
- Contact: Indian phone numbers (+91...)
- Language: English, Hindi, Tamil, Telugu

### Medical Histories
- **Conditions:** Diabetes, Hypertension, Asthma, Arthritis, Migraine
- **Medications:** Metformin, Lisinopril, Albuterol, Ibuprofen, etc.
- **Allergies:** Penicillin, Shellfish, Latex
- **Immunocompromised:** Mix of true/false

## Customizing the Data

### SQL Script Customization:

1. **Change number of doctors:**
   - Modify the INSERT statements to add/remove doctor rows

2. **Adjust slot timing:**
   - Change the `FOR v_hour IN 9..16` to adjust hours
   - Change slot intervals by modifying the `FOR v_minute IN 0..30 BY 30` line

3. **Modify patient data:**
   - Edit the patient INSERT statements directly

### Python Script Customization:

Edit `/scripts/generate_synthetic_doctors.py`:

```python
# Change number of doctors
doctors = create_synthetic_doctors(num_doctors=20)

# Change number of patients
patients = create_synthetic_patients(num_patients=15)

# Change slot creation
create_slots_for_doctor(supabase, doctor_id, doctor["name"], num_slots=30)
```

## Clearing Old Data

If you want to remove previously generated data:

### SQL Method:
Uncomment the DELETE statements at the top of `/scripts/generate_synthetic_data.sql`:

```sql
-- Step 1: Clear existing data (uncomment to enable)
DELETE FROM clinician_slot WHERE doctor_id IN (SELECT id FROM doctor WHERE created_at > NOW() - INTERVAL '1 hour');
DELETE FROM doctor WHERE created_at > NOW() - INTERVAL '1 hour';
DELETE FROM appointment WHERE created_at > NOW() - INTERVAL '1 hour';
DELETE FROM patient WHERE created_at > NOW() - INTERVAL '1 hour';
```

### Manual SQL:
```sql
DELETE FROM doctor;
DELETE FROM patient;
DELETE FROM clinician_slot;
DELETE FROM appointment;
DELETE FROM medical_history;
```

## Testing the Data

Once data is generated, you can test it:

### 1. **View Doctors**
```bash
# In Supabase SQL Editor:
SELECT d.id, d.name, s.name as specialty, d.rating, COUNT(cs.id) as slots
FROM doctor d
JOIN specialty s ON d.specialty_id = s.id
LEFT JOIN clinician_slot cs ON d.id = cs.doctor_id
GROUP BY d.id, d.name, s.name, d.rating;
```

### 2. **Check Availability**
```bash
SELECT 
  d.name,
  COUNT(CASE WHEN cs.status = 'open' THEN 1 END) as open_slots,
  COUNT(CASE WHEN cs.status = 'booked' THEN 1 END) as booked_slots
FROM doctor d
LEFT JOIN clinician_slot cs ON d.id = cs.doctor_id
GROUP BY d.id, d.name;
```

### 3. **View Patients**
```bash
SELECT id, name, age, gender, contact FROM patient LIMIT 10;
```

## Troubleshooting

### "Specialty not found" error
- Ensure specialties are created first:
```sql
INSERT INTO specialty (name) VALUES
  ('Cardiology'), ('Dermatology'), ('Orthopedics'), etc.
ON CONFLICT (name) DO NOTHING;
```

### Foreign key constraint error
- Ensure specialty IDs exist before inserting doctors
- Run the specialty INSERT first

### No slots appearing
- Check that the doctor was created successfully
- Verify the loop is creating slots (may take a moment for large numbers)

### Python script connection error
- Ensure `.env.development.local` has `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
- Check Supabase project credentials

## Next Steps

After generating data:

1. **Test the doctor portal** - Login and view available patients
2. **Start a triage conversation** - See doctors populated in booking flow
3. **Book an appointment** - Select from available slots
4. **Monitor the dashboard** - See metrics update with real data

For questions or issues, check the backend logs:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

---

**Generated data is for testing only.** Do not use in production systems handling real patients.
