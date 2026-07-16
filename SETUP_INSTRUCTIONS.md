# TriagePlus Setup Instructions - Synthetic Data

## Overview

You now have **3 tools** to populate your database with synthetic doctors and patients:

1. **SQL Script** (Fastest - Recommended) ⭐
2. **Python Script** (Programmatic)
3. **Verification Script** (Check data)

---

## Method 1: SQL Script (Recommended)

### Fastest Way to Populate Data

**Time:** ~2 minutes  
**Requirements:** Supabase account only

### Steps:

1. Open your Supabase project dashboard
2. Go to **SQL Editor** (left sidebar)
3. Click **"New Query"**
4. Copy the entire content from:
   ```
   /scripts/generate_synthetic_data.sql
   ```
5. Paste it into the SQL editor
6. Click **"Run"** button (⏵️)
7. Wait for completion (you'll see a summary table at the bottom)

### What Gets Created:
- ✅ 15 doctors with specialties and ratings
- ✅ ~1,500 appointment slots (next 7 days, excluding weekends)
- ✅ 10 patients with realistic Indian names
- ✅ 10 medical histories with conditions and medications

### Verify Results:
You should see output like:
```
entity              | count
--------------------|------
Doctors             | 15
Appointment Slots   | 1500+
Patients            | 10
Medical Histories   | 10
```

---

## Method 2: Python Script

### For Programmatic Control

**Time:** ~5 minutes  
**Requirements:** Python, pip/uv

### Steps:

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create Python virtual environment (if needed):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   cd ..  # Back to project root
   python scripts/generate_synthetic_doctors.py
   ```

5. Monitor output:
   ```
   🚀 TriagePlus Synthetic Data Generator
   ✅ Connected to Supabase
   📝 Generating synthetic doctors...
   ✅ Created doctor: Dr. Rajesh Kumar (Cardiology) - ID: ...
   ... (more doctors)
   ✅ Data generation complete!
   ```

---

## Method 3: Verify Data

### Check What Was Inserted

**After using Method 1 or 2**, verify the data:

### Option A: View in Supabase Console

1. Go to **Supabase Dashboard** → Your Project
2. Navigate to each table:
   - **doctor** - Should show 15 rows
   - **clinician_slot** - Should show 1500+ rows
   - **patient** - Should show 10 rows
   - **medical_history** - Should show 10 rows

### Option B: Run Python Verification Script

```bash
python scripts/verify_data.py
```

Output will show:
```
🔍 TriagePlus Data Verification
✅ Connected to Supabase

👨‍⚕️  DOCTORS
✅ Found 15 doctors:
  1. Dr. Rajesh Kumar - ⭐ 4.8
  2. Dr. Priya Sharma - ⭐ 4.6
  ... (more doctors)

📅 APPOINTMENT SLOTS
✅ Found 1500+ open appointment slots

👥 PATIENTS
✅ Found 10 patients:
  1. Rajesh Kumar (45y, Male) - +919876543210
  ... (more patients)

📊 SUMMARY
  Doctors:           15
  Appointment Slots: 1500+
  Patients:          10
  Medical Histories: 10
✅ All data populated successfully!
```

---

## Test the System

### 1. View Doctors in Backend

Access the doctor API:
```bash
curl http://localhost:8000/api/v1/public/doctors
```

Should return list of all 15 doctors with their specialties.

### 2. Check Available Slots

```bash
curl http://localhost:8000/api/v1/public/doctor/\{doctor_id\}/slots
```

Should return available appointment slots for that doctor.

### 3. Start a Chat Triage

1. Open frontend at `http://localhost:5173`
2. Start a new chat conversation
3. Go through the triage process
4. Select a doctor - should see the 15 synthetic doctors
5. Book an appointment - should see available slots

### 4. Doctor Portal Login

Once you have doctors created, you can:
- Create test user accounts for doctors
- Login to doctor portal to see patient queues
- View appointments and manage schedules

---

## Data Details

### Doctors (15 total)
All have realistic Indian names and varied specialties:

| Doctor | Specialty | Rating | Consult Min |
|--------|-----------|--------|-------------|
| Dr. Rajesh Kumar | Cardiology | 4.8 | 30 |
| Dr. Priya Sharma | Dermatology | 4.6 | 20 |
| Dr. Amit Patel | Orthopedics | 4.7 | 25 |
| ... | ... | ... | ... |

### Appointment Slots
- **Per doctor:** ~100 slots
- **Timing:** 9:00 AM to 5:00 PM
- **Interval:** Every 30 minutes
- **Days:** Next 7 days (weekdays only)
- **Status:** All "open"

Example slot times:
- Monday 9:00 AM
- Monday 9:30 AM
- Monday 10:00 AM
- ... etc

### Patients (10 total)
Realistic Indian patient data:

| Name | Age | Gender | Contact |
|------|-----|--------|---------|
| Rajesh Kumar | 45 | Male | +919876543210 |
| Priya Singh | 32 | Female | +918765432109 |
| ... | ... | ... | ... |

### Medical Histories
Each patient has:
- **Conditions:** e.g., "Diabetes, Hypertension"
- **Medications:** e.g., "Metformin, Lisinopril"
- **Allergies:** e.g., "Penicillin", "Shellfish"
- **Immunocompromised:** true/false

---

## Customization

### Add More Doctors

In SQL script, add more INSERT statements:
```sql
INSERT INTO doctor (name, specialty_id, rating, avg_consult_min) VALUES
  ('Dr. New Doctor', (SELECT id FROM specialty WHERE name = 'Cardiology'), 4.5, 25);
```

### Adjust Appointment Times

Edit the slot generation in SQL (change hours):
```sql
FOR v_hour IN 9..16 LOOP  -- Change 16 to 17 for 5pm slots, etc
```

### Change Patient Data

Edit the patient INSERT statements directly with new names and contact info.

---

## Troubleshooting

### Issue: "Specialty not found" error
**Solution:** Ensure specialty table is populated. Check:
```sql
SELECT * FROM specialty;
```

If empty, they're created automatically in the schema. Rerun the SQL script.

### Issue: No appointment slots appearing
**Solution:** 
1. Verify doctors were created (check doctor table)
2. Check if slot loop ran (slots take a moment to appear)
3. Run verification script to confirm

### Issue: Python script connection error
**Solution:**
1. Check `.env.development.local` has correct Supabase credentials
2. Verify `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set
3. Test connection with verification script

### Issue: Data appears duplicate or corrupt
**Solution:** Clear and regenerate:
```sql
DELETE FROM medical_history;
DELETE FROM clinician_slot;
DELETE FROM patient;
DELETE FROM doctor;
-- Then rerun the generation script
```

---

## Next Steps

After data is populated:

1. ✅ **Test doctor portal** - View patient queues
2. ✅ **Start triage conversations** - See populated doctor list
3. ✅ **Book appointments** - Select from available slots
4. ✅ **Monitor dashboard** - See metrics update
5. ✅ **Test payment flow** - Complete end-to-end booking

---

## Files Reference

| File | Purpose |
|------|---------|
| `/scripts/generate_synthetic_data.sql` | SQL script for Supabase |
| `/scripts/generate_synthetic_doctors.py` | Python script for programmatic generation |
| `/scripts/verify_data.py` | Verification script to check data |
| `/SYNTHETIC_DATA_GUIDE.md` | Detailed guide with examples |
| `/SETUP_INSTRUCTIONS.md` | This file |

---

## Questions?

- Check `SYNTHETIC_DATA_GUIDE.md` for detailed documentation
- Review backend logs: `cd backend && python -m uvicorn app.main:app --reload`
- Check Supabase dashboard for any errors

**You're all set! Start with Method 1 (SQL Script) - it's the fastest.** ⭐
