# TriagePlus - Quick Start Data Population

## 30-Second Setup (Recommended)

### Generate Doctor & Patient Data

**Method 1: SQL (Fastest) ⭐**

1. Go to Supabase Dashboard → SQL Editor
2. Copy all content from: `/scripts/generate_synthetic_data.sql`
3. Paste into SQL editor
4. Click "Run"
5. Done! ✅

**Creates:**
- 15 doctors
- 1,500+ appointment slots
- 10 patients
- 10 medical histories

### Verify It Worked

Open Supabase → Tables:
- `doctor` → Should show 15 rows
- `clinician_slot` → Should show 1500+ rows
- `patient` → Should show 10 rows

---

## Testing the System

### 1. Start Frontend
```bash
cd frontend
npm start
```

### 2. Open Chat
Go to http://localhost:5173

### 3. Start Triage
- Answer health questions
- System assigns triage level
- See list of 15 doctors

### 4. Book Appointment
- Select doctor
- Choose from available slots
- Complete booking

---

## Alternative: Python Script

If you prefer programmatic generation:

```bash
python scripts/generate_synthetic_doctors.py
```

---

## Verify Data Anytime

```bash
python scripts/verify_data.py
```

Shows count of:
- Doctors
- Slots
- Patients
- Medical histories

---

## What's Included

### 15 Doctors
- Various specialties (Cardiology, Dermatology, Neurology, etc.)
- Ratings 4.4-4.9
- Realistic Indian names

### 1,500+ Appointment Slots
- Next 7 days
- 9 AM - 5 PM
- 30-minute intervals
- Weekdays only

### 10 Patients
- Indian names and contacts
- Ages 27-58
- Medical conditions, medications, allergies

---

## Need More Details?

- **Full Setup Guide:** See `SETUP_INSTRUCTIONS.md`
- **Data Details & Customization:** See `SYNTHETIC_DATA_GUIDE.md`

---

## Troubleshooting

**No slots appearing?**
- Verify doctors were created first
- Slots may take a moment to populate

**Connection error?**
- Check Supabase credentials in `.env.development.local`

**Want to clear old data?**
```sql
DELETE FROM doctor;
DELETE FROM patient;
DELETE FROM clinician_slot;
DELETE FROM appointment;
DELETE FROM medical_history;
-- Then rerun the SQL script
```

---

**You're ready to go! Start with the SQL script method above.** ✅
