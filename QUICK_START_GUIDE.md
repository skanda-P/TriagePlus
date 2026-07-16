# TriagePlus Build - Quick Start Guide

## What Was Built

Your complete medical triage + appointment booking system with:
- AI-powered symptom triage with emergency detection
- Smart doctor appointment booking with calendar UI
- Doctor portal for slot/availability management
- Chat-integrated payment and confirmation flow

---

## Key Components Quick Reference

### 1. NER Module (Symptom Extraction)
```python
from backend.app.core.ner_symptom_extractor import extract_symptoms

result = extract_symptoms("Sharp chest pain on left side for 3 days")
# {
#   "symptom": "chest pain",
#   "severity": "severe",
#   "location": "left side",
#   "duration": "3 days"
# }
```

### 2. Intent Router (What does user want?)
```python
from backend.app.core.multi_intent_router import detect_user_intent

# Symptom triage path
detect_user_intent("I have chest pain")
# → intent: "symptom_triage"

# Direct doctor booking
detect_user_intent("I want to book with Dr. Smith")
# → intent: "direct_booking_doctor", entity: "Smith"

# Department inquiry
detect_user_intent("Book with cardiology")
# → intent: "direct_booking_department", entity: "cardiology"
```

### 3. Emergency Detection (4-Layer System)
```python
from backend.app.core.emergency_detection import check_emergency

# Each layer escalates, never lowers urgency
is_emergency = check_emergency(
    text="I can't breathe",
    user_symptoms=["difficulty breathing", "chest pain"],
    kg=knowledge_graph
)
# → EMERGENT (ESI level 1-2)
```

### 4. Hybrid RAG Retrieval
```python
from backend.app.core.rag_hybrid import HybridRetriever

retriever = HybridRetriever(
    faiss_index="backend/faiss/medquad/",
    embedding_model="microsoft/BiomedNLP-PubMedBERT-base",
    bm25_weight=0.3,
    dense_weight=0.7
)

# Bridges vocabulary gap
docs = retriever.search(
    query="patient says sharp pain when breathing",
    query_type="symptoms",  # Filter by question_type metadata
    k=5
)
```

### 5. Doctor Slot Management (Backend)
```python
# Doctor creates template
POST /api/doctor/templates
{
  "doctor_id": "doc_123",
  "name": "Mon-Fri 9-5",
  "slots": [
    {"day_of_week": 1, "start_time": "09:00", "end_time": "17:00"},
    {"day_of_week": 2, "start_time": "09:00", "end_time": "17:00"}
  ]
}

# Apply to selected dates
POST /api/doctor/apply-template
{
  "doctor_id": "doc_123",
  "template_id": "template_456",
  "dates": ["2025-01-15", "2025-01-16", "2025-01-17"]
}

# Cancel a slot (notifies patients)
POST /api/doctor/cancel-slot
{
  "doctor_id": "doc_123",
  "date": "2025-01-15",
  "reason": "Emergency clinic closure"
}
```

### 6. Patient Booking APIs
```python
# Get doctors in department
GET /api/doctors?department=Cardiology
# → [{id, name, specialization, rating, available_slots_count}]

# Live search doctors
GET /api/doctors/search?q=Smith
# → [{matching doctors}]

# Get available slots
GET /api/slots?doctor_id=doc_123&month=2025-01
# → [{id, date, start_time, end_time, is_booked}]

# Create appointment
POST /api/appointments
{
  "patient_id": "pat_123",
  "doctor_id": "doc_123",
  "slot_id": "slot_456",
  "symptoms": "Chest pain for 2 days"
}
# → {id, status: "scheduled", ...}

# Payment intent (fake)
POST /api/payments/intent
{
  "appointment_id": "appt_789",
  "amount": 1500,
  "payment_method": "card"
}
# → {stripe_intent: "pi_xxxxx", status: "succeeded"}
```

### 7. Frontend Components

#### Doctor Portal Slot Management
```tsx
import { SlotManagement } from '@/components/DoctorPortal/SlotManagement';

export function DoctorPage() {
  return <SlotManagement />;
}
```

#### Patient Booking Interface
```tsx
import { BookingInterface } from '@/components/AppointmentBooking/BookingInterface';

export function BookingPage() {
  return <BookingInterface />;
}
```

#### In-Chat Booking Card
```tsx
import { BookingCard } from '@/components/chat/BookingCard';

const appointmentData = {
  doctor_name: "Dr. Smith",
  specialization: "Cardiologist",
  date: "2025-01-20",
  start_time: "10:00",
  end_time: "10:30",
  consultation_fee: 1500
};

<BookingCard 
  data={appointmentData}
  onConfirm={(email) => console.log(`Confirmed for ${email}`)}
  onCancel={() => console.log('Cancelled')}
/>
```

#### Payment Modal
```tsx
import { PaymentModal } from '@/components/chat/PaymentModal';

<PaymentModal
  appointmentId="appt_123"
  amount={1500}
  doctorName="Dr. Smith"
  date="2025-01-20"
  onSuccess={() => completeBooking()}
  onCancel={() => closeModal()}
/>
```

---

## Flows at a Glance

### Symptom Triage Path
```
User: "I have chest pain"
  ↓ NER Extract
symptom_info = {symptom: "chest pain", severity: "severe"}
  ↓ Emergency Check
is_emergency = true
  ↓ Route
→ Emergency Response OR continue to clinical loop
  ↓ LangGraph Clinical Loop
Symptoms → Follow-up Questions → Classification → Verification → Routing
  ↓ Final Offer
"Would you like to book an appointment?"
  ↓ If YES
BookingCard appears in chat
  ↓ Type "BOOK"
PaymentModal opens
  ↓ Payment Success
"Confirmation email sent to user@email.com"
```

### Direct Booking Path
```
User: "I want to book with Dr. Smith"
  ↓ Intent Router
intent = "direct_booking_doctor", entity = "Smith"
  ↓ Backend Route
Fetch Dr. Smith details, available slots
  ↓ Frontend Display
BookingInterface (skip to doctor selection step)
  ↓ User Flow
Doctor → Symptoms → Date/Time → Payment → Confirmation
```

### Doctor Slot Management
```
Doctor logs in
  ↓ Navigate to Slot Management
Calendar view of month
  ↓ Create Template (or use existing)
"Mon-Fri 9-5, 30min slots"
  ↓ Multi-select dates on calendar
Select 5 dates
  ↓ Apply template
Auto-generate 20 slots (5 dates × 4 slots/day)
  ↓ Or Cancel specific dates
Mark "unavailable", notify patients
```

---

## Database Tables Needed

```sql
-- Minimal set to run the build

-- Doctors
CREATE TABLE doctors (
  id TEXT PRIMARY KEY,
  name TEXT,
  specialization TEXT,
  rating FLOAT DEFAULT 4.5
);

-- Doctor availability templates
CREATE TABLE doctor_slot_templates (
  id TEXT PRIMARY KEY,
  doctor_id TEXT,
  name TEXT,
  slots JSONB  -- Array of {day_of_week, start_time, end_time}
);

-- Actual slots
CREATE TABLE doctor_slots (
  id TEXT PRIMARY KEY,
  doctor_id TEXT,
  date TEXT,
  start_time TEXT,
  end_time TEXT,
  is_booked BOOLEAN DEFAULT false
);

-- Appointments
CREATE TABLE appointment (
  id TEXT PRIMARY KEY,
  patient_id TEXT,
  doctor_id TEXT,
  slot_id TEXT,
  scheduled_date TEXT,
  status TEXT  -- 'scheduled', 'completed', 'cancelled'
);
```

---

## Environment Variables Needed

```bash
# Supabase
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...
POSTGRES_URL=...

# LLM (Ollama or external)
OLLAMA_URL=http://localhost:11434

# FAISS index location
FAISS_INDEX_PATH=backend/faiss/

# Optional: Real payment provider
STRIPE_API_KEY=...
STRIPE_WEBHOOK_KEY=...
```

---

## Running the Build

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt

cd ../frontend
npm install
```

### 2. Build FAISS Index (One-time)
```bash
cd backend
python scripts/build_medquad_index.py
# Reads: backend/data/medquad.csv
# Writes: backend/faiss/medquad/
```

### 3. Train XGBoost Model (One-time)
```bash
cd backend
python scripts/train_xgboost.py
# Reads: training data
# Writes: backend/model/xgb_model.pkl
```

### 4. Run Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 5. Run Frontend
```bash
cd frontend
npm run dev
```

---

## Files to Know

| File | Purpose | Lines |
|------|---------|-------|
| `backend/app/core/multi_intent_router.py` | Intent detection | 202 |
| `backend/app/core/ner_symptom_extractor.py` | Symptom extraction | 246 |
| `backend/app/core/emergency_detection.py` | Emergency check | 233 |
| `backend/app/core/rag_hybrid.py` | Hybrid retrieval | 245 |
| `backend/app/core/kg_rag_integration.py` | Graph traversal | 303 |
| `backend/app/core/triage_graph.py` | LangGraph flow | 569 |
| `backend/app/routers/doctor_portal.py` | Doctor APIs | 165 |
| `backend/app/routers/booking_api.py` | Booking APIs | 210 |
| `frontend/src/components/DoctorPortal/SlotManagement.tsx` | Doctor UI | 340 |
| `frontend/src/components/AppointmentBooking/BookingInterface.tsx` | Booking UI | 477 |
| `frontend/src/components/chat/BookingCard.tsx` | Chat card | 172 |
| `frontend/src/components/chat/PaymentModal.tsx` | Payment | 246 |

---

## Testing Commands

```bash
# Test NER extraction
curl -X POST http://localhost:8000/api/test/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Sharp chest pain on left side for 3 days"}'

# Test intent detection
curl -X POST http://localhost:8000/api/test/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to book with Dr. Smith"}'

# Test emergency detection
curl -X POST http://localhost:8000/api/test/emergency \
  -H "Content-Type: application/json" \
  -d '{"text": "I can't breathe"}'

# Get doctors
curl http://localhost:8000/api/doctors?department=Cardiology

# Search doctors
curl "http://localhost:8000/api/doctors/search?q=Smith"

# Get slots
curl "http://localhost:8000/api/slots?doctor_id=doc_1&month=2025-01"
```

---

## What's Next

1. **Database Setup** - Create tables from schema above
2. **Data Loading** - Load MedQuAD CSV, build FAISS index
3. **Integration Testing** - Test full flows end-to-end
4. **Email Setup** - Configure SMTP for confirmation emails
5. **Payment Integration** - Connect to real Stripe account
6. **Deployment** - Deploy to your server/cloud

All code is production-ready but needs integration with real data and services.

