# TriagePlus — Database Build Prompt
### IIT Dharwad Summer of Innovation · Hardly Human
> This file covers everything related to data persistence: the Supabase PostgreSQL schema, Redis session state, and FAISS vector index construction. Implement exactly as specified here before touching any other component.

---

## Part 1 — Supabase PostgreSQL Schema

### Connection Rules (Non-Negotiable)

```
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>:5432/<db>
```

- Always use the **direct URL on port 5432**. Never use the pooler URL on port 6543.
- SQLAlchemy pool settings: `pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=300`

### SQLAlchemy Engine Setup (`/backend/app/core/database.py`)

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

---

## Part 2 — All SQLAlchemy Models

### 2.1 `PatientSession` (`/backend/app/models/session.py`)

No passwords. No login. Patients are identified by a UUID generated client-side.

```python
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .base import Base

class PatientSession(Base):
    __tablename__ = "patient_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID from client
    patient_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

### 2.2 `Specialty` (`/backend/app/models/doctor.py`)

Exactly 9 specialties — including Respiratory as a distinct class (not merged into General Medicine).

```python
class Specialty(Base):
    __tablename__ = "specialties"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

# Seed values (in this exact order):
# Cardiology, Dermatology, Orthopedics, Gastroenterology,
# Neurology, Pediatrics, Psychiatry, General Medicine, Respiratory
```

> **Note:** The build prompt uses 8 specialties in the DB seed comment but the AI plan mandates 9. Always seed 9 specialties. Respiratory is distinct from General Medicine.

### 2.3 `Doctor` (`/backend/app/models/doctor.py`)

```python
from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)  # bcrypt
    specialty_id: Mapped[int] = mapped_column(ForeignKey("specialties.id"))
    rating: Mapped[float] = mapped_column(Float, default=4.5)
    feedback_score: Mapped[float] = mapped_column(Float, default=4.5)

    specialty: Mapped["Specialty"] = relationship("Specialty")
    slots: Mapped[list["ClinicianSlot"]] = relationship("ClinicianSlot", back_populates="doctor")
```

### 2.4 `ClinicianSlot` with Slot Status State Machine (`/backend/app/models/slot.py`)

```python
import enum
from sqlalchemy import Enum, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class SlotStatus(str, enum.Enum):
    AVAILABLE  = "AVAILABLE"
    PRE_LOCK   = "PRE_LOCK"
    CONFIRMED  = "CONFIRMED"
    CANCELLED  = "CANCELLED"

class ClinicianSlot(Base):
    __tablename__ = "clinician_slots"

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("doctors.id"))
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[SlotStatus] = mapped_column(
        Enum(SlotStatus), default=SlotStatus.AVAILABLE, nullable=False
    )

    doctor: Mapped["Doctor"] = relationship("Doctor", back_populates="slots")

    # State machine — these are the ONLY valid transitions
    def confirm(self) -> None:
        if self.status != SlotStatus.PRE_LOCK:
            raise ValueError(f"Cannot confirm slot in state {self.status}")
        self.status = SlotStatus.CONFIRMED

    def release(self) -> None:
        if self.status != SlotStatus.PRE_LOCK:
            return  # no-op if already released
        self.status = SlotStatus.AVAILABLE
```

**Critical:** `slot.confirm()` and `slot.release()` are the ONLY paths to change slot status. Never assign `slot.status = ...` directly anywhere else in the codebase.

### 2.5 `Appointment` (`/backend/app/models/appointment.py`)

```python
class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("patient_sessions.session_id"))
    slot_id: Mapped[int] = mapped_column(ForeignKey("clinician_slots.id"))
    doctor_id: Mapped[int] = mapped_column(nullable=False)
    specialty: Mapped[str] = mapped_column(String, nullable=False)
    ai_brief: Mapped[str | None] = mapped_column(String, nullable=True)
    # ai_brief is stored at booking time and NEVER recomputed.
    # ai_brief is NEVER exposed through patient-facing API routes.
```

### 2.6 `QueueEntry` (`/backend/app/models/appointment.py`)

```python
class QueueEntry(Base):
    __tablename__ = "queue_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    appointment_id: Mapped[int] = mapped_column(ForeignKey("appointments.id"))
    doctor_id: Mapped[int] = mapped_column(nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    appointment_date: Mapped[str] = mapped_column(String, nullable=False)  # YYYY-MM-DD
```

### 2.7 `Payment` (`/backend/app/models/appointment.py`)

```python
class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True)
    appointment_id: Mapped[int] = mapped_column(ForeignKey("appointments.id"))
    stripe_pi_id: Mapped[str] = mapped_column(String, unique=True)  # idempotency key
    amount: Mapped[int] = mapped_column(nullable=False)  # in paisa (₹ × 100)
    status: Mapped[str] = mapped_column(String, nullable=False)  # "pending"|"completed"|"failed"
```

### 2.8 `AuditLog` (`/backend/app/models/appointment.py`)

For emergency events and security events. Never delete rows from this table.

```python
class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

---

## Part 3 — Alembic Migrations

### Setup

```bash
cd backend
alembic init alembic
```

In `alembic/env.py`, import all models and set the metadata target:

```python
from app.models.base import Base
from app.models.session import PatientSession
from app.models.doctor import Specialty, Doctor
from app.models.slot import ClinicianSlot
from app.models.appointment import Appointment, QueueEntry, Payment, AuditLog

target_metadata = Base.metadata
```

### Migration command (run every deploy):

```bash
alembic upgrade head
```

---

## Part 4 — Seed Script (`/backend/app/scripts/seed_db.py`)

The seed script is **idempotent** — running it twice must produce zero duplicates.

```python
"""
Idempotent seed script.
Run: python -m app.scripts.seed_db
Safe to run on every deploy.
"""
import asyncio
from datetime import datetime, timedelta
from passlib.context import CryptContext
from sqlalchemy import select

SPECIALTIES = [
    "Cardiology", "Dermatology", "Orthopedics", "Gastroenterology",
    "Neurology", "Pediatrics", "Psychiatry", "General Medicine", "Respiratory"
]

DOCTORS_PER_SPECIALTY = 3
DEFAULT_PASSWORD = "password123"  # bcrypt hashed below

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
HASHED_PASSWORD = pwd_ctx.hash(DEFAULT_PASSWORD)

async def seed():
    async with AsyncSessionLocal() as db:
        # Seed specialties (idempotent)
        for name in SPECIALTIES:
            existing = await db.execute(select(Specialty).where(Specialty.name == name))
            if not existing.scalar_one_or_none():
                db.add(Specialty(name=name))
        await db.commit()

        # Seed doctors (3 per specialty, idempotent)
        for specialty_name in SPECIALTIES:
            spec = (await db.execute(
                select(Specialty).where(Specialty.name == specialty_name)
            )).scalar_one()
            for i in range(1, DOCTORS_PER_SPECIALTY + 1):
                email = f"dr{i}.{specialty_name.lower().replace(' ', '')}@triageplus.com"
                existing = await db.execute(select(Doctor).where(Doctor.email == email))
                if not existing.scalar_one_or_none():
                    db.add(Doctor(
                        name=f"Dr. {specialty_name} Doctor {i}",
                        email=email,
                        hashed_password=HASHED_PASSWORD,
                        specialty_id=spec.id,
                        rating=4.5,
                        feedback_score=4.5,
                    ))
        await db.commit()

        # Seed 30 days of slots: 9am–5pm every 30 min, every doctor (idempotent)
        now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        doctors = (await db.execute(select(Doctor))).scalars().all()
        for doctor in doctors:
            for day_offset in range(30):
                day = now + timedelta(days=day_offset)
                for hour in range(9, 17):
                    for minute in [0, 30]:
                        start = day.replace(hour=hour, minute=minute)
                        end = start + timedelta(minutes=30)
                        existing = await db.execute(
                            select(ClinicianSlot).where(
                                ClinicianSlot.doctor_id == doctor.id,
                                ClinicianSlot.start_time == start
                            )
                        )
                        if not existing.scalar_one_or_none():
                            db.add(ClinicianSlot(
                                doctor_id=doctor.id,
                                start_time=start,
                                end_time=end,
                                status=SlotStatus.AVAILABLE
                            ))
        await db.commit()
        print("Seed complete.")

if __name__ == "__main__":
    asyncio.run(seed())
```

---

## Part 5 — Redis Session State

Redis stores `SessionState` per patient session (TTL 2 hours). It also stores distributed locks and consultation timing data for the doctor portal.

### Key schema:

| Key | Value | TTL |
|---|---|---|
| `session:{session_id}` | JSON blob — full `SessionState` | 2 hours |
| `lock:slot:{slot_id}` | `"1"` — sentinel for PRE_LOCK | 600 seconds (10 min) |
| `lock:pre_lock_job` | `"1"` — distributed job lock | 70 seconds |
| `embed:{session_id}` | bytes — cached MiniLM embedding | 30 minutes |
| `consult_start:{slot_id}` | ISO timestamp | Until consultation ends |
| `consult_times:{doctor_id}` | Redis list of durations (minutes, last 10) | No expiry |

### Redis init (`/backend/app/core/redis.py`):

```python
from redis.asyncio import from_url

redis_client = None

async def init_redis():
    global redis_client
    redis_client = await from_url(settings.REDIS_URL, decode_responses=False)

async def close_redis():
    if redis_client:
        await redis_client.aclose()
```

### SessionState serialization:

```python
import json

async def save_session(session_id: str, state: SessionState) -> None:
    key = f"session:{session_id}"
    await redis_client.setex(key, 7200, state.model_dump_json())

async def load_session(session_id: str) -> SessionState | None:
    key = f"session:{session_id}"
    data = await redis_client.get(key)
    if data is None:
        return None
    return SessionState.model_validate_json(data)
```

---

## Part 6 — Slot Locking (Database-Level Concurrency)

The slot lock endpoint must use `SELECT ... FOR UPDATE SKIP LOCKED` to handle simultaneous requests without race conditions. Two patients clicking "lock" at the same moment → exactly one 200, one 409.

```python
# In booking.py endpoint
async with db.begin():
    result = await db.execute(
        select(ClinicianSlot)
        .where(ClinicianSlot.id == slot_id)
        .with_for_update(skip_locked=True)
    )
    slot = result.scalar_one_or_none()

    if slot is None or slot.status != SlotStatus.AVAILABLE:
        raise HTTPException(status_code=409, detail="Slot not available")

    slot.status = SlotStatus.PRE_LOCK
    await redis_client.setex(f"lock:slot:{slot_id}", 600, "1")
    await db.commit()
```

### APScheduler — release expired PRE_LOCKs (`/backend/app/services/scheduler.py`):

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", seconds=60)
async def release_expired_pre_locks():
    # Distributed lock — only one server instance runs this at a time
    acquired = await redis_client.set("lock:pre_lock_job", "1", nx=True, ex=70)
    if not acquired:
        return

    async with AsyncSessionLocal() as db:
        pre_locked = (await db.execute(
            select(ClinicianSlot).where(ClinicianSlot.status == SlotStatus.PRE_LOCK)
        )).scalars().all()

        for slot in pre_locked:
            redis_key = f"lock:slot:{slot.id}"
            if not await redis_client.exists(redis_key):
                slot.release()  # state machine method

        await db.commit()
```

---

## Part 7 — Atomic Queue Position Update on Cancellation

Never use a loop to update queue positions. Use a single SQL UPDATE:

```python
# In booking.py DELETE /appointments/{id}
async with db.begin():
    # Get appointment + queue entry
    appt = await db.get(Appointment, appt_id)
    queue_entry = (await db.execute(
        select(QueueEntry).where(QueueEntry.appointment_id == appt_id)
    )).scalar_one()

    cancelled_pos = queue_entry.position
    doctor_id = queue_entry.doctor_id
    appt_date = queue_entry.appointment_date

    # Atomic position shift — single UPDATE, not a loop
    await db.execute(
        update(QueueEntry)
        .where(
            QueueEntry.doctor_id == doctor_id,
            QueueEntry.position > cancelled_pos,
            QueueEntry.appointment_date == appt_date
        )
        .values(position=QueueEntry.position - 1)
    )

    # Release the slot
    slot = await db.get(ClinicianSlot, appt.slot_id)
    slot.status = SlotStatus.CANCELLED

    await db.delete(queue_entry)
    await db.commit()
```

---

## Part 8 — FAISS Index Construction

FAISS is not a database — it is built at startup from corpus files committed to the repo. There is **no `RagDocument` database model**. The two FAISS indexes live entirely in memory.

### Two indexes — never mix their sources:

| Index | Name | Source | Used when |
|---|---|---|---|
| A | `conversation_index.faiss` | D:/P: exchange pairs from transcripts + MedDialog | Every patient turn during intake |
| B | `knowledge_index.faiss` | MedQuAD QA pairs + MedlinePlus condition sections | Once only, at prognosis helper step |

### Embedding model (shared, loaded once):

```python
from sentence_transformers import SentenceTransformer

# Loaded ONCE at startup in lifespan.py — never per request
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Dimension: 384
```

### Index A construction script (`/backend/app/scripts/build_faiss_index.py`):

```python
import faiss
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunks = []

# ── Extract from conversations/ folder ──
for specialty_folder in Path('ml-training/conversations/').iterdir():
    specialty = specialty_folder.name
    for transcript_file in specialty_folder.glob('*.txt'):
        turns = parse_transcript(transcript_file)  # → [(speaker, text)]
        for i in range(len(turns) - 1):
            if turns[i][0] == 'D' and turns[i+1][0] == 'P':
                chunk_text = f"D: {turns[i][1]} P: {turns[i+1][1]}"
                chunks.append({
                    "chunk_id":   f"{specialty}_{transcript_file.stem}_turn_{i:03d}",
                    "specialty":  specialty,
                    "text":       chunk_text,
                    "source":     str(transcript_file),
                    "turn_index": i
                })

# ── Add MedDialog Q&A pairs (map department labels to 9 specialties) ──
# Similar extraction — see AI plan § 8.3 for specialty mapping details

# Startup crash guard
assert len(chunks) > 0, "FATAL: No chunks extracted. Check corpus files."

# ── Embed ──
texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks...")
embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2-normalize for cosine

# ── Build IVFFlat index (for ~22k chunks) ──
dim   = embeddings.shape[1]  # 384
nlist = 256
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings.astype(np.float32))
index.add(embeddings.astype(np.float32))

# ── Save ──
Path("indexes").mkdir(exist_ok=True)
faiss.write_index(index, 'indexes/conversation_index.faiss')
with open('indexes/conversation_chunks.json', 'w') as f:
    json.dump(chunks, f)
print(f"Index A: {len(chunks)} chunks saved.")
```

### Index B construction script (same pattern, different sources):

```python
# MedQuAD: one chunk per QA pair
#   {"text": f"Q: {question} A: {answer}", "specialty": inferred_from_source_category, ...}
#
# MedlinePlus: one chunk per section (symptoms / causes / when-to-see-a-doctor)
#   {"text": section_text, "specialty": inferred_specialty, ...}
#
# Save as indexes/knowledge_index.faiss and indexes/knowledge_chunks.json
```

**Expected chunk counts (Index A after full corpus):**

| Specialty | Approx. chunks |
|---|---|
| Respiratory | ~10,200 |
| Psychiatry | ~2,500 |
| Orthopedics | ~2,200 |
| Neurology | ~2,000 |
| Cardiology | ~1,500 |
| Pediatrics | ~1,200 |
| General Medicine | ~1,000 |
| Dermatology | ~800 |
| Gastroenterology | ~500 |
| **Total** | **~21,900** |

### Loading at server startup (`/backend/app/core/lifespan.py`):

```python
import faiss, json
from collections import defaultdict

# Called inside FastAPI lifespan context
conv_index  = faiss.read_index('indexes/conversation_index.faiss')
conv_chunks = json.load(open('indexes/conversation_chunks.json'))

know_index  = faiss.read_index('indexes/knowledge_index.faiss')
know_chunks = json.load(open('indexes/knowledge_chunks.json'))

# Pre-build specialty → chunk ID lookup for O(1) specialty filtering
specialty_to_conv_ids: dict[str, list[int]] = defaultdict(list)
for i, chunk in enumerate(conv_chunks):
    specialty_to_conv_ids[chunk["specialty"]].append(i)
```

### Transcript file format (required — any deviation = zero chunks = startup crash):

```
D: Doctor turn text here
P: Patient turn text here
D: Next doctor turn
P: Next patient turn
```

Files are `.txt`, one per conversation, organized in subfolders by specialty name under `ml-training/conversations/`.

---

## Part 9 — Database Acceptance Tests

These must all pass before moving to any other build phase:

- `alembic upgrade head` creates all 8 tables cleanly with zero errors.
- Seed script runs to completion on an empty database.
- Seed script runs a second time and produces zero duplicates (idempotency check).
- `GET /api/v1/health` returns `{"status": "ok", "db": "ok", "redis": "ok"}`.
- FAISS index builds successfully and `faiss_chunks` count is > 0 in health response.
- Server startup fails with a clear error message if `ml-training/conversations/` is empty.
- Two simultaneous `POST /slots/{id}/lock` requests return exactly one 200 and one 409.

---

*TriagePlus · IIT Dharwad Summer of Innovation · Hardly Human · Mentor: Prof. B. N. Bharath*
