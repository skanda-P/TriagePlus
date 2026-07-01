# TriagePlus — Frontend & Backend Build Prompt
### IIT Dharwad Summer of Innovation · Hardly Human
> This file covers the React+Vite frontend, the FastAPI backend structure, all REST and WebSocket endpoints, the FSM conversation flow, slot booking, payments, the doctor portal, security requirements, and deployment configuration.

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui |
| State management | Zustand + TanStack React Query |
| Payments (frontend) | `@stripe/react-stripe-js`, `@stripe/stripe-js` |
| Backend | FastAPI + Python 3.11+ |
| Async database | SQLAlchemy async + asyncpg |
| Migrations | Alembic |
| Cache | Redis (asyncio) |
| LLM | `google-generativeai` |
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| Classifier | `scikit-learn` |
| Scheduler | APScheduler |
| Payments (backend) | Stripe Python SDK |
| Email | SendGrid |
| Auth | JWT (doctors only) + UUID (patients) |
| Logging | structlog JSON |

---

## Part 1 — Directory Structure

Generate this exact structure before writing any code:

```
/triageplus
  /frontend
    /src
      /components
        /chat          ChatWindow.tsx  MessageBubble.tsx  NameEntry.tsx
        /booking       SlotPicker.tsx  PaymentForm.tsx  CountdownTimer.tsx
        /doctor        Dashboard.tsx   QueuePanel.tsx   SlotManager.tsx
        /shared        Button.tsx  Badge.tsx  Spinner.tsx  EmergencyBanner.tsx
      /hooks
        useWebSocket.ts    # WS manager: ping filter + exponential backoff reconnect
        useSession.ts      # generates/persists session_id in sessionStorage
        useDoctorAuth.ts   # JWT for doctor login only
      /stores
        chatStore.ts       # Zustand: messages, FSM state, session metadata
        bookingStore.ts    # Zustand: selected slot, payment state
      /lib
        api.ts             # React Query fetchers
        ws.ts              # WebSocket URL builder (uses VITE_WS_BASE_URL)
      /pages
        PatientChat.tsx    # Main patient flow
        DoctorDashboard.tsx
        DoctorLogin.tsx
    vercel.json
    vite.config.ts
    tailwind.config.ts

  /backend
    /app
      /api/v1
        chat.py       # Patient WebSocket + REST fallback
        booking.py    # Slot listing, PRE_LOCK, cancel
        payments.py   # Stripe webhook + payment intent
        doctors.py    # Doctor portal REST + WebSocket
        auth.py       # POST /auth/doctor/login only
        health.py     # GET /health
        classify.py   # POST /classify (debug endpoint)
        voice.py      # POST /voice/transcribe
      /core
        config.py     # Settings(BaseSettings)
        database.py   # async SQLAlchemy engine + session
        redis.py      # Redis client init/close
        security.py   # JWT for doctors; UUID validation for patients
        middleware.py # CORS, security headers, rate limiting
        logging.py    # structlog JSON
        lifespan.py   # startup: load MiniLM, build FAISS, seed DB
      /models         # (see Database Build Prompt)
      /schemas
        chat.py       # FSMState enum, SessionState Pydantic model
        intake.py     # IntakeSlots Pydantic model
        booking.py    # Slot response schemas
      /services
        fsm.py        # ChatFSM state transitions
        gemini.py     # Gemini 1.5 Flash client
        classifier.py # EmbeddingClassifier
        ner.py        # DictionaryNER (lay→clinical mappings)
        rag.py        # FAISSRetriever
        triage.py     # compute_triage_level()
        scheduler.py  # APScheduler: release_expired_pre_locks()
        payments.py   # Stripe helpers
        notifications.py  # SendGrid email
        queue.py      # Wait time calculation + queue broadcast
      /ml
        baseline.py   # TF-IDF + SVM baseline
        eval.py       # Confusion matrix, F1, latency
      /scripts
        seed_db.py            # Idempotent seed
        build_faiss_index.py  # Build FAISS from corpus
    main.py
    entrypoint.sh
    requirements.txt
    Dockerfile

  /ml-training
    /conversations   # COMMITTED TO REPO — FAISS source of truth
    /data
      real_queries/  # Hand-written test queries (eval only, never training)
    evaluate.py
    train_baseline.py
```

---

## Part 2 — Backend: Environment Variables & Config

### All required environment variables (`.env`):

```
DATABASE_URL=postgresql+asyncpg://...     # Supabase direct URL, port 5432
REDIS_URL=rediss://...                    # Upstash Redis
GEMINI_API_KEY=AIzaSy...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
JWT_SECRET=<64+ char random string>
ALLOWED_ORIGINS=["http://localhost:5173"]
SENDGRID_API_KEY=SG....
SENDGRID_FROM_EMAIL=noreply@example.com
```

### Settings with validation (`/backend/app/core/config.py`):

```python
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    GEMINI_API_KEY: str
    STRIPE_SECRET_KEY: str
    STRIPE_PUBLISHABLE_KEY: str
    STRIPE_WEBHOOK_SECRET: str
    JWT_SECRET: str
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]
    SENDGRID_API_KEY: str
    SENDGRID_FROM_EMAIL: str

    @validator("GEMINI_API_KEY")
    def api_key_must_not_be_placeholder(cls, v):
        if not v or "CHANGEME" in v:
            raise ValueError("GEMINI_API_KEY not configured")
        return v

settings = Settings()
```

### Backend requirements.txt:

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
pydantic-settings==2.3.1
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
alembic==1.13.1
redis[asyncio]==5.0.6
google-generativeai>=0.7.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
scikit-learn>=1.5.0
numpy>=1.26.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
slowapi==0.1.9
stripe==9.12.0
sendgrid==6.11.0
apscheduler==3.10.4
structlog==24.2.0
httpx==0.27.0
python-multipart==0.0.9
```

---

## Part 3 — Backend: FastAPI App & Lifespan

### `main.py`:

```python
from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.v1 import chat, booking, payments, doctors, auth, health, classify, voice
from app.core.middleware import setup_middleware

app = FastAPI(title="TriagePlus API", lifespan=lifespan)
setup_middleware(app)

app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router,   prefix="/api/v1/auth")
app.include_router(chat.router,   prefix="/api/v1")
app.include_router(booking.router,prefix="/api/v1")
app.include_router(payments.router,prefix="/api/v1")
app.include_router(doctors.router, prefix="/api/v1/doctors")
app.include_router(classify.router,prefix="/api/v1")
app.include_router(voice.router,   prefix="/api/v1/voice")
```

### `/backend/app/core/lifespan.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    await init_redis()

    # Load MiniLM embedding model (once — shared across classifier and FAISS)
    global embedding_model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load classifier
    with open('models/specialty_classifier.pkl', 'rb') as f:
        saved = pickle.load(f)
    global clf, le
    clf = saved['clf']
    le  = saved['label_encoder']

    # Build / load FAISS indexes
    if not Path('indexes/conversation_index.faiss').exists():
        subprocess.run(['python', '-m', 'app.scripts.build_faiss_index'], check=True)
    global conv_index, conv_chunks, know_index, know_chunks, specialty_to_conv_ids
    conv_index  = faiss.read_index('indexes/conversation_index.faiss')
    conv_chunks = json.load(open('indexes/conversation_chunks.json'))
    know_index  = faiss.read_index('indexes/knowledge_index.faiss')
    know_chunks = json.load(open('indexes/knowledge_chunks.json'))
    specialty_to_conv_ids = build_specialty_lookup(conv_chunks)

    # Start APScheduler
    scheduler.start()

    yield  # Application running

    # ── Shutdown ──
    scheduler.shutdown()
    await close_redis()
```

---

## Part 4 — Backend: Conversation FSM

### States:

```
NAME_ENTRY → COLLECTING_PROFILE → COLLECTING_SYMPTOMS → RECOMMENDING → BOOKING → PAYMENT → CONFIRMED
```

### Rules per state:

**NAME_ENTRY:**
- Opening message: `"Welcome to TriagePlus! 👋 What's your name?"`
- Hardcoded template. Zero LLM calls.
- On submission: create `PatientSession` row in DB, store `patient_name` in `SessionState`.

**COLLECTING_PROFILE (all hardcoded templates — Zero LLM calls):**
- Ask age → validate integer
- Ask gender → accept "Male" / "Female" / "Other" only
- Ask phone number → validate 10-digit Indian mobile
- Ask preferred time: "Morning" / "Afternoon" / "Evening"
- Re-ask with corrective guidance on invalid input

**COLLECTING_SYMPTOMS (Gemini is in charge):**
- Call Gemini with slot extraction + follow-up generation (see AI Components prompt)
- RAG retrieval runs before each Gemini call
- Classifier runs in parallel with RAG (shared embedding vector)
- Ask ONE follow-up question per turn
- When chief_complaint + severity + onset are all filled: transition to RECOMMENDING

**RECOMMENDING:**
- Present top-1 specialty with confidence label
- Show triage level (color-coded: GREEN/YELLOW/ORANGE/RED)
- Show prognosis helper output + hardcoded disclaimer
- Show top-3 doctor recommendations
- Offer slot picker

**BOOKING → PAYMENT → CONFIRMED:** handled by frontend SlotPicker + Stripe

### Emergency detector runs FIRST on every message (see AI Components prompt):

```python
# In /backend/app/api/v1/chat.py WebSocket handler
async for message in websocket.iter_json():
    if check_emergency(message["content"]):
        await db.add(AuditLog(
            session_id=session_id,
            event_type="emergency_detected",
            payload=message["content"][:500]
        ))
        await websocket.send_json({
            "type": "emergency",
            "content": "⚠️ This sounds like a medical emergency. Please call 112 immediately or go to your nearest emergency room. Do not wait for an appointment."
        })
        await websocket.close()
        return
    # proceed to FSM
```

### `SessionState` Pydantic model (`/backend/app/schemas/chat.py`):

```python
from pydantic import BaseModel
from enum import Enum

class FSMState(str, Enum):
    NAME_ENTRY          = "NAME_ENTRY"
    COLLECTING_PROFILE  = "COLLECTING_PROFILE"
    COLLECTING_SYMPTOMS = "COLLECTING_SYMPTOMS"
    RECOMMENDING        = "RECOMMENDING"
    BOOKING             = "BOOKING"
    PAYMENT             = "PAYMENT"
    CONFIRMED           = "CONFIRMED"

class SessionState(BaseModel):
    session_id: str
    patient_name: str | None = None
    fsm_state: FSMState = FSMState.NAME_ENTRY
    age: int | None = None
    gender: str | None = None
    contact: str | None = None
    preferred_timeframe: str | None = None
    # AI intake slots
    chief_complaint: str | None = None
    severity: str | None = None
    severity_numeric: int | None = None
    onset: str | None = None
    onset_days: int | None = None
    associated_symptoms: list[str] = []
    medical_history_flags: list[str] = []
    # Classifier output
    provisional_specialty: str | None = None
    provisional_confidence: float | None = None
    confidence_label: str | None = None
    # Computed
    triage_level: int | None = None
    locked_slot_id: int | None = None
    classifier_result: list | None = None
    turn_count: int = 0
```

---

## Part 5 — Backend: WebSocket Architecture

### Patient WebSocket: `GET /ws/chat/{session_id}`

```python
@router.websocket("/ws/chat/{session_id}")
async def patient_ws(websocket: WebSocket, session_id: str, db: AsyncSession = Depends(get_db)):
    # 1. Validate session_id is a valid UUID
    try:
        uuid.UUID(session_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # 2. Load or create SessionState from Redis
    state = await load_session(session_id) or SessionState(session_id=session_id)

    # 3. Send opening message (NAME_ENTRY state)
    if state.fsm_state == FSMState.NAME_ENTRY:
        await websocket.send_json({
            "type": "message",
            "content": "Welcome to TriagePlus! 👋 What's your name?"
        })

    # 4. Keepalive ping loop (Render drops idle at 30s)
    async def ping_loop():
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    ping_task = asyncio.create_task(ping_loop())

    try:
        async for data in websocket.iter_json():
            if data.get("type") == "pong":
                continue

            # Emergency check FIRST
            content = data.get("content", "")
            if check_emergency(content):
                # ... (emergency handling as above)
                break

            # FSM transition
            response = await process_fsm(state, content, db)
            state.turn_count += 1

            # Save updated state to Redis
            await save_session(session_id, state)

            # Send response
            await websocket.send_json({"type": "message", "content": response})

    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
```

**Session resumption:** if the patient reloads within the same browser tab, `sessionStorage` has the same `session_id`, and `load_session()` returns the existing `SessionState` from Redis. The conversation resumes from where it left off.

### Doctor WebSocket: `GET /ws/doctor/queue`

```python
@router.websocket("/ws/doctor/queue")
async def doctor_ws(websocket: WebSocket):
    await websocket.accept()

    # Require auth message within 5 seconds
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except asyncio.TimeoutError:
        await websocket.close(code=1008)
        return

    doctor = verify_doctor_token(auth_msg.get("token"))
    if not doctor:
        await websocket.close(code=1008)
        return

    # Register for queue broadcasts
    connected_doctors[doctor.id] = websocket

    try:
        async for _ in websocket.iter_json():
            pass  # No messages expected from doctor on this channel
    except WebSocketDisconnect:
        pass
    finally:
        connected_doctors.pop(doctor.id, None)
```

**Queue broadcast (called from Stripe webhook handler):**
```python
async def broadcast_queue_update(doctor_id: int, payload: dict):
    ws = connected_doctors.get(doctor_id)
    if ws:
        try:
            await ws.send_json(payload)
        except Exception:
            connected_doctors.pop(doctor_id, None)
```

---

## Part 6 — Backend: All REST Endpoints

### `GET /api/v1/health`

```python
@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    redis_status = "ok" if await redis_client.ping() else "error"
    faiss_chunks = len(conv_chunks) if conv_chunks else 0

    return {
        "status": "ok" if db_status == "ok" and redis_status == "ok" else "degraded",
        "db": db_status,
        "redis": redis_status,
        "faiss_chunks": faiss_chunks
    }
```

### `POST /api/v1/auth/doctor/login`

```python
@router.post("/auth/doctor/login")
async def doctor_login(email: str, password: str, db: AsyncSession = Depends(get_db)):
    doctor = await get_doctor_by_email(db, email)
    if not doctor or not verify_password(password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_jwt({"sub": str(doctor.id), "role": "doctor"})
    return {"access_token": token, "token_type": "bearer"}
```

### `GET /api/v1/slots` — list available slots filtered by specialty

```python
@router.get("/slots")
async def list_slots(
    specialty: str,
    triage_level: int = 4,
    db: AsyncSession = Depends(get_db)
):
    # Filter by specialty, AVAILABLE status, triage-appropriate time window
    ...
```

### `POST /api/v1/slots/{slot_id}/lock`

```python
@router.post("/slots/{slot_id}/lock")
@limiter.limit("20/minute")
async def lock_slot(slot_id: int, session_id: str, db: AsyncSession = Depends(get_db)):
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

    # Update SessionState with locked_slot_id
    state = await load_session(session_id)
    if state:
        state.locked_slot_id = slot_id
        await save_session(session_id, state)

    return {"slot_id": slot_id, "status": "PRE_LOCK", "expires_in_seconds": 600}
```

### `POST /api/v1/payments/create-intent`

```python
@router.post("/payments/create-intent")
async def create_payment_intent(slot_id: int, session_id: str):
    intent = stripe.PaymentIntent.create(
        amount=50000,  # ₹500 in paisa — adjust per hospital
        currency="inr",
        metadata={"slot_id": str(slot_id), "session_id": session_id}
    )
    return {"client_secret": intent.client_secret}
```

### `POST /api/v1/webhooks/stripe`

```python
@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "payment_intent.succeeded":
        pi = event["data"]["object"]
        pi_id     = pi["id"]
        slot_id   = int(pi["metadata"]["slot_id"])
        session_id = pi["metadata"]["session_id"]

        # Idempotency check
        existing = await db.execute(
            select(Payment).where(Payment.stripe_pi_id == pi_id, Payment.status == "completed")
        )
        if existing.scalar_one_or_none():
            return {"status": "already_processed"}

        # Single transaction: confirm slot + create appointment + enqueue + payment record
        async with db.begin():
            slot = (await db.execute(
                select(ClinicianSlot).where(ClinicianSlot.id == slot_id).with_for_update()
            )).scalar_one()
            slot.confirm()  # state machine method

            state = await load_session(session_id)

            # Generate doctor brief
            prognosis = state.prognosis_text  # stored earlier
            brief = await generate_doctor_brief(state, prognosis)

            appt = Appointment(
                session_id=session_id,
                slot_id=slot_id,
                doctor_id=slot.doctor_id,
                specialty=state.provisional_specialty,
                ai_brief=brief
            )
            db.add(appt)
            await db.flush()

            position = await get_next_queue_position(db, slot.doctor_id)
            db.add(QueueEntry(
                appointment_id=appt.id,
                doctor_id=slot.doctor_id,
                position=position,
                appointment_date=slot.start_time.date().isoformat()
            ))
            db.add(Payment(
                appointment_id=appt.id,
                stripe_pi_id=pi_id,
                amount=pi["amount"],
                status="completed"
            ))
            await db.commit()

        # Push to doctor's WebSocket queue
        await broadcast_queue_update(slot.doctor_id, {
            "type": "new_patient",
            "patient_name": state.patient_name,
            "specialty": state.provisional_specialty,
            "ai_brief": brief,  # only to doctor WebSocket — never patient
            "position": position,
        })

        # Send confirmation email
        session_obj = await db.get(PatientSession, session_id)
        await send_confirmation_email(session_obj, appt, slot)

    elif event["type"] == "payment_intent.payment_failed":
        # Release PRE_LOCK
        pi = event["data"]["object"]
        slot_id = int(pi["metadata"]["slot_id"])
        async with db.begin():
            slot = await db.get(ClinicianSlot, slot_id)
            slot.release()
            await db.commit()

    return {"received": True}
```

### `DELETE /api/v1/appointments/{id}`

Uses atomic queue position update — single SQL UPDATE, not a loop. See Database Build Prompt Part 7.

### `GET /api/v1/doctors/me/queue`

```python
@router.get("/me/queue")
async def get_queue(
    doctor: Doctor = Depends(require_role("doctor")),
    db: AsyncSession = Depends(get_db)
):
    today = date.today().isoformat()
    entries = (await db.execute(
        select(QueueEntry, Appointment, PatientSession)
        .join(Appointment, QueueEntry.appointment_id == Appointment.id)
        .join(PatientSession, Appointment.session_id == PatientSession.session_id)
        .where(QueueEntry.doctor_id == doctor.id, QueueEntry.appointment_date == today)
        .order_by(QueueEntry.position)
    )).all()

    return [
        {
            "position": e.position,
            "patient_name": ps.patient_name,
            "specialty": a.specialty,
            "ai_brief": e_appt.ai_brief,  # returned here to doctor only
            "wait_time_minutes": compute_wait_time(doctor.id, e.position)
        }
        for e, a, ps in entries
    ]
```

### `POST /api/v1/classify` (debug endpoint)

```python
@router.post("/classify")
async def debug_classify(text: str):
    result = classify_specialty_raw(text)
    return result
```

### `POST /api/v1/voice/transcribe`

```python
@router.post("/voice/transcribe")
async def transcribe_voice(audio: UploadFile, language: str = "hi"):
    audio_bytes = await audio.read()
    transcript = await whisper_transcribe(audio_bytes, language)
    return {"transcript": transcript}
```

---

## Part 7 — Backend: Security Requirements

### Patient WebSocket UUID validation:

```python
try:
    uuid.UUID(session_id)
except ValueError:
    await websocket.close(code=1008)
    return
```

### Doctor JWT:

```python
# /backend/app/core/security.py
from jose import jwt, JWTError
from datetime import datetime, timedelta

def create_jwt(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

def verify_doctor_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        if payload.get("role") != "doctor":
            return None
        return payload
    except JWTError:
        return None

def require_role(role: str):
    def dependency(token: str = Depends(oauth2_scheme)):
        payload = verify_doctor_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return payload
    return dependency
```

JWT check: uses `token.role` only — NO database lookup per request.

### Rate limiting (slowapi):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/slots/{slot_id}/lock")
@limiter.limit("20/minute")
async def lock_slot(...): ...

# WebSocket upgrade rate limiting: 10 new sessions per IP per hour
```

### CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# NEVER use allow_origins=["*"]
```

### Stripe webhook verification — always verify signature before processing:

```python
event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
# Raises SignatureVerificationError on tampered webhooks → return 400
```

---

## Part 8 — Frontend Architecture

### Frontend environment variables (`.env` in `/frontend`):

```
VITE_API_BASE_URL=https://your-backend.onrender.com
VITE_WS_BASE_URL=wss://your-backend.onrender.com
VITE_STRIPE_PUBLISHABLE_KEY=pk_test_...
```

### Session management (`/frontend/src/hooks/useSession.ts`):

```typescript
// Generate UUID on first load, persist in sessionStorage
// Reuse across reloads in same tab
// Different tab = new session_id = fresh conversation
export function useSession(): string {
    const [sessionId] = useState<string>(() => {
        const existing = sessionStorage.getItem("triageplus_session_id");
        if (existing) return existing;
        const newId = crypto.randomUUID();
        sessionStorage.setItem("triageplus_session_id", newId);
        return newId;
    });
    return sessionId;
}
```

### WebSocket hook (`/frontend/src/hooks/useWebSocket.ts`):

```typescript
// 1. Exponential backoff reconnect: delay = min(1000 * 2^attempt, 30000)
// 2. Filter ping messages: if (data.type === "ping") return
// 3. Show "reconnecting..." indicator during backoff
// 4. On reconnect: backend reloads FSM from Redis using session_id

export function useWebSocket(sessionId: string) {
    const [status, setStatus] = useState<"connecting"|"open"|"reconnecting"|"closed">("connecting");
    const wsRef = useRef<WebSocket | null>(null);
    const attemptRef = useRef(0);

    const connect = useCallback(() => {
        const wsUrl = `${import.meta.env.VITE_WS_BASE_URL}/api/v1/ws/chat/${sessionId}`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => { setStatus("open"); attemptRef.current = 0; };
        ws.onmessage = (evt) => {
            const data = JSON.parse(evt.data);
            if (data.type === "ping") return; // filter keepalive pings
            // dispatch to chatStore
        };
        ws.onclose = () => {
            setStatus("reconnecting");
            const delay = Math.min(1000 * Math.pow(2, attemptRef.current++), 30000);
            setTimeout(connect, delay);
        };
    }, [sessionId]);

    useEffect(() => { connect(); }, [connect]);

    return { status, send: (msg: object) => wsRef.current?.send(JSON.stringify(msg)) };
}
```

### Zustand store (`/frontend/src/stores/chatStore.ts`):

```typescript
import { create } from 'zustand';

interface Message { role: "patient"|"assistant"|"emergency"; content: string; }
interface FSMState { current: string; }

interface ChatStore {
    messages: Message[];
    fsmState: FSMState;
    sessionMeta: { specialty?: string; triageLevel?: number; confidence?: string };
    addMessage: (msg: Message) => void;
    setFsmState: (state: FSMState) => void;
    setSessionMeta: (meta: Partial<ChatStore["sessionMeta"]>) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
    messages: [],
    fsmState: { current: "NAME_ENTRY" },
    sessionMeta: {},
    addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
    setFsmState: (state) => set({ fsmState: state }),
    setSessionMeta: (meta) => set((s) => ({ sessionMeta: { ...s.sessionMeta, ...meta } })),
}));
```

### Chat UI requirements:

- Name entry field before chat opens (not a login form)
- Patient messages: right-aligned; assistant messages: left-aligned
- Emergency messages: red background, `"📞 Call 112"` as a `tel:112` link, never dismissible
- Slot picker shown as an inline card after recommendation — not a page navigation
- 10-minute countdown timer during PRE_LOCK state (rendered inside the chat)
- Stripe Elements card form rendered inline in chat (not a separate page)
- Hardcoded disclaimer banner always visible on result screen: `"⚠ This information is general in nature and does not constitute a medical diagnosis. Always consult a qualified medical professional for advice specific to your situation."`

### Doctor dashboard requirements:

- Route guard: redirect to `/doctor/login` if no valid JWT in sessionStorage
- Queue panel: WebSocket-subscribed, shows position, patient_name, ai_brief, wait time
- "Start Consultation" and "End Consultation" buttons per patient
- Slot manager: add / remove available slots
- All `ai_brief` content visible only on the doctor dashboard — never in any patient-facing view

### Triage level display (color coding):

| Level | Label | Color |
|---|---|---|
| 1 | Emergency | RED — also triggers emergency flow |
| 2 | Urgent | ORANGE |
| 3 | Soon | YELLOW |
| 4 | Routine | GREEN |

---

## Part 9 — Wait Time Calculation

```python
# /backend/app/services/queue.py
from statistics import mean

async def compute_wait_time(doctor_id: int, patient_position: int) -> int:
    """Returns estimated wait in minutes."""
    # Rolling average of last 10 consultation durations
    raw = await redis_client.lrange(f"consult_times:{doctor_id}", 0, -1)
    if raw:
        t_avg = mean(float(v) for v in raw)
    else:
        t_avg = 15.0  # default assumption

    # Time already elapsed in current consultation
    current_start = await redis_client.get(f"consult_start:current:{doctor_id}")
    if current_start:
        elapsed = (datetime.utcnow() - datetime.fromisoformat(current_start.decode())).seconds // 60
        remaining_current = max(0, t_avg - elapsed)
    else:
        remaining_current = 0.0

    n_ahead = max(0, patient_position - 1)
    return int(remaining_current + (n_ahead * t_avg))
```

**Doctor consultation endpoints:**

```python
@router.put("/slots/{id}/start")
async def start_consultation(id: int, doctor = Depends(require_role("doctor"))):
    await redis_client.set(f"consult_start:current:{doctor['sub']}", datetime.utcnow().isoformat())
    return {"status": "started"}

@router.put("/slots/{id}/end")
async def end_consultation(id: int, doctor = Depends(require_role("doctor"))):
    start_raw = await redis_client.get(f"consult_start:current:{doctor['sub']}")
    if start_raw:
        duration = (datetime.utcnow() - datetime.fromisoformat(start_raw.decode())).seconds // 60
        await redis_client.rpush(f"consult_times:{doctor['sub']}", str(duration))
        await redis_client.ltrim(f"consult_times:{doctor['sub']}", -10, -1)  # keep last 10
        await redis_client.delete(f"consult_start:current:{doctor['sub']}")
    return {"status": "ended"}
```

---

## Part 10 — Deployment Configuration

### `backend/entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Wait for Supabase to accept connections
echo "Waiting for database..."
for i in $(seq 1 30); do
    python -c "
import asyncio, asyncpg, os, sys
async def check():
    try:
        conn = await asyncpg.connect(os.environ['DATABASE_URL'].replace('+asyncpg', ''))
        await conn.close()
    except Exception:
        sys.exit(1)
asyncio.run(check())
" && break
    echo "Attempt $i failed, retrying in 2s..."
    sleep 2
done

alembic upgrade head
python -m app.scripts.seed_db
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### `frontend/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

### `vite.config.ts`:

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/api': 'http://localhost:8000',
            '/ws':  { target: 'ws://localhost:8000', ws: true }
        }
    }
});
```

### Render keep-alive:

Use UptimeRobot (free tier) to ping `GET /api/v1/health` every 5 minutes. Render free tier sleeps after 15 minutes of inactivity.

---

## Part 11 — Acceptance Tests (Per Checkpoint)

### Checkpoint 0 — Skeleton:
- `GET /api/v1/health` → `{"status": "ok", "db": "ok", "redis": "ok"}`
- Missing `GEMINI_API_KEY` → startup crash with clear error
- React dev server shows the health response

### Checkpoint 1 — Data Layer:
- `alembic upgrade head` creates all tables cleanly
- Seed script is idempotent (run twice = zero duplicates)
- FAISS index builds; `faiss_chunks` > 0 in health response
- Startup fails clearly if `ml-training/conversations/` is empty

### Checkpoint 2 — Doctor Auth:
- `POST /auth/doctor/login` with correct credentials → JWT with `"role": "doctor"`
- Protected routes return 401 without token
- Tampered JWT returns 401

### Checkpoint 3 — Chat + Emergency:
- Enter name → receive age question
- Type "I'm having a heart attack" → emergency message in <200ms, WebSocket closes
- Emergency event appears in AuditLog
- Page reload in same tab → same session resumes

### Checkpoint 4 — Classifier:
- "sharp chest pain worse when lying down" → Cardiology in top-3, confidence > 0.70
- "tummy ache" → normalized to "abdominal pain" by NER → Gastroenterology in top-3
- p95 latency < 200ms over 50 calls on `POST /classify`

### Checkpoint 5 — RAG + Gemini:
- Full intake: name → age/gender/contact/timeframe (hardcoded) → symptom (Gemini)
- Gemini asks 1–2 follow-up questions before transitioning to recommendation
- Prompt injection `"ignore previous instructions"` → input guardrail rejects
- Gemini output `"you have pericarditis"` → output guardrail rewrites

### Checkpoint 6 — Scheduling:
- Two simultaneous lock requests → exactly one 200, one 409
- Lock then wait 11 minutes → slot released to AVAILABLE
- Cancellation updates queue positions atomically

### Checkpoint 7 — Payments:
- Test card `4242 4242 4242 4242` → confirmed appointment + queue entry + email
- Same webhook event twice → second is a no-op
- Declined card `4000 0000 0000 0002` → PRE_LOCK released

### Checkpoint 8 — Doctor Portal:
- Doctor sees today's queue with AI briefs
- New patient books → doctor's queue updates without page refresh
- Start consultation → wait times update

### Checkpoint 9 — Voice:
- "I have chest pain" spoken → transcript in chat → FSM processes correctly
- Voice mode toggle does not reset FSM state

### Checkpoint 10 — Evaluation:
- Confusion matrix PNG generated with real test queries
- Security checklist fully passed
- README has one-command local setup and deploy instructions

---

## Implementation Rules

1. **Order is fixed** — do not start a checkpoint until the previous one passes all acceptance tests.
2. **No `RagDocument` database model** — RAG lives entirely in FAISS in memory.
3. **No `Patient` model with password** — patients are sessions, not accounts.
4. **`slot.confirm()` and `slot.release()` are the ONLY paths to change slot status** — no direct `slot.status = ...` assignment anywhere else.
5. **Gemini is called only for symptom extraction and follow-up questions** — all profile questions use hardcoded templates.
6. **FAISS index is rebuilt at every startup** — Render free tier has no persistent disk. Corpus files are committed to the repo.
7. **All Gemini JSON responses are validated against `GeminiIntakeResponse` Pydantic schema** before the FSM accepts them.
8. **MiniLM inference always runs in `run_in_executor`** — never block the async event loop.
9. **`ai_brief` is never exposed through patient-facing API routes** — validate this in tests.
10. **The disclaimer text is hardcoded in the frontend React component** — it is never generated by the model.

---

*TriagePlus · IIT Dharwad Summer of Innovation · Hardly Human · Mentor: Prof. B. N. Bharath*
