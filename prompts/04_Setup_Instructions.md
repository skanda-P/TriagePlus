# TriagePlus — Setup Instructions for You
### IIT Dharwad Summer of Innovation · Hardly Human
> Follow these steps before any code is written. Every item here is a prerequisite. The sections are ordered: download datasets first, then create accounts and API keys, then set up local environment.

---

## Section 1: Datasets to Download

You need 5 datasets. Download them all before starting the data pipeline.

### Dataset 1 — MedDialog (English)

**What it is:** ~250,000 real doctor-patient dialogues from HealthCareMagic and iCliniq, tagged by department. Used for both the FAISS conversation index and classifier training samples.

**Where to get it:** Hugging Face Datasets

**URL:** https://huggingface.co/datasets/medical_dialog

**How to download:**
```bash
pip install datasets
```
```python
from datasets import load_dataset
ds = load_dataset("medical_dialog", "processed.en")
# Save the splits you need
ds["train"].to_json("ml-training/data/meddialog_en_train.jsonl")
```

**What to keep:** the `description` field (patient's message) and the `utterances` field (the full conversation). You will extract patient's first 1–3 turns and map department labels to your 9 specialties.

---

### Dataset 2 — mtsamples

**What it is:** ~4,999 medical transcription samples organized by clinical specialty. Each sample has a chief complaint field and a full transcription. Used for classifier training.

**Where to get it:** Kaggle

**URL:** https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

**How to download:**
1. Log into Kaggle → go to the URL above → click **Download**.
2. Or via CLI:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/ first (see Kaggle API key section below)
kaggle datasets download -d tboyle10/medicaltranscriptions
unzip medicaltranscriptions.zip -d ml-training/data/mtsamples/
```

**What to keep:** the `medical_specialty` column (maps to your specialties) and the `transcription` column (first 2–3 sentences as training input).

---

### Dataset 3 — Symptom2Disease

**What it is:** labeled dataset of symptom descriptions mapped to disease names. Used for classifier training after applying the disease→specialty JSON mapping in the AI Components prompt.

**Where to get it:** Kaggle

**URL:** https://www.kaggle.com/datasets/niyarrbarman/symptom2disease

**How to download:**
```bash
kaggle datasets download -d niyarrbarman/symptom2disease
unzip symptom2disease.zip -d ml-training/data/symptom2disease/
```

**What to keep:** the `label` column (disease name → map to specialty) and the `text` column (symptom description → classifier input).

---

### Dataset 4 — MedQuAD (for FAISS Index B)

**What it is:** ~47,000 QA pairs from NIH websites covering symptoms, causes, and treatments organized by condition. Used for the knowledge index (FAISS Index B), which grounds the prognosis helper.

**Where to get it:** Hugging Face Datasets

**URL:** https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

**How to download:**
```python
from datasets import load_dataset
ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
ds["train"].to_json("ml-training/data/medquad.jsonl")
```

**What to keep:** the `Question` and `Answer` fields. Each QA pair becomes one chunk in Index B.

---

### Dataset 5 — MedlinePlus XML Dump (for FAISS Index B)

**What it is:** ~1,000 plain-English condition summaries written for patients, covering symptoms, causes, and when-to-see-a-doctor sections. Used for FAISS Index B alongside MedQuAD.

**Where to get it:** Direct download from National Library of Medicine (no account needed)

**URL:** https://medlineplus.gov/xml.html

**How to download:**
1. Go to the URL above.
2. Download `mplus_topics_2015-01-01.xml` (or the most recent date available). The file is around 20–30 MB.
3. Place it at `ml-training/data/medlineplus.xml`.

**What to keep:** the `<summary>` section, `<also-called>` section, and `<see-reference>` section per health topic. Parse with Python's `xml.etree.ElementTree`.

---

### Optional Dataset — CounselChat (Psychiatry only)

**What it is:** real therapy conversation logs. Used only for Psychiatry classifier samples. Keep only the patient's opening message that describes a clinical symptom. Discard life-situation text.

**Where to get it:** Hugging Face Datasets

**URL:** https://huggingface.co/datasets/nbertagnolli/counsel-chat

**How to download:**
```python
from datasets import load_dataset
ds = load_dataset("nbertagnolli/counsel-chat")
ds["train"].to_json("ml-training/data/counselchat.jsonl")
```

---

## Section 2: Your `conversations.zip` Corpus

You should already have a `conversations.zip` file with real doctor-patient transcripts. If you have it, place it at:

```
triageplus/ml-training/conversations/
```

Organize the `.txt` files into subfolders named after each specialty:

```
ml-training/conversations/
  Cardiology/
    transcript_001.txt
    transcript_002.txt
  Respiratory/
    CAR0004.txt
    ...
  Orthopedics/
  Neurology/
  Gastroenterology/
  Dermatology/
  Pediatrics/
  Psychiatry/
  General_Medicine/
```

Each `.txt` file must be in this exact format (any deviation = zero chunks from that file):
```
D: Doctor turn text here
P: Patient turn text here
D: Next doctor turn
P: Next patient turn
```

---

## Section 3: API Keys to Generate

### 3.1 Google Gemini API Key

**Why:** powers the conversational intake (slot extraction + follow-up questions), prognosis helper, and doctor brief generator.

**Steps:**
1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with a Google account.
3. Click **Create API key**.
4. Copy the key (starts with `AIzaSy...`).
5. Add to your `.env`: `GEMINI_API_KEY=AIzaSy...`

**Free tier:** 15 requests/minute on Gemini 1.5 Flash — sufficient for development and demo. No billing required initially.

---

### 3.2 Supabase Project (Database)

**Why:** provides PostgreSQL for all persistent data (doctors, slots, appointments, payments, etc.)

**Steps:**
1. Go to: https://supabase.com
2. Create a free account → click **New Project**.
3. Choose a name (e.g. `triageplus`), set a database password, choose region closest to you (Singapore or Mumbai).
4. Wait for the project to provision (~2 minutes).
5. Go to **Settings → Database → Connection string**.
6. Select the **URI** tab, choose the **Direct connection** (NOT Transaction pooler).
7. Copy the URL — it looks like: `postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres`
8. Change `postgresql://` to `postgresql+asyncpg://` for SQLAlchemy.
9. Add to `.env`: `DATABASE_URL=postgresql+asyncpg://postgres:[PASSWORD]@db.[ref].supabase.co:5432/postgres`

**Important:** Always port 5432 (direct). Never port 6543 (pooler).

---

### 3.3 Upstash Redis

**Why:** stores `SessionState` per patient (with 2-hour TTL), slot pre-lock sentinels, and doctor consultation timing data.

**Steps:**
1. Go to: https://upstash.com
2. Create a free account → click **Create Database**.
3. Choose **Redis**, select region (Singapore), keep **TLS enabled**.
4. After creation, go to the database page → copy the **UPSTASH_REDIS_REST_URL** — but you need the standard Redis URL, not the REST URL.
5. Click **Details** → copy the `rediss://` URL (starts with `rediss://default:...`).
6. Add to `.env`: `REDIS_URL=rediss://default:[TOKEN]@[HOST]:[PORT]`

**Free tier:** 10,000 commands/day — more than enough for development.

---

### 3.4 Stripe (Payments)

**Why:** processes appointment booking payments. You will use test mode throughout development — no real money is charged.

**Steps:**
1. Go to: https://dashboard.stripe.com/register
2. Create an account. When prompted, you can skip business details for now.
3. Make sure the **Test mode** toggle is ON (top right of the dashboard).
4. Go to **Developers → API keys**.
5. Copy the **Publishable key** (starts with `pk_test_...`) and the **Secret key** (starts with `sk_test_...`).
6. For the webhook secret: go to **Developers → Webhooks → Add endpoint**.
   - Endpoint URL: `https://your-backend.onrender.com/api/v1/webhooks/stripe`
   - For local testing: use the Stripe CLI (see below)
   - Events to listen for: `payment_intent.succeeded`, `payment_intent.payment_failed`
   - Copy the **Signing secret** (starts with `whsec_...`).
7. Add to `.env`:
   ```
   STRIPE_SECRET_KEY=sk_test_...
   STRIPE_PUBLISHABLE_KEY=pk_test_...
   STRIPE_WEBHOOK_SECRET=whsec_...
   ```

**Stripe CLI (for local webhook testing):**
```bash
# Install: https://stripe.com/docs/stripe-cli
stripe login
stripe listen --forward-to localhost:8000/api/v1/webhooks/stripe
# This prints a temporary webhook signing secret — use it as STRIPE_WEBHOOK_SECRET locally
```

**Test cards (use these when testing payments):**
- Success: `4242 4242 4242 4242` · Any future date · Any CVC
- Decline: `4000 0000 0000 0002` · Any future date · Any CVC

---

### 3.5 SendGrid (Email)

**Why:** sends appointment confirmation emails and cancellation notifications.

**Steps:**
1. Go to: https://signup.sendgrid.com
2. Create a free account (free tier: 100 emails/day, no credit card needed).
3. Go to **Settings → API Keys → Create API Key**.
4. Choose **Restricted Access**, enable **Mail Send** permission only.
5. Copy the key (starts with `SG....`).
6. Go to **Settings → Sender Authentication → Single Sender Verification**.
7. Add an email address you own (even your college Gmail works). Verify it.
8. Add to `.env`:
   ```
   SENDGRID_API_KEY=SG....
   SENDGRID_FROM_EMAIL=your-verified-email@gmail.com
   ```

---

### 3.6 JWT Secret

**Why:** signs doctor login tokens.

**How to generate (run once, save the output):**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
This gives you a 64-character hex string. Add to `.env`: `JWT_SECRET=<the 64-char string>`

---

### 3.7 Kaggle API Key (only needed to download Kaggle datasets)

**Steps:**
1. Go to: https://www.kaggle.com → click your profile picture → **Settings**.
2. Scroll to **API** section → click **Create New Token**.
3. This downloads a `kaggle.json` file.
4. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<User>\.kaggle\kaggle.json` (Windows).
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

### 3.8 OpenAI Whisper API Key (for regional language voice input)

**Why:** transcribes audio in Hindi, Kannada, Tamil, Telugu for the voice input feature. Only needed for Phase 3 (voice integration).

**Steps:**
1. Go to: https://platform.openai.com/api-keys
2. Sign in → click **Create new secret key**.
3. Copy the key (starts with `sk-...`).
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Free tier:** OpenAI does not have a free tier for Whisper API — it charges per minute of audio (~$0.006/min). For the hackathon demo, this is negligible. Alternatively, you can run Whisper locally (medium model, ~1.5GB) for free.

---

## Section 4: Local Environment Setup

### 4.1 Required Software

Install these if you don't have them:

```bash
# Python 3.11+
python --version  # should print 3.11.x or 3.12.x

# Node.js 20+
node --version    # should print 20.x.x or higher

# Git
git --version

# Docker (optional — only for local Redis if not using Upstash)
docker --version
```

### 4.2 Backend Setup

```bash
cd triageplus/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and fill in the environment file
cp ../.env.example .env
# Open .env and add all keys from Section 3 above

# Run database migrations
alembic upgrade head

# Seed the database
python -m app.scripts.seed_db

# Build FAISS index (after placing corpus files)
python -m app.scripts.build_faiss_index

# Train the classifier (after downloading datasets and running data pipeline)
python ml-training/train_classifier.py

# Start the backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4.3 Frontend Setup

```bash
cd triageplus/frontend

# Install dependencies
npm install

# Copy and fill in Vite environment file
cp .env.example .env
# VITE_API_BASE_URL=http://localhost:8000
# VITE_WS_BASE_URL=ws://localhost:8000
# VITE_STRIPE_PUBLISHABLE_KEY=pk_test_...

# Start the dev server
npm run dev
# Opens at http://localhost:5173
```

### 4.4 Local Redis (if not using Upstash)

```bash
# Using Docker (easiest)
docker run -p 6379:6379 redis:alpine

# Then in .env:
REDIS_URL=redis://localhost:6379
```

---

## Section 5: Deployment

### 5.1 Deploy Backend to Render

1. Go to: https://render.com → create a free account.
2. Click **New → Web Service** → connect your GitHub repo.
3. Set:
   - **Root directory:** `backend`
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `bash entrypoint.sh`
4. Go to **Environment** tab → add all `.env` variables from Section 3.
5. Set `ALLOWED_ORIGINS=["https://your-app.vercel.app"]` once you have the Vercel URL.
6. Deploy. The entrypoint script will run migrations, seed, and start uvicorn.

**Keep-alive:** Go to https://uptimerobot.com → create a free account → add a monitor for `https://your-backend.onrender.com/api/v1/health` every 5 minutes. Render free tier sleeps after 15 minutes of inactivity.

### 5.2 Deploy Frontend to Vercel

1. Go to: https://vercel.com → create a free account.
2. Click **New Project** → import from GitHub.
3. Set:
   - **Framework preset:** Vite
   - **Root directory:** `frontend`
4. Go to **Settings → Environment Variables** → add:
   - `VITE_API_BASE_URL` = your Render backend URL
   - `VITE_WS_BASE_URL` = your Render backend URL (replace `https://` with `wss://`)
   - `VITE_STRIPE_PUBLISHABLE_KEY` = your Stripe publishable key
5. Deploy. Vercel auto-deploys on every push to `main`.

---

## Section 6: Summary Checklist

Complete every item before starting the data pipeline:

### Datasets
- [ ] MedDialog downloaded from Hugging Face (`medical_dialog` dataset)
- [ ] mtsamples downloaded from Kaggle
- [ ] Symptom2Disease downloaded from Kaggle
- [ ] MedQuAD downloaded from Hugging Face (`keivalya/MedQuad-MedicalQnADataset`)
- [ ] MedlinePlus XML downloaded from NLM website
- [ ] CounselChat downloaded (if needed for Psychiatry)
- [ ] `conversations.zip` extracted and organized into specialty subfolders

### API Keys
- [ ] `GEMINI_API_KEY` obtained from Google AI Studio
- [ ] `DATABASE_URL` obtained from Supabase (direct connection, port 5432)
- [ ] `REDIS_URL` obtained from Upstash
- [ ] `STRIPE_SECRET_KEY` and `STRIPE_PUBLISHABLE_KEY` from Stripe dashboard (test mode)
- [ ] `STRIPE_WEBHOOK_SECRET` from Stripe webhook endpoint setup
- [ ] `SENDGRID_API_KEY` from SendGrid
- [ ] `SENDGRID_FROM_EMAIL` verified in SendGrid sender authentication
- [ ] `JWT_SECRET` generated (64-char hex string)
- [ ] Kaggle API key downloaded and placed in `~/.kaggle/kaggle.json`

### Environment
- [ ] Python 3.11+ installed
- [ ] Node.js 20+ installed
- [ ] Backend virtual environment created and dependencies installed
- [ ] `.env` filled in with all keys
- [ ] `alembic upgrade head` ran successfully
- [ ] Seed script ran successfully
- [ ] `GET /health` returns `{"status": "ok", "db": "ok", "redis": "ok"}`
- [ ] FAISS index built (after corpus files are in place)
- [ ] Classifier trained and macro-F1 ≥ 0.70 confirmed
- [ ] Frontend `npm install` ran successfully
- [ ] Frontend dev server starts at `localhost:5173`

---

*TriagePlus · IIT Dharwad Summer of Innovation · Hardly Human · Mentor: Prof. B. N. Bharath*
