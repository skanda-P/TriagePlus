import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import chat, doctor, public
from .db.supabase_client import get_supabase
from .core.rag import load_rag_models

async def release_stale_holds_task():
    while True:
        try:
            supabase = get_supabase()
            res = supabase.rpc('release_stale_holds', {'p_max_age_minutes': 10}).execute()
            print(f"Released stale holds: {res.data}")
        except Exception as e:
            print(f"Error releasing stale holds: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models in background task (non-blocking)
    import asyncio
    warmup_task = asyncio.create_task(asyncio.to_thread(load_rag_models))
    
    # Stale hold reaper
    reaper = asyncio.create_task(release_stale_holds_task())
    
    yield
    
    warmup_task.cancel()
    reaper.cancel()

app = FastAPI(lifespan=lifespan)

origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(doctor.router)
app.include_router(public.router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
