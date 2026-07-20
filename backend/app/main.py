import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import chat, doctor, public
from .db.supabase_client import get_supabase
from .core.unified_retrieval import get_unified_retriever
from .core.triage_graph import build_graph
from langgraph.checkpoint.sqlite import SqliteSaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global graph reference
_graph = None


async def warmup_models():
    """Warm up the unified retriever (loads embeddings + FAISS indices)."""
    try:
        retriever = get_unified_retriever()
        # Accessing the retriever triggers lazy loading
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


async def release_stale_holds_task():
    """Background task to release stale slot holds."""
    while True:
        try:
            supabase = get_supabase()
            res = await asyncio.to_thread(
                supabase.rpc('release_stale_holds', {'p_max_age_minutes': 10}).execute
            )
            logger.info(f"Released stale holds: {res.data}")
        except Exception as e:
            logger.error(f"Error releasing stale holds: {e}")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    
    # Create checkpointer and compile graph
    with SqliteSaver.from_conn_string("sqlite:///checkpoints.db") as checkpointer:
        _graph = build_graph().compile(checkpointer=checkpointer)
        # Inject graph into chat router
        chat.set_graph(_graph)
    
    # Warm up models in background task (non-blocking)
    warmup_task = asyncio.create_task(warmup_models())
    
    # Stale hold reaper
    reaper = asyncio.create_task(release_stale_holds_task())
    
    yield
    
    # Cleanup on shutdown
    warmup_task.cancel()
    reaper.cancel()
    await asyncio.gather(warmup_task, reaper, return_exceptions=True)


app = FastAPI(lifespan=lifespan)

# CORS configuration
origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(chat.router)
app.include_router(doctor.router)
app.include_router(public.router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)