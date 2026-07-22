import os
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from .routers import chat, doctor, public
from .db.supabase_client import get_supabase
from .core.unified_retrieval import get_unified_retriever
from .core.triage_graph import build_graph
from .core.error_handler import TriagePlusException
from .core.ner_symptom_extractor import get_biomedical_ner
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global graph reference
_graph = None


def validate_required_env_vars():
    """Validate required environment variables at startup."""
    required = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Optional but recommended
    optional = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "DEVELOPER_PASSWORD": None,
        "CONFIDENCE_FLOOR": "0.3",
    }
    for var, default in optional.items():
        if not os.getenv(var):
            if default is None:
                logger.warning(f"Optional env var {var} not set - some features may be disabled")
            else:
                logger.info(f"Optional env var {var} not set, using default: {default}")


async def check_ollama_health() -> bool:
    """Check if Ollama is available."""
    import httpx
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]
                expected_model = os.getenv("OLLAMA_MODEL", "llama3.2")
                if expected_model in model_names:
                    logger.info(f"Ollama connected - model '{expected_model}' available")
                    return True
                else:
                    logger.warning(f"Ollama connected but model '{expected_model}' not found. Available: {model_names}")
                    return False
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e} - LLM features will use fallbacks")
    return False


async def warmup_models():
    """Warm up the unified retriever (loads embeddings + FAISS indices) and the NER model."""
    try:
        get_unified_retriever()
        # Accessing the retriever triggers lazy loading
        logger.info("Retriever warmup complete")
    except Exception as e:
        logger.warning(f"Retriever warmup failed: {e}")
    
    # Warm up the NER model (downloads from HF Hub if needed)
    try:
        get_biomedical_ner()
        logger.info("NER model warmup complete")
    except Exception as e:
        logger.warning(f"NER model warmup failed: {e}")

    # Validate XGBoost artifacts exist
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model"))
    xgb_path = os.path.join(model_dir, "xgb_model.json")
    mlb_path = os.path.join(model_dir, "mlb.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(mlb_path) and os.path.exists(le_path)):
        logger.warning(
            f"XGBoost artifacts not found in {model_dir}. "
            "Triage classification will fall back to 'Uncertain Diagnosis'. "
            "Run scripts/train_xgboost.py to generate them."
        )
    else:
        logger.info("XGBoost artifacts present")


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
    
    # Validate required environment variables
    validate_required_env_vars()
    
    # Check Ollama availability (non-blocking)
    ollama_available = await check_ollama_health()
    if not ollama_available:
        logger.warning("Ollama not available - LLM features will use fallback responses")
    
    # Create checkpointer and compile graph - keep connection open for app lifetime
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        _graph = build_graph().compile(checkpointer=checkpointer)
        # Inject graph into chat router
        chat.set_graph(_graph)
        
        # Warm up models in background task (non-blocking)
        warmup_task = asyncio.create_task(warmup_models())
        
        # Stale hold reaper
        reaper = asyncio.create_task(release_stale_holds_task())
        
        try:
            yield
        finally:
            # Cleanup on shutdown
            warmup_task.cancel()
            reaper.cancel()
            await asyncio.gather(warmup_task, reaper, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(TriagePlusException)
async def triageplus_exception_handler(request: Request, exc: TriagePlusException):
    """Convert TriagePlusException into the documented structured error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.to_dict()["error"]},
    )


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