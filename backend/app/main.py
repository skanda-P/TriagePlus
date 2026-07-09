import sys, os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.")))

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import chat, auth

app = FastAPI(title="TriagePlus API", version="1.0.0")

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:4173")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

# CORS for local development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
