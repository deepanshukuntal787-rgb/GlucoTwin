"""Main FastAPI application entrypoint for GlucoTwin."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gluco_twin.service.config import settings
from gluco_twin.service.db import init_db, SessionLocal, Patient
from gluco_twin.service.registry import ModelRegistry
from gluco_twin.service.routers.patients import router as patients_router
from gluco_twin.service.schemas import HealthResponse

# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("glucotwin.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown lifecycle: initializes DB & pre-warms model."""
    logger.info("Initializing database...")
    init_db()

    # Pre-warm model registry
    logger.info("Pre-warming ModelRegistry and PyTorch device...")
    reg = ModelRegistry.get_instance()
    logger.info(f"ModelRegistry ready on device: {reg.device}")

    yield

    logger.info("GlucoTwin API shutting down...")


app = FastAPI(
    title="GlucoTwin API",
    description="Production REST & WebSocket API for GlucoTwin Probabilistic Physiological Digital Twin",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware (permits Vite frontend at localhost:5173, Docker, or any client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Structured Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration_ms}ms)")
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred.", "path": str(request.url.path)},
    )


# ─── System Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """Health check endpoint reporting model status, device, and database connectivity."""
    reg = ModelRegistry.get_instance()
    db = SessionLocal()
    try:
        p_count = db.query(Patient).count()
    finally:
        db.close()

    return HealthResponse(
        status="healthy",
        model_loaded=bool(reg.model is not None),
        device=str(reg.device),
        patients_count=p_count,
        version="1.0.0",
    )


@app.get("/", tags=["system"])
def root():
    return {
        "service": "GlucoTwin Digital Twin API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "api_v1": "/api/v1",
    }


# Mount API v1 router
app.include_router(patients_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gluco_twin.service.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.env == "development"),
        workers=settings.workers if settings.env != "development" else 1,
    )
