"""
Main FastAPI application entry point.

Initializes the framework, middleware, event handlers, and routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Brain Tumor AI Framework")

    # Ensure MinIO buckets exist
    try:
        from app.services.storage import get_storage_backend
        storage = get_storage_backend()
        if hasattr(storage, 'client'):
            for bucket in [settings.minio_bucket_dicom, settings.minio_bucket_results]:
                if not storage.client.bucket_exists(bucket):
                    storage.client.make_bucket(bucket)
                    logger.info(f"Created MinIO bucket: {bucket}")
    except Exception as e:
        logger.error(f"Failed to initialize storage buckets: {e}")

    # Initialize FalkorDB schema
    try:
        from app.services.graph_db import get_falkordb
        graph_db = get_falkordb()
        graph_db.initialize_schema()
        logger.info("FalkorDB graph schema initialized")
    except Exception as e:
        logger.error(f"Failed to initialize FalkorDB: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Brain Tumor AI Framework")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Enterprise AI Framework for Brain Tumor Segmentation and Classification",
    version=settings.app_version,
    lifespan=lifespan,
)

# Middleware
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts (in production, set to specific domains)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.example.com", "example.com"],
    )


# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "detail": str(exc) if settings.debug else None,
        },
    )


# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )
