"""
Main FastAPI application entry point.

Initializes the framework, middleware, event handlers, and routes.
Includes rate limiting, Prometheus metrics, and OpenTelemetry tracing.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import router
from app.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiter (backed by Redis for distributed deployments)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],
    storage_uri=settings.redis_url,
)


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

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics — exposes /metrics endpoint
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed — metrics disabled")

# OpenTelemetry distributed tracing
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    tracer_provider = TracerProvider()
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.jaeger_host,
        agent_port=settings.jaeger_port,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.set_tracer_provider(tracer_provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info(f"OpenTelemetry tracing enabled → Jaeger at {settings.jaeger_host}:{settings.jaeger_port}")
except Exception as e:
    logger.warning(f"OpenTelemetry tracing not available: {e}")

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
