"""Workers package."""

from app.workers.celery_worker import celery_app, run_tumor_inference

__all__ = [
    "celery_app",
    "run_tumor_inference",
]
