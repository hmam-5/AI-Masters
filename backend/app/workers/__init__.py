"""Workers package."""

from app.workers.sync_inference import run_inference_sync

__all__ = [
    "run_inference_sync",
]
