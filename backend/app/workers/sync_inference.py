"""
Synchronous inference for local development.

This module provides the same inference logic as the Celery worker,
but runs synchronously (blocking) without requiring Redis/Celery.

Use this when `use_sync_inference=True` in settings.
"""

import io
import logging
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from app.config import get_settings
from app.dataset.ensemble import EnsembleEngine
from app.dataset.trainer import CLASSES
from app.models.database import JobStatus
from app.services.graph_db import get_falkordb
from app.services.storage import get_storage_backend

logger = logging.getLogger(__name__)
settings = get_settings()


def run_inference_sync(
    job_id: str,
    scan_id: str,
    modality_paths: dict[str, str],
) -> dict:
    """
    Execute tumor analysis inference synchronously (no Celery).

    This is the same logic as run_tumor_inference but runs in-process.
    Use this for local development without Redis/Celery.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage = get_storage_backend()
    gdb = get_falkordb()

    try:
        logger.info(f"[SYNC] Starting ensemble inference for job {job_id}")

        # Update job status
        job = gdb.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        gdb.update_job(job_id, status=JobStatus.PROCESSING.value,
                       started_at=datetime.utcnow().isoformat(), progress_percentage=10)

        # 1. Load and preprocess image
        logger.info("[SYNC] Loading image from storage")
        is_regular_image = modality_paths.get("is_regular_image", False)
        image_path = modality_paths.get("image_path", "")

        gdb.update_job(job_id, progress_percentage=20)

        if is_regular_image and image_path:
            image_bytes = storage.download_file(image_path)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            pil_image = Image.new("RGB", (settings.image_size, settings.image_size))

        gdb.update_job(job_id, progress_percentage=30)

        # 2. Load ensemble (all 4 models)
        logger.info("[SYNC] Loading multi-model ensemble")
        gdb.update_job(job_id, progress_percentage=40)

        ensemble = EnsembleEngine(device=device, image_size=settings.image_size)
        models_loaded = ensemble.num_models
        logger.info(f"[SYNC] Ensemble loaded {models_loaded} models: {ensemble.model_names}")
        if models_loaded == 0:
            raise RuntimeError("No trained models found. Run training before inference.")

        # 3. Run ensemble classification with Test-Time Augmentation
        logger.info("[SYNC] Running ensemble classification with TTA")
        gdb.update_job(job_id, progress_percentage=60)

        grade_mapping = {
            "glioma": "Grade IV",
            "meningioma": "Grade II",
            "no_tumor": "No Tumor",
            "pituitary": "Grade III",
        }
        tumor_type_mapping = {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "no_tumor": None,
            "pituitary": "Pituitary",
        }

        ensemble_result = ensemble.predict_with_tta(pil_image)

        predicted_class = ensemble_result["predicted_class"]
        top_confidence = ensemble_result["confidence"]
        min_conf_for_auto = float(getattr(settings, "min_confidence_for_auto_decision", 0.99))

        auto_decision = top_confidence >= min_conf_for_auto
        predicted_grade = grade_mapping.get(predicted_class, "Unknown")
        predicted_tumor_type = tumor_type_mapping.get(predicted_class)
        probs = ensemble_result["probabilities"]

        final_predicted_class = predicted_class if auto_decision else "review_required"
        if not auto_decision:
            predicted_grade = "Indeterminate"
            predicted_tumor_type = None

        # 4. Compare with historical high-confidence cases in FalkorDB
        logger.info("[SYNC] Comparing with historical cases")
        gdb.update_job(job_id, progress_percentage=75)

        segmentation_results = []
        similar_cases = []
        if auto_decision and predicted_class != "no_tumor":
            try:
                similar_cases = gdb.find_similar_cases(predicted_grade, min_confidence=0.9)
            except Exception as compare_err:
                logger.warning(f"[SYNC] Similar case comparison failed (non-fatal): {compare_err}")

        # 5. Save classification result
        logger.info("[SYNC] Saving classification results")
        gdb.update_job(job_id, progress_percentage=85)

        classification_details = {
            "probabilities": {
                cls_name: round(float(probs.get(cls_name, 0.0)), 4)
                for cls_name in CLASSES
            },
            "predicted_class": final_predicted_class,
            "raw_predicted_class": predicted_class,
            "image_processed": True,
            "model_type": "Ensemble (4 models + TTA)" if models_loaded > 0 else "No models loaded",
            "models_loaded": models_loaded,
            "model_names": ensemble.model_names,
            "agreement_score": ensemble_result.get("agreement_score", 0.0),
            "tta_variants": ensemble_result.get("tta_variants", 0),
            "per_model_predictions": ensemble_result.get("per_model_predictions", {}),
            "similar_cases": similar_cases[:5],
            "similar_cases_count": len(similar_cases),
            "decision_status": "auto_accepted" if auto_decision else "review_required",
            "min_confidence_threshold": min_conf_for_auto,
        }

        gdb.save_classification_result(
            job_id=job_id,
            tumor_grade=predicted_grade,
            confidence_score=float(top_confidence),
            classification_details=classification_details,
        )

        # 6. Store analysis result in graph (for analytics)
        logger.info("[SYNC] Storing analysis result in FalkorDB graph")
        gdb.update_job(job_id, progress_percentage=90)

        try:
            scan = gdb.get_scan(scan_id)
            patient_mrn = scan.get("patient_mrn", "UNKNOWN") if scan else "UNKNOWN"

            gdb.store_analysis_result(
                job_id=job_id,
                patient_mrn=patient_mrn,
                scan_id=scan_id,
                tumor_grade=predicted_grade,
                confidence=float(top_confidence),
                tumor_type=predicted_tumor_type,
                segmentation_results=segmentation_results,
                classification_details=classification_details,
            )
        except Exception as graph_err:
            logger.warning(f"[SYNC] Failed to store analysis graph (non-fatal): {graph_err}")

        # 7. Mark complete
        gdb.update_job(job_id, status=JobStatus.COMPLETED.value,
                       completed_at=datetime.utcnow().isoformat(), progress_percentage=100)

        logger.info(
            f"[SYNC] Ensemble inference completed for job {job_id} — final={final_predicted_class}, "
            f"raw={predicted_class}, conf={top_confidence:.4f}, auto_decision={auto_decision}, "
            f"models={models_loaded}, agreement={ensemble_result.get('agreement_score', 0):.2f}"
        )
        return {
            "job_id": job_id,
            "status": "completed",
            "segmentation_results": segmentation_results,
            "classification": {
                "class": final_predicted_class,
                "grade": predicted_grade,
                "tumor_type": predicted_tumor_type,
                "confidence": float(top_confidence),
                "auto_decision": auto_decision,
                "min_confidence_threshold": min_conf_for_auto,
                "models_loaded": models_loaded,
                "model_names": ensemble.model_names,
                "agreement_score": ensemble_result.get("agreement_score", 0.0),
                "per_model_predictions": ensemble_result.get("per_model_predictions", {}),
            },
        }

    except Exception as e:
        logger.error(f"[SYNC] Inference failed: {str(e)}", exc_info=True)
        try:
            gdb.update_job(job_id, status=JobStatus.FAILED.value,
                           error_message=str(e),
                           completed_at=datetime.utcnow().isoformat())
        except Exception as db_error:
            logger.error(f"[SYNC] Failed to update job status: {str(db_error)}")
        raise
