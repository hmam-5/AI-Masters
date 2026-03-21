"""
Celery worker configuration and task definitions.

Handles asynchronous inference for segmentation and classification.
"""

import logging
from typing import Optional

from celery import Celery, Task
from celery.utils.log import get_task_logger

from app.config import get_settings

settings = get_settings()

# Configure Celery
celery_app = Celery(
    "brain_tumor_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

logger = get_task_logger(__name__)


class CallbackTask(Task):
    """Task base class with callbacks for progress updates."""

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} is being retried: {exc}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")

    def on_success(self, result, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Task {task_id} completed successfully")


celery_app.Task = CallbackTask


@celery_app.task(bind=True, max_retries=3)
def run_tumor_inference(
    self,
    job_id: str,
    scan_id: str,
    modality_paths: dict[str, str],
) -> dict:
    """
    Execute tumor analysis inference using multi-model ensemble.

    All persistence is handled via FalkorDB.
    """
    import io
    from datetime import datetime

    import torch
    from PIL import Image

    from app.dataset.ensemble import EnsembleEngine
    from app.dataset.trainer import CLASSES
    from app.models.database import JobStatus
    from app.services.graph_db import get_falkordb
    from app.services.storage import get_storage_backend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    storage = get_storage_backend()
    gdb = get_falkordb()

    try:
        logger.info(f"Starting ensemble inference for job {job_id}")

        # Update job status
        job = gdb.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        gdb.update_job(job_id, status=JobStatus.PROCESSING.value,
                       started_at=datetime.utcnow().isoformat(), progress_percentage=10)
        self.update_state(state="PROGRESS", meta={"progress": 10, "stage": "loading"})

        # 1. Load and preprocess image
        logger.info("Loading image from storage")
        is_regular_image = modality_paths.get("is_regular_image", False)
        image_path = modality_paths.get("image_path", "")

        gdb.update_job(job_id, progress_percentage=20)
        self.update_state(state="PROGRESS", meta={"progress": 20, "stage": "preprocessing"})

        if is_regular_image and image_path:
            image_bytes = storage.download_file(image_path)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            pil_image = Image.new("RGB", (settings.image_size, settings.image_size))

        gdb.update_job(job_id, progress_percentage=30)
        self.update_state(state="PROGRESS", meta={"progress": 30, "stage": "preprocessing_done"})

        # 2. Load ensemble (all 4 models)
        logger.info("Loading multi-model ensemble")
        gdb.update_job(job_id, progress_percentage=40)
        self.update_state(state="PROGRESS", meta={"progress": 40, "stage": "loading_ensemble"})

        ensemble = EnsembleEngine(device=device, image_size=settings.image_size)
        models_loaded = ensemble.num_models
        logger.info(f"Ensemble loaded {models_loaded} models: {ensemble.model_names}")
        if models_loaded == 0:
            raise RuntimeError("No trained models found. Run training before inference.")

        # 3. Run ensemble classification with Test-Time Augmentation
        logger.info("Running ensemble classification with TTA")
        gdb.update_job(job_id, progress_percentage=60)
        self.update_state(state="PROGRESS", meta={"progress": 60, "stage": "ensemble_classification"})

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
        logger.info("Comparing with historical cases")
        gdb.update_job(job_id, progress_percentage=75)
        self.update_state(state="PROGRESS", meta={"progress": 75, "stage": "case_comparison"})

        segmentation_results = []
        similar_cases = []
        if auto_decision and predicted_class != "no_tumor":
            try:
                similar_cases = gdb.find_similar_cases(predicted_grade, min_confidence=0.9)
            except Exception as compare_err:
                logger.warning(f"Similar case comparison failed (non-fatal): {compare_err}")

        # 5. Save classification result
        logger.info("Saving classification results")
        gdb.update_job(job_id, progress_percentage=85)
        self.update_state(state="PROGRESS", meta={"progress": 85, "stage": "saving"})

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
        logger.info("Storing analysis result in FalkorDB graph")
        gdb.update_job(job_id, progress_percentage=90)
        self.update_state(state="PROGRESS", meta={"progress": 90, "stage": "graph_storage"})

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
            logger.warning(f"Failed to store analysis graph (non-fatal): {graph_err}")

        # 7. Mark complete
        gdb.update_job(job_id, status=JobStatus.COMPLETED.value,
                       completed_at=datetime.utcnow().isoformat(), progress_percentage=100)

        logger.info(
            f"Ensemble inference completed for job {job_id} — final={final_predicted_class}, "
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
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        try:
            gdb.update_job(job_id, status=JobStatus.FAILED.value,
                           error_message=str(e),
                           completed_at=datetime.utcnow().isoformat())
        except Exception as db_error:
            logger.error(f"Failed to update job status: {str(db_error)}")

        if isinstance(e, (IOError, OSError)):
            raise self.retry(exc=e, countdown=60, max_retries=3)
        raise


        if not data_roots:
            raise FileNotFoundError(
                "No usable dataset roots found. Expected Kaggle dataset under /data/datasets "
                "or organized dataset under /data/combined."
            )

        logger.info(f"Training will use {len(data_roots)} dataset root(s): {data_roots}")

        # Store dataset metadata in FalkorDB
        try:
            from app.services.graph_db import get_falkordb
            graph_db = get_falkordb()
            from app.dataset.trainer import CLASS_ALIASES

            images = []
            for root in data_roots:
                root_path = Path(root)
                if root_path.name.lower() == "datasets":
                    split_map = {"train": root_path / "Training", "val": root_path / "Testing"}
                else:
                    split_map = {"train": root_path / "train", "val": root_path / "val"}

                for split, split_dir in split_map.items():
                    if not split_dir.exists():
                        continue
                    for cls_dir in split_dir.iterdir():
                        if cls_dir.is_dir():
                            class_label = CLASS_ALIASES.get(cls_dir.name.lower(), cls_dir.name.lower())
                            for img_path in cls_dir.iterdir():
                                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                    images.append({
                                        "path": str(img_path),
                                        "class_label": class_label,
                                        "split": split,
                                    })

            if images:
                graph_db.store_dataset_metadata(images, "combined_multi_source")
                logger.info(f"Stored {len(images)} images in FalkorDB")
        except Exception as graph_err:
            logger.warning(f"FalkorDB dataset storage failed (non-fatal): {graph_err}")

        # Train the model
        self.update_state(state="PROGRESS", meta={"stage": "training"})
        logger.info("Starting multi-model ensemble training (4 models)...")

        from app.dataset.trainer import train_all_models

        cpu_mode = not torch.cuda.is_available()
        epochs = 8 if cpu_mode else 30
        batch_size = 24 if cpu_mode else 32
        image_size = 160 if cpu_mode else settings.image_size
        patience = 4 if cpu_mode else 10
        logger.info(
            f"Training config: cpu_mode={cpu_mode}, epochs={epochs}, "
            f"batch_size={batch_size}, image_size={image_size}, patience={patience}"
        )

        results = train_all_models(
            data_dir=data_roots,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            patience=patience,
            image_size=image_size,
        )

        # Store training runs for each model in FalkorDB
        try:
            from app.services.graph_db import get_falkordb
            graph_db = get_falkordb()
            for model_name, model_result in results.items():
                run_id = str(uuid.uuid4())
                graph_db.store_training_run(
                    run_id=run_id,
                    accuracy=model_result["best_val_accuracy"],
                    epochs=model_result["epochs_trained"],
                    model_path=model_result["model_path"],
                    class_distribution={**stats, "model_name": model_name},
                )
        except Exception as graph_err:
            logger.warning(f"FalkorDB training run storage failed (non-fatal): {graph_err}")

        # Build summary
        summary = {}
        for model_name, model_result in results.items():
            model_result.pop("history", None)
            summary[model_name] = model_result

        best_acc = max(r["best_val_accuracy"] for r in results.values())
        logger.info(f"Ensemble training complete. Best individual accuracy: {best_acc:.4f}")
        return {"models": summary, "best_individual_accuracy": best_acc}

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
