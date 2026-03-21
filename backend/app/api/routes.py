"""
FastAPI API routes for file uploads and inference management.

All persistence is handled via FalkorDB — no SQLAlchemy.
"""

import logging
import uuid
from datetime import datetime

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from app.config import get_settings
from app.models.database import JobStatus
from app.schemas import (
    AnalysisResultResponse,
    ImageAnalyzeResponse,
)
from app.services.graph_db import get_falkordb
from app.services.storage import get_storage_backend
from app.utils.validators import ImageValidationError, NIfTIValidator, DICOMValidator, RegularImageValidator
from app.workers import run_tumor_inference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["medical-imaging"])
settings = get_settings()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict) -> None:
        if job_id not in self.active_connections:
            return
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {str(e)}")


manager = ConnectionManager()


# ── Simple Image Analysis Endpoint (PNG/JPG) ─────────────────────

@router.post("/analyze", response_model=ImageAnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
) -> ImageAnalyzeResponse:
    """Upload a brain MRI image and start AI analysis."""
    gdb = get_falkordb()
    storage = get_storage_backend()

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a filename")

    filename_lower = file.filename.lower()
    is_regular_image = filename_lower.endswith(('.png', '.jpg', '.jpeg'))
    is_medical = filename_lower.endswith(('.nii', '.nii.gz', '.dcm'))

    if not is_regular_image and not is_medical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file format. Please upload PNG, JPG, JPEG, DICOM, or NIfTI files.",
        )

    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"File size exceeds {settings.max_file_size_mb}MB limit",
        )

    try:
        if is_regular_image:
            metadata = RegularImageValidator.validate(file_content)
        elif filename_lower.endswith(('.nii', '.nii.gz')):
            metadata = NIfTIValidator.validate(file_content)
        else:
            metadata = DICOMValidator.validate(file_content)
    except ImageValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Image validation failed: {str(e)}")

    # Get or create demo patient in FalkorDB
    patient = gdb.get_or_create_demo_patient()
    patient_mrn = patient["mrn"]

    # Create scan in FalkorDB
    scan_id = str(uuid.uuid4())
    storage_path = f"scans/{patient_mrn}/scan_{scan_id}_{file.filename}"
    gdb.create_scan(scan_id, patient_mrn, ["image"], storage_location=storage_path)

    # Upload to object storage
    storage.upload_file(storage_path, file_content)

    # Create job in FalkorDB
    job_id = str(uuid.uuid4())
    gdb.create_job(job_id, scan_id)

    # Start Celery task
    try:
        celery_task = run_tumor_inference.apply_async(
            args=[job_id, scan_id, {
                "image_path": storage_path,
                "filename": file.filename,
                "is_regular_image": is_regular_image,
                "metadata": metadata,
            }],
            task_id=job_id,
        )
        gdb.update_job(job_id, celery_task_id=celery_task.id)
    except Exception as e:
        gdb.update_job(job_id, status=JobStatus.FAILED.value, error_message=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}",
        )

    return ImageAnalyzeResponse(
        job_id=uuid.UUID(job_id),
        scan_id=uuid.UUID(scan_id),
        filename=file.filename,
        status="processing",
        message="Image uploaded successfully. Analysis started.",
    )


# ── Results Endpoint ──────────────────────────────────────────────

@router.get("/inference/{job_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(job_id: str) -> AnalysisResultResponse:
    """Get the full analysis results for a completed inference job."""
    gdb = get_falkordb()

    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job_id format")

    job = gdb.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    if job["status"] in (JobStatus.PROCESSING.value, JobStatus.PENDING.value):
        raise HTTPException(status_code=status.HTTP_202_ACCEPTED, detail="Analysis is still in progress")

    if job["status"] == JobStatus.FAILED.value:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {job['error_message']}")

    # Fetch results from FalkorDB
    clf_results = gdb.get_classification_results(job_id)
    seg_results = gdb.get_segmentation_results(job_id)

    # Get filename from scan
    scan = gdb.get_scan(job["scan_id"])
    filename = (scan["storage_location"].split("/")[-1] if scan and scan.get("storage_location") else "unknown")

    tumor_detected = len(clf_results) > 0
    tumor_grade = clf_results[0]["tumor_grade"] if clf_results else None
    confidence = clf_results[0]["confidence_score"] if clf_results else 0.0
    clf_details = clf_results[0]["classification_details"] if clf_results else None

    predicted_class = clf_details.get("predicted_class", "unknown") if clf_details else "unknown"
    decision_status = clf_details.get("decision_status", "auto_accepted") if clf_details else "auto_accepted"
    tumor_type_map = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary"}
    actual_tumor_type = tumor_type_map.get(predicted_class)

    if predicted_class in ("no_tumor", "unknown", "review_required"):
        tumor_detected = False

    if decision_status == "review_required":
        tumor_detected = False
        tumor_grade = None
        actual_tumor_type = None

    # Build findings
    findings = []
    if decision_status == "review_required":
        threshold = (clf_details or {}).get("min_confidence_threshold", 0.99)
        findings.append(
            f"Prediction confidence ({confidence:.1%}) is below strict auto-decision threshold ({threshold:.0%})"
        )
        findings.append("Case automatically routed for specialist review")
    elif tumor_detected:
        findings.append(f"Tumor detected with {confidence:.1%} confidence")
        if tumor_grade:
            findings.append(f"Classified as {tumor_grade}")
        if clf_details and clf_details.get("similar_cases_count") is not None:
            findings.append(
                f"Matched against {clf_details.get('similar_cases_count', 0)} similar historical cases"
            )
        for seg in seg_results:
            findings.append(
                f"{seg['subregion'].replace('_', ' ').title()}: "
                f"confidence {seg['confidence_score']:.1%}, volume {seg['volume_mm3']:.1f} mm³"
            )
    else:
        findings.append("No significant abnormalities detected")

    explanation = _build_explanation(tumor_detected, tumor_grade, confidence, seg_results, clf_details)
    recommendations = _build_recommendations(tumor_detected, tumor_grade, confidence)

    seg_summary = {}
    for seg in seg_results:
        seg_summary[seg["subregion"]] = {"confidence": seg["confidence_score"], "volume_mm3": seg["volume_mm3"]}

    return AnalysisResultResponse(
        job_id=uuid.UUID(job_id),
        status="completed",
        image_filename=filename,
        tumor_detected=tumor_detected,
        tumor_type=actual_tumor_type if tumor_detected else None,
        tumor_grade=tumor_grade,
        confidence=confidence,
        findings=findings,
        explanation=explanation,
        recommendations=recommendations,
        classification_details=clf_details,
        segmentation_summary=seg_summary if seg_summary else None,
        completed_at=job["completed_at"],
    )


# ── Graph Analytics Endpoints ─────────────────────────────────────

@router.get("/analytics/history/{patient_mrn}")
async def get_patient_history(patient_mrn: str):
    gdb = get_falkordb()
    return gdb.get_analysis_history(patient_mrn)


@router.get("/analytics/grades")
async def get_grade_stats():
    gdb = get_falkordb()
    return gdb.get_grade_statistics()


@router.get("/analytics/dataset")
async def get_dataset_overview():
    gdb = get_falkordb()
    return gdb.get_dataset_overview()


@router.get("/analytics/training")
async def get_training_history():
    gdb = get_falkordb()
    return gdb.get_training_history()


# ── Helpers ───────────────────────────────────────────────────────

def _build_explanation(tumor_detected, tumor_grade, confidence, seg_results, clf_details):
    if clf_details and clf_details.get("decision_status") == "review_required":
        threshold = clf_details.get("min_confidence_threshold", 0.99)
        return (
            f"The AI model produced a confidence of {confidence:.1%}, which is below the strict "
            f"auto-decision threshold of {threshold:.0%}. To prioritize safety, this case was marked "
            "as indeterminate and requires review by a qualified radiologist before any diagnosis."
        )

    if not tumor_detected:
        return (
            "The AI analysis of the provided brain MRI image did not detect any significant "
            "tumor-like abnormalities. The brain structures appear within normal parameters "
            "based on the model's assessment. However, this is an AI-assisted analysis and "
            "should be reviewed by a qualified radiologist for clinical decision-making."
        )

    grade_info = {
        "Grade II": "a low-grade glioma, which is a slow-growing tumor",
        "Grade III": "an anaplastic glioma, which is a moderately aggressive tumor",
        "Grade IV": "a glioblastoma (GBM), which is the most aggressive type of primary brain tumor",
    }
    grade_desc = grade_info.get(tumor_grade, f"a brain tumor classified as {tumor_grade}")

    parts = [f"The AI analysis has detected {grade_desc} with a confidence level of {confidence:.1%}."]

    for seg in seg_results:
        region = seg["subregion"].replace("_", " ")
        conf = seg["confidence_score"]
        vol = seg["volume_mm3"]
        if region == "enhancing tumor":
            parts.append(f"The enhancing tumor region (actively growing area) was identified with {conf:.1%} confidence, measuring approximately {vol:.1f} mm³.")
        elif region == "edema":
            parts.append(f"Peritumoral edema (swelling around the tumor) was detected with {conf:.1%} confidence, measuring approximately {vol:.1f} mm³.")
        elif region == "necrotic core":
            parts.append(f"A necrotic core (dead tissue within the tumor) was identified with {conf:.1%} confidence, measuring approximately {vol:.1f} mm³.")

    if clf_details and "probabilities" in clf_details:
        probs = clf_details["probabilities"]
        parts.append("Classification probability breakdown: " + ", ".join(f"{k}: {v:.1%}" for k, v in probs.items()))

    if clf_details and clf_details.get("similar_cases_count") is not None:
        sim_count = clf_details.get("similar_cases_count", 0)
        parts.append(
            f"The prediction was compared with {sim_count} historical cases in FalkorDB "
            "with matching grade and high confidence."
        )

    if confidence < 0.95:
        parts.append(
            "Confidence is below 95%, so this result should be treated as uncertain and "
            "confirmed with specialist review and/or additional imaging."
        )

    parts.append(
        "IMPORTANT: This is an AI-assisted analysis for educational/research purposes. "
        "Always consult a qualified medical professional for diagnosis and treatment decisions."
    )
    return " ".join(parts)


def _build_recommendations(tumor_detected, tumor_grade, confidence):
    if confidence < 0.99:
        return [
            "Auto-decision blocked: confidence below strict 99% threshold",
            "Require specialist radiology review before any clinical conclusion",
            "Consider repeat imaging and/or additional MRI sequences for confirmation",
        ]

    if not tumor_detected:
        return [
            "No immediate concerns detected by AI analysis",
            "Routine follow-up imaging as recommended by your physician",
            "Consult a radiologist for official interpretation",
        ]

    recs = [
        "Consult a neuro-oncologist for clinical evaluation",
        "Obtain a formal radiological interpretation",
    ]

    if tumor_grade == "Grade IV":
        recs.extend([
            "Consider urgent neurosurgical consultation",
            "Discuss treatment options including surgery, radiation, and chemotherapy",
            "Molecular testing (IDH, MGMT) recommended for treatment planning",
        ])
    elif tumor_grade == "Grade III":
        recs.extend([
            "Neurosurgical consultation recommended",
            "Consider molecular profiling for treatment planning",
            "Follow-up MRI recommended in 4-6 weeks",
        ])
    else:
        recs.extend([
            "Follow-up MRI recommended in 3-6 months",
            "Monitor for any symptom changes",
            "Discuss observation vs. treatment options with your physician",
        ])

    if confidence < 0.95:
        recs.append("Confidence below 95% - request specialist radiology review before clinical action")
        recs.append("Consider repeat MRI protocol or additional sequence acquisition for confirmation")

    return recs


# ── WebSocket ─────────────────────────────────────────────────────

@router.websocket("/ws/job/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str) -> None:
    gdb = get_falkordb()

    try:
        uuid.UUID(job_id)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid job_id")
        return

    job = gdb.get_job(job_id)
    if not job:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Job not found")
        return

    await manager.connect(websocket, job_id)

    try:
        await websocket.send_json({
            "event": "status_update",
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress_percentage"],
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"event": "echo", "data": data})

    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")


# ── Health & Utility Endpoints ────────────────────────────────────

@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "brain-tumor-ai",
        "version": settings.app_version,
    }


@router.get("/graph/stats")
async def get_graph_statistics() -> dict:
    """Get aggregate analysis statistics from FalkorDB."""
    try:
        gdb = get_falkordb()
        return {
            "grade_statistics": gdb.get_grade_statistics(),
            "dataset_overview": gdb.get_dataset_overview(),
            "training_history": gdb.get_training_history(),
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"FalkorDB query failed: {str(e)}")


@router.get("/graph/patient/{mrn}/history")
async def get_patient_analysis_history(mrn: str) -> dict:
    """Get all analysis results for a patient from the graph database."""
    try:
        gdb = get_falkordb()
        history = gdb.get_analysis_history(mrn)
        return {"mrn": mrn, "analyses": history}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"FalkorDB query failed: {str(e)}")


@router.get("/graph/similar/{tumor_grade}")
async def find_similar_cases(tumor_grade: str) -> dict:
    """Find similar historical cases from the graph database."""
    try:
        gdb = get_falkordb()
        cases = gdb.find_similar_cases(tumor_grade, 0.8)
        return {"tumor_grade": tumor_grade, "similar_cases": cases}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"FalkorDB query failed: {str(e)}")


@router.post("/training/start")
async def start_training() -> dict:
    """Trigger ensemble training."""
    try:
        from app.workers.celery_worker import train_model_task
        task = train_model_task.apply_async()
        return {
            "task_id": task.id,
            "status": "training_started",
            "message": "Ensemble training queued. Check task status for progress.",
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to start training: {str(e)}")


@router.get("/ensemble/status")
async def get_ensemble_status() -> dict:
    """Check which trained models are available for ensemble inference."""
    from pathlib import Path
    try:
        from app.dataset.models import MODEL_REGISTRY
    except ImportError:
        return {"error": "Model registry not available"}

    status_info = {}
    for name, info in MODEL_REGISTRY.items():
        model_path = info["path"]
        exists = Path(model_path).exists()
        status_info[name] = {"path": model_path, "trained": exists, "weight": info["weight"]}

    trained_count = sum(1 for v in status_info.values() if v["trained"])
    return {
        "models": status_info,
        "total_models": len(status_info),
        "trained_models": trained_count,
        "ensemble_ready": trained_count >= 2,
        "full_ensemble": trained_count == len(status_info),
    }
