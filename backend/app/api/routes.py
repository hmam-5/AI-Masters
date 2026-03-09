"""
FastAPI API routes for file uploads and inference management.

Implements RESTful endpoints and WebSocket connections for real-time updates.
"""

import logging
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import InferenceJob, JobStatus, MRIScan, Patient
from app.models.session import get_db
from app.schemas import (
    AnalysisResultResponse,
    ErrorResponse,
    FileUploadResponse,
    ImageAnalyzeResponse,
    InferenceJobCreate,
    InferenceJobResponse,
    MRIScanCreate,
    MRIScanResponse,
    MRIScanUploadResponse,
    PatientCreate,
    PatientResponse,
)
from app.services.storage import get_storage_backend
from app.utils.validators import ImageValidationError, NIfTIValidator, DICOMValidator, RegularImageValidator
from app.workers import run_tumor_inference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["medical-imaging"])
settings = get_settings()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Connect WebSocket for a job."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Disconnect WebSocket."""
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict) -> None:
        """Broadcast message to all connected clients for a job."""
        if job_id not in self.active_connections:
            return

        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {str(e)}")


manager = ConnectionManager()


# Patient Endpoints
@router.post("/patients", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
) -> PatientResponse:
    """
    Create a new patient record.

    Args:
        patient: Patient creation data
        db: Database session

    Returns:
        PatientResponse: Created patient

    Raises:
        HTTPException: If MRN already exists
    """
    # Check if MRN exists
    existing = db.query(Patient).filter(Patient.mrn == patient.mrn).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient with MRN {patient.mrn} already exists",
        )

    db_patient = Patient(
        mrn=patient.mrn,
        date_of_birth=patient.date_of_birth,
        sex=patient.sex,
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    logger.info(f"Created patient: {db_patient.id}")
    return PatientResponse.from_orm(db_patient)


@router.get("/patients/{mrn}", response_model=PatientResponse)
async def get_patient(
    mrn: str,
    db: Session = Depends(get_db),
) -> PatientResponse:
    """
    Get patient by MRN.

    Args:
        mrn: Medical Record Number
        db: Database session

    Returns:
        PatientResponse: Patient data

    Raises:
        HTTPException: If patient not found
    """
    patient = db.query(Patient).filter(Patient.mrn == mrn).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {mrn} not found",
        )
    return PatientResponse.from_orm(patient)


# MRI Upload Endpoints
@router.post("/scans/upload", response_model=MRIScanUploadResponse)
async def upload_mri_scan(
    patient_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> MRIScanUploadResponse:
    """
    Upload multi-modal MRI scan (T1, T1ce, T2, FLAIR).

    Supports chunked uploads and validates file integrity.

    Args:
        patient_id: Patient UUID
        files: List of MRI DICOM/NIfTI files
        db: Database session

    Returns:
        MRIScanUploadResponse: Upload metadata

    Raises:
        HTTPException: If validation or upload fails
    """
    storage = get_storage_backend()

    # Auto-create demo patient if patient_id is not a valid UUID
    try:
        patient_uuid = uuid.UUID(patient_id)
    except ValueError:
        # Create or fetch a demo patient for easy testing
        patient = db.query(Patient).filter(Patient.mrn == "DEMO-001").first()
        if not patient:
            from datetime import date
            patient = Patient(
                mrn="DEMO-001",
                date_of_birth=datetime(1990, 1, 1),
                sex="Unknown",
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)
        patient_uuid = patient.id

    patient = db.query(Patient).filter(Patient.id == patient_uuid).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found",
        )

    # Create scan record
    scan = MRIScan(
        patient_id=patient.id,
        scan_date=datetime.utcnow(),
        modalities=[],
        storage_location="",
    )

    uploaded_modalities = {}
    total_size = 0

    try:
        for file in files:
            # Validate file
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must have a filename",
                )

            # Extract modality from filename
            modality = None
            for mod in ["T1", "T1ce", "T2", "FLAIR"]:
                if mod.lower() in file.filename.lower():
                    modality = mod
                    break

            if not modality:
                logger.warning(f"Could not determine modality for {file.filename}")
                modality = f"Unknown_{file.filename}"

            # Read file content
            file_content = await file.read()
            file_size = len(file_content)

            # Validate total size
            total_size += file_size
            if total_size > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
                    detail=f"Total upload size exceeds {settings.max_file_size_mb}MB",
                )

            # Validate image format
            try:
                if file.filename.endswith(".nii") or file.filename.endswith(".nii.gz"):
                    metadata = NIfTIValidator.validate(file_content)
                elif file.filename.endswith(".dcm"):
                    metadata = DICOMValidator.validate(file_content, modality)
                else:
                    raise ImageValidationError("Unsupported file format")

                logger.info(f"Validated {modality}: {metadata}")
            except ImageValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image validation failed for {modality}: {str(e)}",
                )

            # Upload to storage
            storage_path = f"scans/{patient.mrn}/scan_{scan.id}_{modality}_{file.filename}"
            try:
                storage.upload_file(storage_path, file_content)
                logger.info(f"Uploaded {modality} to {storage_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to upload {modality}: {str(e)}",
                )

            # Record upload
            uploaded_modalities[modality] = FileUploadResponse(
                filename=file.filename,
                size_bytes=file_size,
                storage_path=storage_path,
                upload_timestamp=datetime.utcnow(),
            )

            scan.modalities.append(modality)

        # Save scan to database
        scan.storage_location = f"s3://brain-tumor-dicom/scans/{patient.mrn}/scan_{scan.id}"
        db.add(scan)
        db.commit()
        db.refresh(scan)

        logger.info(f"Created scan {scan.id} with modalities {scan.modalities}")

        return MRIScanUploadResponse(
            scan_id=scan.id,
            patient_id=patient.id,
            uploaded_modalities=uploaded_modalities,
            total_size_mb=total_size / (1024 * 1024),
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


# Inference Endpoints
@router.post("/inference/start", response_model=InferenceJobResponse)
async def start_inference(
    request: InferenceJobCreate,
    db: Session = Depends(get_db),
) -> InferenceJobResponse:
    """
    Start segmentation and classification inference.

    Triggers asynchronous Celery task.

    Args:
        request: Inference job creation request
        db: Database session

    Returns:
        InferenceJobResponse: Job metadata

    Raises:
        HTTPException: If scan not found or task fails
    """
    # Verify scan exists
    scan = db.query(MRIScan).filter(MRIScan.id == request.scan_id).first()
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan {request.scan_id} not found",
        )

    # Create job record
    job = InferenceJob(
        scan_id=scan.id,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Prepare file paths
    modality_paths = {
        f"modality_{i}": f"scans/{scan.patient.mrn}/scan_{scan.id}_{mod}_*.nii.gz"
        for i, mod in enumerate(scan.modalities)
    }

    # Trigger Celery task
    try:
        celery_task = run_tumor_inference.apply_async(
            args=[str(job.id), str(scan.id), modality_paths],
            task_id=str(job.id),
        )
        job.celery_task_id = celery_task.id
        db.commit()

        logger.info(f"Started inference task {celery_task.id} for job {job.id}")
    except Exception as e:
        logger.error(f"Failed to submit task: {str(e)}")
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start inference: {str(e)}",
        )

    return InferenceJobResponse.from_orm(job)


# Simple Image Analysis Endpoint (PNG/JPG)
@router.post("/analyze", response_model=ImageAnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ImageAnalyzeResponse:
    """
    Upload a brain MRI image (PNG/JPG) and start AI analysis.

    This is the simplified endpoint - upload one image, get analysis back.
    """
    storage = get_storage_backend()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    # Check extension
    filename_lower = file.filename.lower()
    is_regular_image = filename_lower.endswith(('.png', '.jpg', '.jpeg'))
    is_medical = filename_lower.endswith(('.nii', '.nii.gz', '.dcm'))

    if not is_regular_image and not is_medical:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file format. Please upload PNG, JPG, JPEG, DICOM, or NIfTI files.",
        )

    # Read content
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"File size exceeds {settings.max_file_size_mb}MB limit",
        )

    # Validate
    try:
        if is_regular_image:
            metadata = RegularImageValidator.validate(file_content)
        elif filename_lower.endswith(('.nii', '.nii.gz')):
            metadata = NIfTIValidator.validate(file_content)
        else:
            metadata = DICOMValidator.validate(file_content)
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image validation failed: {str(e)}",
        )

    # Get or create demo patient
    patient = db.query(Patient).filter(Patient.mrn == "DEMO-001").first()
    if not patient:
        patient = Patient(
            mrn="DEMO-001",
            date_of_birth=datetime(1990, 1, 1),
            sex="Unknown",
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Create scan record
    scan = MRIScan(
        patient_id=patient.id,
        scan_date=datetime.utcnow(),
        modalities=["image"],
        storage_location="",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # Upload to storage
    storage_path = f"scans/{patient.mrn}/scan_{scan.id}_{file.filename}"
    storage.upload_file(storage_path, file_content)
    scan.storage_location = storage_path
    db.commit()

    # Create inference job
    job = InferenceJob(
        scan_id=scan.id,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start Celery task
    try:
        celery_task = run_tumor_inference.apply_async(
            args=[str(job.id), str(scan.id), {
                "image_path": storage_path,
                "filename": file.filename,
                "is_regular_image": is_regular_image,
                "metadata": metadata,
            }],
            task_id=str(job.id),
        )
        job.celery_task_id = celery_task.id
        db.commit()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}",
        )

    return ImageAnalyzeResponse(
        job_id=job.id,
        scan_id=scan.id,
        filename=file.filename,
        status="processing",
        message="Image uploaded successfully. Analysis started.",
    )


@router.get("/inference/{job_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(
    job_id: str,
    db: Session = Depends(get_db),
) -> AnalysisResultResponse:
    """
    Get the full analysis results for a completed inference job.

    Returns human-readable explanation and findings.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format",
        )

    job = db.query(InferenceJob).filter(InferenceJob.id == job_uuid).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.status == JobStatus.PROCESSING or job.status == JobStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Analysis is still in progress",
        )

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {job.error_message}",
        )

    # Get classification results
    from app.models import ClassificationResult, SegmentationResult
    clf_results = db.query(ClassificationResult).filter(
        ClassificationResult.job_id == job_uuid
    ).all()
    seg_results = db.query(SegmentationResult).filter(
        SegmentationResult.job_id == job_uuid
    ).all()

    # Get filename from scan storage path
    scan = db.query(MRIScan).filter(MRIScan.id == job.scan_id).first()
    filename = scan.storage_location.split("/")[-1] if scan else "unknown"

    # Build analysis response
    tumor_detected = len(clf_results) > 0
    tumor_grade = clf_results[0].tumor_grade if clf_results else None
    confidence = clf_results[0].confidence_score if clf_results else 0.0
    clf_details = clf_results[0].classification_details if clf_results else None

    # Build findings list
    findings = []
    if tumor_detected:
        findings.append(f"Tumor detected with {confidence:.1%} confidence")
        if tumor_grade:
            findings.append(f"Classified as {tumor_grade}")
        for seg in seg_results:
            findings.append(
                f"{seg.subregion.replace('_', ' ').title()}: "
                f"confidence {seg.confidence_score:.1%}, "
                f"volume {seg.volume_mm3:.1f} mm³"
            )
    else:
        findings.append("No significant abnormalities detected")

    # Build explanation
    explanation = _build_explanation(tumor_detected, tumor_grade, confidence, seg_results, clf_details)

    # Build recommendations
    recommendations = _build_recommendations(tumor_detected, tumor_grade, confidence)

    # Segmentation summary
    seg_summary = {}
    for seg in seg_results:
        seg_summary[seg.subregion] = {
            "confidence": seg.confidence_score,
            "volume_mm3": seg.volume_mm3,
        }

    return AnalysisResultResponse(
        job_id=job_uuid,
        status="completed",
        image_filename=filename,
        tumor_detected=tumor_detected,
        tumor_type="Glioma" if tumor_detected else None,
        tumor_grade=tumor_grade,
        confidence=confidence,
        findings=findings,
        explanation=explanation,
        recommendations=recommendations,
        classification_details=clf_details,
        segmentation_summary=seg_summary if seg_summary else None,
        completed_at=job.completed_at,
    )


def _build_explanation(tumor_detected, tumor_grade, confidence, seg_results, clf_details):
    """Build a human-readable analysis explanation."""
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

    parts = [
        f"The AI analysis has detected {grade_desc} with a confidence level of {confidence:.1%}.",
    ]

    # Add segmentation details
    for seg in seg_results:
        region = seg.subregion.replace("_", " ")
        if region == "enhancing tumor":
            parts.append(
                f"The enhancing tumor region (actively growing area) was identified "
                f"with {seg.confidence_score:.1%} confidence, measuring approximately "
                f"{seg.volume_mm3:.1f} mm³."
            )
        elif region == "edema":
            parts.append(
                f"Peritumoral edema (swelling around the tumor) was detected "
                f"with {seg.confidence_score:.1%} confidence, measuring approximately "
                f"{seg.volume_mm3:.1f} mm³."
            )
        elif region == "necrotic core":
            parts.append(
                f"A necrotic core (dead tissue within the tumor) was identified "
                f"with {seg.confidence_score:.1%} confidence, measuring approximately "
                f"{seg.volume_mm3:.1f} mm³."
            )

    # Add probability breakdown
    if clf_details and "probabilities" in clf_details:
        probs = clf_details["probabilities"]
        parts.append("Classification probability breakdown: " + ", ".join(
            f"{k}: {v:.1%}" for k, v in probs.items()
        ))

    parts.append(
        "IMPORTANT: This is an AI-assisted analysis for educational/research purposes. "
        "Always consult a qualified medical professional for diagnosis and treatment decisions."
    )

    return " ".join(parts)


def _build_recommendations(tumor_detected, tumor_grade, confidence):
    """Build recommendations based on analysis."""
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

    if confidence < 0.7:
        recs.append("Low confidence result - additional imaging may be needed")

    return recs


@router.get("/inference/{job_id}", response_model=InferenceJobResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
) -> InferenceJobResponse:
    """
    Get inference job status.

    Args:
        job_id: Job UUID
        db: Database session

    Returns:
        InferenceJobResponse: Current job status

    Raises:
        HTTPException: If job not found
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format",
        )

    job = db.query(InferenceJob).filter(InferenceJob.id == job_uuid).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return InferenceJobResponse.from_orm(job)


# WebSocket for real-time updates
@router.websocket("/ws/job/{job_id}")
async def websocket_job_status(
    websocket: WebSocket,
    job_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    WebSocket endpoint for real-time job status updates.

    Args:
        websocket: WebSocket connection
        job_id: Job UUID to monitor
        db: Database session
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid job_id")
        return

    # Verify job exists
    job = db.query(InferenceJob).filter(InferenceJob.id == job_uuid).first()
    if not job:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Job not found")
        return

    await manager.connect(websocket, str(job_id))

    try:
        # Send initial status
        await websocket.send_json({
            "event": "status_update",
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress_percentage,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep connection alive and listen for disconnects
        while True:
            data = await websocket.receive_text()
            # Echo received message
            await websocket.send_json({
                "event": "echo",
                "data": data,
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket, str(job_id))
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, str(job_id))


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "brain-tumor-ai",
        "version": settings.app_version,
    }
