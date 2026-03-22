"""
Pydantic schemas for API request/response validation.

Defines data models for type safety and OpenAPI documentation.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Patient Schemas
class PatientBase(BaseModel):
    """Base patient schema."""

    mrn: str = Field(..., description="Medical Record Number")
    sex: Optional[str] = Field(None, description="Patient sex (M/F)")


class PatientCreate(PatientBase):
    """Patient creation schema."""

    date_of_birth: datetime = Field(..., description="Patient date of birth")


class PatientResponse(PatientBase):
    """Patient response schema."""

    id: UUID
    date_of_birth: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# MRI Scan Schemas
class MRIScanBase(BaseModel):
    """Base MRI scan schema."""

    modalities: list[str] = Field(
        ...,
        description="List of imaging modalities (T1, T1ce, T2, FLAIR)",
    )


class MRIScanCreate(MRIScanBase):
    """MRI scan creation schema."""

    patient_id: UUID = Field(..., description="Patient ID")
    scan_date: datetime = Field(default_factory=datetime.utcnow)


class MRIScanResponse(MRIScanBase):
    """MRI scan response schema."""

    id: UUID
    patient_id: UUID
    scan_date: datetime
    status: str
    preprocessing_complete: bool
    image_shape: Optional[list[int]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Inference Job Schemas
class InferenceJobResponse(BaseModel):
    """Inference job response schema."""

    id: UUID
    scan_id: UUID
    celery_task_id: Optional[str] = None
    status: str = Field(..., description="Job status (pending, processing, completed, failed)")
    progress_percentage: int = Field(0, ge=0, le=100)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


class InferenceJobCreate(BaseModel):
    """Inference job creation schema."""

    scan_id: UUID = Field(..., description="MRI scan ID to process")


# Analysis Result Schema (for regular image uploads)
class AnalysisResultResponse(BaseModel):
    """Full analysis result with human-readable explanation."""

    job_id: UUID
    status: str
    image_filename: str
    tumor_detected: bool
    tumor_type: Optional[str] = None
    tumor_grade: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    findings: list[str] = Field(default_factory=list)
    explanation: str = Field(..., description="Human-readable analysis explanation")
    recommendations: list[str] = Field(default_factory=list)
    classification_details: Optional[dict] = None
    segmentation_summary: Optional[dict] = None
    completed_at: Optional[datetime] = None


class ImageAnalyzeResponse(BaseModel):
    """Response after uploading an image for analysis."""

    job_id: UUID
    scan_id: UUID
    filename: str
    status: str
    message: str


# Segmentation Result Schemas
class SegmentationResultResponse(BaseModel):
    """Segmentation result response schema."""

    id: UUID
    job_id: UUID
    subregion: str = Field(
        ...,
        description="Tumor subregion (enhancing_tumor, edema, necrotic_core)",
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    volume_mm3: Optional[float] = None
    mask_storage_path: str
    dice_coefficient: Optional[float] = None
    hausdorff_distance: Optional[float] = None
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Classification Result Schemas
class ClassificationResultResponse(BaseModel):
    """Classification result response schema."""

    id: UUID
    job_id: UUID
    tumor_grade: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    classification_details: Optional[dict] = None
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Upload Schema
class FileUploadResponse(BaseModel):
    """File upload response."""

    filename: str
    size_bytes: int
    storage_path: str
    upload_timestamp: datetime


class MRIScanUploadResponse(BaseModel):
    """Multi-modal MRI scan upload response."""

    scan_id: UUID
    patient_id: UUID
    uploaded_modalities: dict[str, FileUploadResponse]
    total_size_mb: float
    timestamp: datetime


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema."""

    error_code: str
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Doctor Schemas (M:N with Patient)
class DoctorBase(BaseModel):
    """Base doctor schema."""

    name: str = Field(..., description="Doctor's full name")
    specialization: str = Field(..., description="Medical specialization (e.g., Neuro-oncology)")
    license_number: str = Field(..., description="Medical license number")
    email: str = Field(..., description="Contact email")


class DoctorCreate(DoctorBase):
    """Doctor creation schema."""
    pass


class DoctorResponse(DoctorBase):
    """Doctor response schema."""

    id: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DoctorAssignmentResponse(BaseModel):
    """Response for doctor-patient or doctor-job assignment."""

    status: str
    doctor_id: str
    patient_mrn: Optional[str] = None
    job_id: Optional[str] = None


# Tag Schema (M:N with Scan)
class TagResponse(BaseModel):
    """Tag response schema."""

    name: str = Field(..., description="Tag name (e.g., urgent, second-opinion)")


class ScanTagResponse(BaseModel):
    """Scan tag assignment response."""

    scan_id: str
    tag: str
    status: str


# Audit Log Schema
class AuditLogResponse(BaseModel):
    """Audit log entry response schema."""

    id: str
    action: str = Field(..., description="Action performed (CREATE, UPDATE, DELETE, ASSIGN)")
    entity_type: str = Field(..., description="Type of entity affected")
    entity_id: str
    actor: str = Field(..., description="Who performed the action")
    timestamp: datetime
    details: Optional[str] = None


# Model Version Schema
class ModelVersionResponse(BaseModel):
    """Model version response schema."""

    id: str
    model_name: str
    version: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    path: str
    status: str = Field(..., description="active or superseded")
    created_at: Optional[datetime] = None
