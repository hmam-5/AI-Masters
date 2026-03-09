"""
Configuration settings for the Brain Tumor AI Framework.

This module manages all environment-based and static configuration
for the FastAPI backend, including database, storage, and ML settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main settings configuration with validation."""

    # Application
    app_name: str = "Brain Tumor AI Framework"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = ["http://localhost", "http://localhost:3000", "http://localhost:8000"]
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql://brain_tumor_user:secure_password_change_in_prod@localhost:5432/brain_tumor_db"
    sqlalchemy_echo: bool = False

    # Redis & Celery
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Storage (MinIO/S3)
    storage_backend: Literal["minio", "s3"] = "minio"
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_dicom: str = "dicom-scans"
    minio_bucket_results: str = "segmentation-results"
    minio_use_ssl: bool = False

    aws_s3_bucket_dicom: str = "brain-tumor-dicom"
    aws_s3_bucket_results: str = "brain-tumor-results"
    aws_region: str = "us-east-1"

    # File Upload
    max_file_size_mb: int = 500
    allowed_extensions: list[str] = ["png", "jpg", "jpeg", "dcm", "nii", "nii.gz"]
    chunk_size_bytes: int = 1024 * 1024  # 1MB chunks

    # ML Model Configuration
    model_unet_path: str = "/models/unet_3d_segmentation.pth"
    model_resnet_path: str = "/models/resnet3d_classification.pth"
    model_2d_path: str = "/models/brain_tumor_2d.pth"
    image_size: int = 224
    device: str = "cuda"  # 'cuda' or 'cpu'
    inference_batch_size: int = 1
    num_inference_workers: int = 4

    # Preprocessing
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normalization_scheme: str = "zscore"  # 'zscore' or 'minmax'

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Task Configuration
    celery_task_time_limit: int = 3600  # 1 hour
    celery_task_soft_time_limit: int = 3300  # 55 minutes

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings singleton.

    Benefits:
        - Caches settings to avoid reparsing .env file
        - Only instantiates once per application lifecycle
    """
    return Settings()
