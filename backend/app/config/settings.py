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
    app_name: str = "AI Masters"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = ["http://localhost", "http://localhost:3000", "http://localhost:8000"]
    api_prefix: str = "/api/v1"

    # Redis & Celery
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # FalkorDB Graph Database
    falkordb_host: str = "falkordb"
    falkordb_port: int = 6379

    # Dataset
    data_dir: str = "/data"
    combined_dataset_dir: str = "/data/combined"

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
    model_2d_path: str = "/models/brain_tumor_2d.pth"
    model_resnet50_path: str = "/models/brain_tumor_resnet50.pth"
    model_efficientnet_path: str = "/models/brain_tumor_efficientnet.pth"
    model_densenet_path: str = "/models/brain_tumor_densenet.pth"
    image_size: int = 224
    device: str = "cuda"  # 'cuda' or 'cpu'
    inference_batch_size: int = 1
    num_inference_workers: int = 4
    min_confidence_for_auto_decision: float = 0.99

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Rate Limiting
    rate_limit_default: str = "200/minute"
    rate_limit_analyze: str = "10/minute"

    # Observability: Jaeger distributed tracing
    jaeger_host: str = "jaeger"
    jaeger_port: int = 6831
    tracing_enabled: bool = True

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
        protected_namespaces = ("settings_",)


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
