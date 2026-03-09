"""
Storage service for DICOM/NIfTI file management.

Supports MinIO and AWS S3 backends with encryption and versioning.
"""

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import boto3
from minio import Minio
from minio.error import S3Error

from app.config import get_settings

settings = get_settings()


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload_file(
        self,
        file_path: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload file to storage backend."""

    @abstractmethod
    def download_file(self, file_path: str) -> bytes:
        """Download file from storage backend."""

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        """Delete file from storage backend."""

    @abstractmethod
    def list_files(self, prefix: str) -> list[str]:
        """List files by prefix."""


class MinIOBackend(StorageBackend):
    """MinIO S3-compatible storage backend."""

    def __init__(self) -> None:
        """Initialize MinIO client."""
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_use_ssl,
        )

    def upload_file(
        self,
        file_path: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload file to MinIO.

        Args:
            file_path: Path for file in bucket (e.g., 'scans/patient123/t1.nii.gz')
            file_content: File bytes
            content_type: MIME type

        Returns:
            str: Object path in MinIO

        Raises:
            S3Error: If upload fails
        """
        # Determine bucket based on file type
        bucket = (
            settings.minio_bucket_dicom
            if file_path.endswith(".dcm")
            else settings.minio_bucket_dicom
        )

        try:
            self.client.put_object(
                bucket_name=bucket,
                object_name=file_path,
                data=io.BytesIO(file_content),
                length=len(file_content),
                content_type=content_type,
            )
            return f"s3://{bucket}/{file_path}"
        except S3Error as e:
            raise RuntimeError(f"MinIO upload failed: {str(e)}")

    def download_file(self, file_path: str) -> bytes:
        """
        Download file from MinIO.

        Args:
            file_path: Path in format 's3://bucket/object' or just 'object'

        Returns:
            bytes: File content

        Raises:
            S3Error: If download fails
        """
        # Parse path
        if file_path.startswith("s3://"):
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            obj_name = parts[1]
        else:
            bucket = settings.minio_bucket_dicom
            obj_name = file_path

        try:
            response = self.client.get_object(bucket_name=bucket, object_name=obj_name)
            return response.read()
        except S3Error as e:
            raise RuntimeError(f"MinIO download failed: {str(e)}")

    def delete_file(self, file_path: str) -> None:
        """Delete file from MinIO."""
        if file_path.startswith("s3://"):
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            obj_name = parts[1]
        else:
            bucket = settings.minio_bucket_dicom
            obj_name = file_path

        try:
            self.client.remove_object(bucket_name=bucket, object_name=obj_name)
        except S3Error as e:
            raise RuntimeError(f"MinIO deletion failed: {str(e)}")

    def list_files(self, prefix: str) -> list[str]:
        """List files in MinIO bucket by prefix."""
        bucket = settings.minio_bucket_dicom
        try:
            objects = self.client.list_objects(bucket_name=bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            raise RuntimeError(f"MinIO list failed: {str(e)}")


class S3Backend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self) -> None:
        """Initialize S3 client."""
        self.s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region,
        )

    def upload_file(
        self,
        file_path: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload file to S3.

        Args:
            file_path: Path for file in bucket
            file_content: File bytes
            content_type: MIME type

        Returns:
            str: S3 object URI
        """
        bucket = settings.aws_s3_bucket_dicom
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=file_path,
                Body=file_content,
                ContentType=content_type,
            )
            return f"s3://{bucket}/{file_path}"
        except Exception as e:
            raise RuntimeError(f"S3 upload failed: {str(e)}")

    def download_file(self, file_path: str) -> bytes:
        """Download file from S3."""
        if file_path.startswith("s3://"):
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = settings.aws_s3_bucket_dicom
            key = file_path

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            raise RuntimeError(f"S3 download failed: {str(e)}")

    def delete_file(self, file_path: str) -> None:
        """Delete file from S3."""
        if file_path.startswith("s3://"):
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = settings.aws_s3_bucket_dicom
            key = file_path

        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            raise RuntimeError(f"S3 deletion failed: {str(e)}")

    def list_files(self, prefix: str) -> list[str]:
        """List files in S3 bucket by prefix."""
        bucket = settings.aws_s3_bucket_dicom
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            files = []
            for page in pages:
                if "Contents" in page:
                    files.extend([obj["Key"] for obj in page["Contents"]])
            return files
        except Exception as e:
            raise RuntimeError(f"S3 list failed: {str(e)}")


def get_storage_backend() -> StorageBackend:
    """
    Factory function to get configured storage backend.

    Returns:
        StorageBackend: MinIO or S3 backend based on configuration

    Raises:
        ValueError: If invalid storage backend is configured
    """
    if settings.storage_backend == "minio":
        return MinIOBackend()
    elif settings.storage_backend == "s3":
        return S3Backend()
    else:
        raise ValueError(f"Unknown storage backend: {settings.storage_backend}")
