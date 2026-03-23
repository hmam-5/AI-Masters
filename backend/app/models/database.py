"""
Data models for the Brain Tumor AI Framework.

Plain Python enums — all persistence is handled by FalkorDB.
"""

import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
