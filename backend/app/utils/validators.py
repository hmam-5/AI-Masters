"""
Image validation and preprocessing for medical and regular images.

Handles corrupted file detection, format validation, and metadata extraction.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError


class ImageValidationError(Exception):
    """Raised when image validation fails."""

    pass


class DICOMValidator:
    """Validation for DICOM files."""

    @staticmethod
    def validate(file_content: bytes, modality: Optional[str] = None) -> dict:
        """
        Validate DICOM file and extract metadata.

        Args:
            file_content: DICOM file bytes
            modality: Expected modality (T1, T1ce, T2, FLAIR)

        Returns:
            dict: Metadata including shape, modality, manufacturer

        Raises:
            ImageValidationError: If file is corrupted or invalid
        """
        try:
            # Parse DICOM
            dataset = pydicom.dcmread(file_content, force=True)

            # Validate essential elements
            if not hasattr(dataset, "pixel_array"):
                raise ImageValidationError("DICOM file has no pixel data")

            pixel_array = dataset.pixel_array
            if pixel_array.size == 0:
                raise ImageValidationError("DICOM pixel array is empty")

            # Check modality if provided
            actual_modality = getattr(dataset, "Modality", "MR")
            if modality and modality not in actual_modality:
                raise ImageValidationError(
                    f"Modality mismatch: expected {modality}, got {actual_modality}"
                )

            # Extract metadata
            metadata = {
                "shape": pixel_array.shape,
                "dtype": str(pixel_array.dtype),
                "modality": actual_modality,
                "patient_id": getattr(dataset, "PatientID", "Unknown"),
                "study_date": getattr(dataset, "StudyDate", None),
                "pixel_spacing": getattr(dataset, "PixelSpacing", None),
            }

            return metadata
        except InvalidDicomError as e:
            raise ImageValidationError(f"Invalid DICOM file: {str(e)}")
        except Exception as e:
            raise ImageValidationError(f"DICOM validation failed: {str(e)}")


class NIfTIValidator:
    """Validation for NIfTI files."""

    @staticmethod
    def validate(file_content: bytes) -> dict:
        """
        Validate NIfTI file and extract metadata.

        Args:
            file_content: NIfTI file bytes (.nii or .nii.gz)

        Returns:
            dict: Metadata including shape, affine, datatype

        Raises:
            ImageValidationError: If file is corrupted or invalid
        """
        try:
            # Load NIfTI from bytes via temp file
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
                f.write(file_content)
                tmp_path = f.name
            img = nib.load(tmp_path)
            data = img.get_fdata()

            if data.size == 0:
                raise ImageValidationError("NIfTI image data is empty")

            # 3D validation
            if data.ndim != 3:
                raise ImageValidationError(
                    f"Expected 3D image, got {data.ndim}D"
                )

            # Check for NaN values
            nan_count = np.isnan(data).sum()
            if nan_count > data.size * 0.5:
                raise ImageValidationError(
                    f"NIfTI contains {nan_count} NaN values (>{50}% of image)"
                )

            # Extract metadata
            header = img.header
            metadata = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "affine": img.affine.tolist(),
                "voxel_spacing": header.get_zooms()[:3],
                "data_type": str(header.get_data_dtype()),
                "nan_count": int(nan_count),
            }

            return metadata
        except Exception as e:
            raise ImageValidationError(f"NIfTI validation failed: {str(e)}")

    @staticmethod
    def standardize_4d_to_3d(file_content: bytes) -> bytes:
        """
        Convert 4D NIfTI (e.g., time series) to 3D by taking first volume.

        Args:
            file_content: 4D NIfTI file bytes

        Returns:
            bytes: 3D NIfTI file bytes
        """
        try:
            img = nib.load(nib.FileHolder.from_bytes(file_content))
            data = img.get_fdata()

            if data.ndim == 4:
                # Take first volume
                data_3d = data[..., 0]
                img_3d = nib.Nifti1Image(data_3d, affine=img.affine, header=img.header)
                output = nib.FileHolder()
                nib.save(img_3d, output)
                return output.getbuffer().tobytes()
            else:
                return file_content
        except Exception as e:
            raise ImageValidationError(f"4D conversion failed: {str(e)}")


class RegularImageValidator:
    """Validation for regular image files (PNG, JPG, JPEG)."""

    @staticmethod
    def validate(file_content: bytes) -> dict:
        """
        Validate a regular image file and extract metadata.

        Args:
            file_content: Image file bytes

        Returns:
            dict: Metadata including shape, format, mode

        Raises:
            ImageValidationError: If file is corrupted or invalid
        """
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify image integrity

            # Re-open after verify (verify can close the file)
            img = Image.open(io.BytesIO(file_content))
            width, height = img.size

            if width < 10 or height < 10:
                raise ImageValidationError(
                    f"Image too small: {width}x{height} (minimum 10x10)"
                )

            if width > 10000 or height > 10000:
                raise ImageValidationError(
                    f"Image too large: {width}x{height} (maximum 10000x10000)"
                )

            metadata = {
                "shape": (height, width),
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(file_content),
            }

            return metadata
        except ImageValidationError:
            raise
        except Exception as e:
            raise ImageValidationError(f"Image validation failed: {str(e)}")


class MultimodalValidator:
    """Validation for multi-modal image sets."""

    @staticmethod
    def validate_modality_set(
        modalities: dict[str, bytes],
    ) -> Tuple[bool, str, dict]:
        """
        Validate that all modalities have consistent shapes.

        Args:
            modalities: Dictionary of {modality: file_bytes} e.g., {'T1': bytes, 'T2': bytes}

        Returns:
            Tuple of (valid: bool, error_msg: str, shapes: dict)
        """
        shapes = {}

        for modality, file_content in modalities.items():
            try:
                validator = NIfTIValidator if modality.endswith(".nii") else DICOMValidator
                metadata = validator.validate(file_content)
                shapes[modality] = metadata["shape"]
            except ImageValidationError as e:
                return False, f"{modality} validation failed: {str(e)}", {}

        # Check shape consistency
        unique_shapes = set(str(s) for s in shapes.values())
        if len(unique_shapes) > 1:
            return (
                False,
                f"Inconsistent shapes across modalities: {shapes}",
                {},
            )

        return True, "", shapes
