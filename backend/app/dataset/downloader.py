"""
Dataset downloader for brain tumor classification.

Downloads and organizes multiple publicly available brain tumor MRI datasets
for training a high-accuracy classification model.

Supported datasets:
1. Br35H - Brain Tumor Detection (Kaggle) — ~3,000 images
2. Brain Tumor Classification (MRI) — ~7,000 images (4 classes)
3. Brain MRI Images for Tumor Detection — ~3,000 images

All datasets are downloaded to /data/ and organized into:
  /data/combined/
    train/
      glioma/ meningioma/ pituitary/ no_tumor/
    val/
      glioma/ meningioma/ pituitary/ no_tumor/
"""

import hashlib
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Dataset registry — public direct-download URLs
# These use Kaggle dataset references; the actual download requires kaggle API or manual step
DATASET_REGISTRY = {
    "brain_tumor_classification_mri": {
        "description": "Brain Tumor Classification (MRI) — 7,023 images in 4 classes",
        "classes": ["glioma", "meningioma", "pituitary", "no_tumor"],
        "kaggle_dataset": "sartajbhuvaji/brain-tumor-classification-mri",
        "expected_images": 7023,
    },
    "brain_tumor_detection_br35h": {
        "description": "Br35H Brain Tumor Detection 2020 — 3,060 images",
        "classes": ["tumor", "no_tumor"],
        "kaggle_dataset": "ahmedhamada0/brain-tumor-detection",
        "expected_images": 3060,
    },
    "brain_mri_tumor_detection": {
        "description": "Brain MRI Images for Brain Tumor Detection — 253 images",
        "classes": ["tumor", "no_tumor"],
        "kaggle_dataset": "navoneel/brain-mri-images-for-brain-tumor-detection",
        "expected_images": 253,
    },
}

# Grade mapping for the 4-class classification
CLASS_TO_GRADE = {
    "glioma": "Grade IV",
    "meningioma": "Grade II",
    "pituitary": "Grade III",
    "no_tumor": "No Tumor",
}


class DatasetDownloader:
    """Downloads and organizes brain tumor datasets for training."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.combined_dir = self.data_dir / "combined"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.combined_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(self, dataset_ref: str, target_dir: Path) -> bool:
        """
        Download a dataset from Kaggle using the kaggle API.

        Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {dataset_ref} from Kaggle...")
            api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=True)
            logger.info(f"Downloaded {dataset_ref} to {target_dir}")
            return True

        except ImportError:
            logger.warning(
                "kaggle package not installed. Install with: pip install kaggle"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to download {dataset_ref}: {e}")
            return False

    def download_all_datasets(self) -> dict:
        """
        Download all registered datasets.

        Returns:
            dict: Status of each download
        """
        results = {}
        for name, info in DATASET_REGISTRY.items():
            target = self.raw_dir / name
            if target.exists() and any(target.iterdir()):
                logger.info(f"Dataset {name} already exists, skipping download")
                results[name] = "exists"
                continue

            success = self.download_kaggle_dataset(info["kaggle_dataset"], target)
            results[name] = "downloaded" if success else "failed"

        return results

    def organize_combined_dataset(self, val_split: float = 0.15) -> dict:
        """
        Organize all downloaded datasets into a unified train/val structure.

        The primary dataset (brain_tumor_classification_mri) has 4 classes:
        glioma, meningioma, pituitary, no_tumor.

        Secondary datasets with binary labels (tumor/no_tumor) are mapped:
        - tumor images -> distributed across glioma/meningioma/pituitary
        - no_tumor images -> no_tumor class

        Returns:
            dict: Statistics about the combined dataset
        """
        import random

        random.seed(42)

        classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

        # Create directories
        for split in ["train", "val"]:
            for cls in classes:
                (self.combined_dir / split / cls).mkdir(parents=True, exist_ok=True)

        stats = {cls: {"train": 0, "val": 0} for cls in classes}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        # Collect all images per class
        class_images: dict[str, list[Path]] = {cls: [] for cls in classes}

        # 1. Process primary 4-class dataset
        primary_dir = self.raw_dir / "brain_tumor_classification_mri"
        if primary_dir.exists():
            self._collect_from_directory(primary_dir, class_images, image_extensions)

        # 2. Process secondary binary datasets
        for ds_name in ["brain_tumor_detection_br35h", "brain_mri_tumor_detection"]:
            ds_dir = self.raw_dir / ds_name
            if not ds_dir.exists():
                continue
            self._collect_binary_dataset(ds_dir, class_images, image_extensions)

        # 3. Split into train/val and copy files
        total_images = 0
        for cls, images in class_images.items():
            random.shuffle(images)
            val_count = max(1, int(len(images) * val_split))
            val_images = images[:val_count]
            train_images = images[val_count:]

            for img_path in train_images:
                dest = self.combined_dir / "train" / cls / f"{cls}_{stats[cls]['train']:06d}{img_path.suffix}"
                shutil.copy2(img_path, dest)
                stats[cls]["train"] += 1
                total_images += 1

            for img_path in val_images:
                dest = self.combined_dir / "val" / cls / f"{cls}_{stats[cls]['val']:06d}{img_path.suffix}"
                shutil.copy2(img_path, dest)
                stats[cls]["val"] += 1
                total_images += 1

        logger.info(f"Combined dataset organized: {total_images} total images")
        logger.info(f"Class distribution: {stats}")

        return {"total_images": total_images, "classes": stats}

    def _collect_from_directory(
        self,
        root_dir: Path,
        class_images: dict[str, list[Path]],
        extensions: set[str],
    ) -> None:
        """Recursively collect images from a 4-class directory structure."""
        class_mappings = {
            "glioma": "glioma",
            "glioma_tumor": "glioma",
            "meningioma": "meningioma",
            "meningioma_tumor": "meningioma",
            "pituitary": "pituitary",
            "pituitary_tumor": "pituitary",
            "no_tumor": "no_tumor",
            "notumor": "no_tumor",
            "no": "no_tumor",
            "healthy": "no_tumor",
            "normal": "no_tumor",
        }

        for path in root_dir.rglob("*"):
            if path.suffix.lower() not in extensions:
                continue
            # Determine class from parent directory name
            parent_name = path.parent.name.lower().replace(" ", "_")
            mapped_class = class_mappings.get(parent_name)
            if mapped_class and mapped_class in class_images:
                class_images[mapped_class].append(path)

    def _collect_binary_dataset(
        self,
        root_dir: Path,
        class_images: dict[str, list[Path]],
        extensions: set[str],
    ) -> None:
        """Collect images from binary (tumor/no_tumor) datasets."""
        tumor_keywords = {"yes", "tumor", "positive", "y"}
        no_tumor_keywords = {"no", "no_tumor", "negative", "n", "notumor", "healthy"}
        tumor_classes = ["glioma", "meningioma", "pituitary"]

        idx = 0
        for path in root_dir.rglob("*"):
            if path.suffix.lower() not in extensions:
                continue

            parent_name = path.parent.name.lower().replace(" ", "_")

            if parent_name in no_tumor_keywords:
                class_images["no_tumor"].append(path)
            elif parent_name in tumor_keywords:
                # Distribute across tumor classes round-robin
                target_class = tumor_classes[idx % len(tumor_classes)]
                class_images[target_class].append(path)
                idx += 1

    def get_dataset_stats(self) -> dict:
        """Get statistics about the combined dataset."""
        stats = {}
        for split in ["train", "val"]:
            split_dir = self.combined_dir / split
            if not split_dir.exists():
                continue
            stats[split] = {}
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    count = len([
                        f for f in cls_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    ])
                    stats[split][cls_dir.name] = count
        return stats

    def prepare_dataset(self) -> dict:
        """
        Full pipeline: download all datasets and organize them.

        Returns:
            dict: Combined statistics
        """
        logger.info("Starting dataset preparation...")
        download_results = self.download_all_datasets()
        logger.info(f"Download results: {download_results}")

        organize_results = self.organize_combined_dataset()
        logger.info(f"Organization results: {organize_results}")

        return {
            "downloads": download_results,
            "combined": organize_results,
        }
