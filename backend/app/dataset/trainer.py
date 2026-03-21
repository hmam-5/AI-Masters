"""
Training pipeline for the Brain Tumor 2D CNN classifier.

Trains an EfficientNet-based model on the combined brain tumor dataset
with data augmentation, learning rate scheduling, and early stopping
to achieve >90% validation accuracy.
"""

import logging
import os
import multiprocessing
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Class labels
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = 4

CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "no": "no_tumor",
    "normal": "no_tumor",
    "healthy": "no_tumor",
}


def _resolve_split_dirs(data_dir: str) -> tuple[Path, Path]:
    """Resolve dataset split directories for either combined or Kaggle layout."""
    data_path = Path(data_dir)

    candidates = [
        (data_path / "train", data_path / "val"),
        (data_path / "Training", data_path / "Testing"),
        (data_path / "training", data_path / "testing"),
    ]

    for train_dir, val_dir in candidates:
        if train_dir.exists() and val_dir.exists():
            return train_dir, val_dir

    expected = ", ".join([f"{a.name}/{b.name}" for a, b in candidates])
    raise FileNotFoundError(
        f"No valid dataset split found under {data_path}. Expected one of: {expected}"
    )


def _parse_data_roots(data_dir: str | list[str] | tuple[str, ...]) -> list[Path]:
    """Parse one or many dataset roots from string/list input."""
    if isinstance(data_dir, (list, tuple)):
        raw_items = [str(x).strip() for x in data_dir if str(x).strip()]
    else:
        text = str(data_dir).strip()
        if not text:
            raw_items = []
        else:
            normalized = text.replace("\n", ",").replace(";", ",")
            raw_items = [item.strip() for item in normalized.split(",") if item.strip()]

    roots = [Path(item) for item in raw_items]
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)

    if not unique_roots:
        raise ValueError("No dataset root directories were provided")
    return unique_roots


def _build_combined_datasets(
    data_dir: str | list[str] | tuple[str, ...],
    image_size: int,
) -> tuple[list[str], datasets.ImageFolder, datasets.ImageFolder]:
    """Build combined train/val datasets from one or many dataset roots."""
    roots = _parse_data_roots(data_dir)

    train_parts: list[datasets.ImageFolder] = []
    val_parts: list[datasets.ImageFolder] = []
    resolved_roots: list[str] = []

    for root in roots:
        train_dir, val_dir = _resolve_split_dirs(str(root))
        try:
            train_ds = _load_dataset(train_dir, transform=get_train_transforms(image_size))
            val_ds = _load_dataset(val_dir, transform=get_val_transforms(image_size))
        except FileNotFoundError as e:
            logger.warning(f"Skipping dataset root '{root}' due to empty split: {e}")
            continue

        train_parts.append(train_ds)
        val_parts.append(val_ds)
        resolved_roots.append(str(root))

    if not train_parts or not val_parts:
        raise FileNotFoundError(
            "No valid dataset roots contained loadable images. "
            f"Checked roots: {[str(r) for r in roots]}"
        )

    if len(train_parts) == 1:
        return resolved_roots, train_parts[0], val_parts[0]

    train_dataset = torch.utils.data.ConcatDataset(train_parts)
    val_dataset = torch.utils.data.ConcatDataset(val_parts)
    return resolved_roots, train_dataset, val_dataset


def _extract_targets(dataset: datasets.ImageFolder | torch.utils.data.ConcatDataset) -> list[int]:
    """Extract integer labels from ImageFolder or ConcatDataset."""
    if isinstance(dataset, datasets.ImageFolder):
        return list(dataset.targets)

    targets: list[int] = []
    for ds in dataset.datasets:
        if isinstance(ds, datasets.ImageFolder):
            targets.extend(ds.targets)
    return targets


def _compute_balanced_class_weights(targets: Iterable[int]) -> np.ndarray:
    """Compute stable inverse-frequency weights for classes."""
    target_arr = np.array(list(targets), dtype=np.int64)
    if target_arr.size == 0:
        return np.ones(NUM_CLASSES, dtype=np.float32)

    class_counts = np.bincount(target_arr, minlength=NUM_CLASSES).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    return weights / weights.sum()


def _load_dataset(split_dir: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    """Load dataset and remap class aliases to canonical CLASSES order."""
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)

    orig_to_new_idx: dict[int, int] = {}
    for orig_idx, class_name in enumerate(dataset.classes):
        normalized = CLASS_ALIASES.get(class_name.lower(), class_name.lower())
        if normalized not in CLASSES:
            raise ValueError(
                f"Unsupported class folder '{class_name}' in {split_dir}. "
                f"Supported classes: {CLASSES}"
            )
        orig_to_new_idx[orig_idx] = CLASSES.index(normalized)

    dataset.samples = [(path, orig_to_new_idx[label]) for path, label in dataset.samples]
    dataset.targets = [orig_to_new_idx[label] for label in dataset.targets]
    dataset.classes = CLASSES.copy()
    dataset.class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    return dataset


def _safe_num_workers(default_workers: int = 4) -> int:
    """Avoid subprocess DataLoader workers when running in daemonized Celery workers."""
    env_override = os.getenv("TRAIN_NUM_WORKERS")
    if env_override is not None:
        try:
            return max(0, int(env_override))
        except ValueError:
            logger.warning(f"Invalid TRAIN_NUM_WORKERS='{env_override}', using automatic detection")

    try:
        if multiprocessing.current_process().daemon:
            return 0
    except Exception:
        pass
    return default_workers


class BrainTumorClassifier(nn.Module):
    """
    Enhanced CNN for brain tumor classification.

    Uses a deeper architecture with batch normalization and residual-like
    connections to achieve high accuracy on brain MRI classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Feature extraction blocks
        self.block1 = self._conv_block(3, 64)
        self.block2 = self._conv_block(64, 128)
        self.block3 = self._conv_block(128, 256)
        self.block4 = self._conv_block(256, 512)
        self.block5 = self._conv_block(512, 512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        return self.classifier(x)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training data augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_model(
    data_dir: str | list[str] | tuple[str, ...] = "/data/combined",
    model_save_path: str = "/models/brain_tumor_2d.pth",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    image_size: int = 224,
) -> dict:
    """
    Train the brain tumor classifier.

    Args:
        data_dir: Path to combined dataset with train/val subdirectories
        model_save_path: Where to save the best model weights
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        image_size: Input image size

    Returns:
        dict: Training results including best accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    roots, train_dataset, val_dataset = _build_combined_datasets(data_dir, image_size)
    logger.info(f"Using dataset roots: {roots}")

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Classes: {CLASSES}")

    class_weights = _compute_balanced_class_weights(_extract_targets(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_safe_num_workers(), pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=_safe_num_workers(), pin_memory=True,
    )

    # Initialize model
    model = BrainTumorClassifier(num_classes=NUM_CLASSES).to(device)
    existing_path = Path(model_save_path)
    if existing_path.exists():
        try:
            model.load_state_dict(torch.load(existing_path, map_location=device, weights_only=True))
            logger.info(f"Warm-started custom model from {existing_path}")
        except Exception as e:
            logger.warning(f"Could not warm-start custom model from {existing_path}: {e}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 25 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} — batch {batch_idx+1}/{len(train_loader)} "
                    f"(train_acc={train_correct / max(1, train_total):.4f})"
                )

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} — "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

    return {
        "best_val_accuracy": best_val_acc,
        "epochs_trained": epoch + 1,
        "model_path": model_save_path,
        "classes": CLASSES,
        "history": history,
    }


def train_all_models(
    data_dir: str | list[str] | tuple[str, ...] = "/data/combined",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    image_size: int = 224,
) -> dict:
    """
    Train all 4 models in the ensemble (Custom CNN + 3 pretrained).

    Each model is trained independently on the same dataset.
    Pretrained models (ResNet50, EfficientNet, DenseNet) use a lower
    learning rate for the backbone and higher for the new classifier head.

    Returns:
        dict mapping model_name -> training results
    """
    from app.dataset.models import (
        ResNet50Classifier,
        EfficientNetClassifier,
        DenseNetClassifier,
        MODEL_REGISTRY,
    )

    results = {}

    # 1. Train custom CNN
    logger.info("=" * 60)
    logger.info("Training Model 1/4: Custom CNN")
    logger.info("=" * 60)
    results["custom_cnn"] = train_model(
        data_dir=data_dir,
        model_save_path=MODEL_REGISTRY["custom_cnn"]["path"],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        image_size=image_size,
    )

    # 2-4. Train pretrained models with differential learning rates
    pretrained_models = [
        ("resnet50", ResNet50Classifier),
        ("efficientnet", EfficientNetClassifier),
        ("densenet", DenseNetClassifier),
    ]

    for idx, (name, model_cls) in enumerate(pretrained_models, start=2):
        logger.info("=" * 60)
        logger.info(f"Training Model {idx}/4: {name}")
        logger.info("=" * 60)

        results[name] = _train_pretrained_model(
            model_cls=model_cls,
            model_save_path=MODEL_REGISTRY[name]["path"],
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate * 0.1,  # lower LR for pretrained backbone
            patience=patience,
            image_size=image_size,
        )

    # Summary
    logger.info("=" * 60)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    for name, res in results.items():
        logger.info(f"  {name}: best_val_accuracy = {res['best_val_accuracy']:.4f}")
    logger.info("=" * 60)

    return results


def _train_pretrained_model(
    model_cls,
    model_save_path: str,
    data_dir: str | list[str] | tuple[str, ...] = "/data/combined",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 10,
    image_size: int = 224,
) -> dict:
    """
    Train a pretrained model with differential learning rates.

    The backbone uses a lower LR (preserves pretrained features),
    while the new classifier head trains with a higher LR.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_cls.__name__} on {device}")

    _, train_dataset, val_dataset = _build_combined_datasets(data_dir, image_size)

    class_weights = _compute_balanced_class_weights(_extract_targets(train_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_safe_num_workers(), pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=_safe_num_workers(), pin_memory=True,
    )

    model = model_cls(num_classes=NUM_CLASSES).to(device)
    existing_path = Path(model_save_path)
    if existing_path.exists():
        try:
            model.load_state_dict(torch.load(existing_path, map_location=device, weights_only=True))
            logger.info(f"Warm-started {model_cls.__name__} from {existing_path}")
        except Exception as e:
            logger.warning(f"Could not warm-start {model_cls.__name__} from {existing_path}: {e}")

    # Differential learning rates: backbone gets lower LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": learning_rate},
        {"params": head_params, "lr": learning_rate * 10},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 25 == 0:
                logger.info(
                    f"[{model_cls.__name__}] Epoch {epoch+1}/{epochs} — "
                    f"batch {batch_idx+1}/{len(train_loader)} "
                    f"(train_acc={train_correct / max(1, train_total):.4f})"
                )

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"[{model_cls.__name__}] Epoch {epoch+1}/{epochs} — "
            f"Train Acc: {train_acc:.4f} — Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best {model_cls.__name__} (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping {model_cls.__name__} at epoch {epoch+1}")
                break

    return {
        "best_val_accuracy": best_val_acc,
        "epochs_trained": epoch + 1,
        "model_path": model_save_path,
        "classes": CLASSES,
        "history": history,
    }
