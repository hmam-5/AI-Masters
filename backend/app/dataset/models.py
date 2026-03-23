"""
Multi-model architectures for brain tumor classification.

Provides four different pretrained models that are fine-tuned on brain tumor data:
  1. BrainTumorCNN      — Custom 5-block CNN (from trainer.py)
  2. ResNet50Classifier  — ResNet-50 pretrained on ImageNet
  3. EfficientNetClassifier — EfficientNet-B0 pretrained on ImageNet
  4. DenseNetClassifier  — DenseNet-121 pretrained on ImageNet

Using multiple architectures in an ensemble dramatically boosts confidence and
accuracy beyond what any single model can achieve (target: >98%).
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


class ResNet50Classifier(nn.Module):
    """ResNet-50 fine-tuned for brain tumor classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 fine-tuned for brain tumor classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNetClassifier(nn.Module):
    """DenseNet-121 fine-tuned for brain tumor classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


from app.config import get_settings

# Registry mapping model names to classes and their save paths.
# Paths come from settings (which reads from .env), so they work
# in both Docker (/models/) and local dev (../models/) modes.
def _build_registry() -> dict:
    s = get_settings()
    return {
        "custom_cnn": {
            "path": s.model_2d_path,
            "weight": 1.0,
        },
        "resnet50": {
            "class": ResNet50Classifier,
            "path": s.model_resnet50_path,
            "weight": 1.5,
        },
        "efficientnet": {
            "class": EfficientNetClassifier,
            "path": s.model_efficientnet_path,
            "weight": 1.5,
        },
        "densenet": {
            "class": DenseNetClassifier,
            "path": s.model_densenet_path,
            "weight": 1.3,
        },
    }

MODEL_REGISTRY = _build_registry()
