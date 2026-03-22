"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI.

Generates visual heatmaps showing which regions of a brain MRI image
each model focused on when making its classification decision.

This is research-level XAI — it provides visual proof of model reasoning
by backpropagating gradients to the last convolutional layer.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for any CNN-based classifier.

    How it works:
      1. Forward pass: capture feature maps from the target convolutional layer
      2. Backward pass: compute gradients of the predicted class w.r.t. those feature maps
      3. Weight: global average pool the gradients → importance weights per channel
      4. Combine: weighted sum of feature maps → ReLU → normalize to [0, 1]
      5. Result: a 2D heatmap (same spatial size as the feature map) showing
         which regions contributed most to the prediction
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Trained PyTorch classifier (evaluation mode).
            target_layer: The convolutional layer to hook into (usually the last conv).
        """
        self.model = model
        self.target_layer = target_layer

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: capture feature map activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: capture gradients flowing back through the target layer."""
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given input.

        Args:
            input_tensor: [1, C, H, W] preprocessed image tensor.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            2D numpy array (H, W) with values in [0, 1] — the heatmap.
        """
        self.model.eval()

        # Enable gradient computation for the input
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass from the target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Get captured gradients and activations
        gradients = self._gradients  # [1, C, h, w]
        activations = self._activations  # [1, C, h, w]

        if gradients is None or activations is None:
            logger.error("Grad-CAM: No gradients or activations captured")
            return np.zeros((7, 7))

        # Global average pooling of gradients → channel importance weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def generate_heatmap_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Generate a Grad-CAM heatmap overlaid on the original image.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W].
            original_image: Original PIL image for overlay.
            target_class: Class to explain (None = predicted class).
            alpha: Overlay transparency (0 = original only, 1 = heatmap only).

        Returns:
            PIL Image with heatmap overlay (RGB).
        """
        cam = self.generate(input_tensor, target_class)

        # Resize heatmap to original image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                original_image.size, Image.BILINEAR
            )
        ) / 255.0

        # Apply colormap (jet-like: blue → green → red)
        heatmap_colored = self._apply_colormap(cam_resized)

        # Blend with original
        original_np = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
        blended = (1 - alpha) * original_np + alpha * heatmap_colored
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    @staticmethod
    def _apply_colormap(gray: np.ndarray) -> np.ndarray:
        """Apply a jet-like colormap to a grayscale [0,1] array → RGB [0,1] float."""
        r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0, 1)
        return np.stack([r, g, b], axis=-1)

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_gradcam_for_model(model: torch.nn.Module, model_name: str) -> Optional[GradCAM]:
    """
    Factory function to create a GradCAM instance for a given model architecture.

    Automatically identifies the correct target layer based on the model name.

    Args:
        model: Loaded PyTorch model.
        model_name: One of 'custom_cnn', 'resnet50', 'efficientnet', 'densenet'.

    Returns:
        GradCAM instance or None if the target layer cannot be found.
    """
    target_layer = None

    try:
        if model_name == "custom_cnn":
            # Last conv block in the custom BrainTumorClassifier
            target_layer = model.features[-1] if hasattr(model, "features") else None
            # Fallback: find the last Conv2d layer
            if target_layer is None:
                for module in reversed(list(model.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break

        elif model_name == "resnet50":
            # ResNet-50: last layer of the 4th residual block
            if hasattr(model, "backbone"):
                target_layer = model.backbone.layer4[-1].conv3
            elif hasattr(model, "layer4"):
                target_layer = model.layer4[-1].conv3

        elif model_name == "efficientnet":
            # EfficientNet-B0: last convolutional feature
            if hasattr(model, "backbone"):
                target_layer = model.backbone.features[-1]
            elif hasattr(model, "features"):
                target_layer = model.features[-1]

        elif model_name == "densenet":
            # DenseNet-121: last dense block
            if hasattr(model, "backbone"):
                target_layer = model.backbone.features.denseblock4
            elif hasattr(model, "features"):
                target_layer = model.features.denseblock4

    except (AttributeError, IndexError) as e:
        logger.warning(f"Could not find target layer for {model_name}: {e}")

    if target_layer is None:
        # Ultimate fallback: last Conv2d in the model
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        logger.error(f"No suitable Conv2d layer found for Grad-CAM in {model_name}")
        return None

    logger.info(f"Grad-CAM initialized for {model_name} → {type(target_layer).__name__}")
    return GradCAM(model, target_layer)
