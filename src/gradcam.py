from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _colormap_jet(gray: np.ndarray) -> np.ndarray:
    """Small dependency-free JET-like colormap, returns RGB float image [0, 1]."""
    gray = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np -= cam_np.min()
        max_value = cam_np.max()
        if max_value > 0:
            cam_np /= max_value
        return cam_np.astype("float32")


def make_overlay(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    original = original_image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    original_np = np.asarray(original).astype("float32") / 255.0
    colored_heatmap = _colormap_jet(heatmap)
    overlay = (1.0 - alpha) * original_np + alpha * colored_heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(overlay)


def heatmap_to_image(heatmap: np.ndarray) -> Image.Image:
    colored = (_colormap_jet(heatmap) * 255.0).astype("uint8")
    return Image.fromarray(colored)
