"""GradCAM++ wrapper, per-condition heatmap builder, and semantic context generator.

Produces per-condition heatmap overlays (base64) and a structured semantic
context string for LLM Call 1 input.

Reference: Chattopadhay et al. (2018), GradCAM++: Improved Visual Explanations
for Deep Convolutional Networks, WACV 2018.
"""

import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config.settings import settings
from core.gradcam.region_map import (
    compute_zone_stats,
    get_dominant_zones,
    check_zone_alignment,
    ZONE_LABELS,
)
from core.cnn.inference import ALL_CONDITIONS

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_target_layer(model: nn.Module) -> nn.Module:
    """Return last dense block of DenseNet-121, unwrapping DataParallel if needed."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    return base.features.denseblock4


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized float32 tensor (1, 3, H, W)."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((settings.cnn_image_size, settings.cnn_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(_DEVICE)


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert PIL image to float32 RGB array in [0, 1] for overlay."""
    img = image.convert("RGB").resize(
        (settings.cnn_image_size, settings.cnn_image_size)
    )
    return np.array(img, dtype=np.float32) / 255.0


def _heatmap_to_b64(
    heatmap: np.ndarray,
    rgb_array: np.ndarray,
    condition: str,
) -> str:
    """Blend GradCAM heatmap onto original image and return as base64 PNG.

    Uses condition-specific color from settings.gradcam_condition_colors
    as a tint overlay blended with the standard jet colormap heatmap.
    """
    color_rgba = settings.gradcam_condition_colors.get(condition, [255, 0, 0, 160])
    alpha = color_rgba[3] / 255.0

    # Standard cam overlay (jet colormap blended onto original)
    cam_image = show_cam_on_image(rgb_array, heatmap, use_rgb=True)
    cam_array = np.array(cam_image, dtype=np.float32) / 255.0

    # Tint with condition color
    tint = np.array(color_rgba[:3], dtype=np.float32) / 255.0
    blended = (1 - alpha) * cam_array + alpha * tint
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    pil_out = Image.fromarray(blended)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_gradcam(
    image: Image.Image,
    model: nn.Module,
    above_threshold: list[str],
) -> dict:
    """Run GradCAM++ for each condition above threshold.

    Args:
        image: validated PIL chest X-ray.
        model: loaded DenseNet-121 in eval mode.
        above_threshold: conditions to explain (from CNN inference).

    Returns:
        Dict with gradcam_results per condition and semantic_context string.
    """
    if not above_threshold:
        return {"gradcam_results": {}, "semantic_context": "No significant findings detected above threshold."}

    input_tensor = _image_to_tensor(image)
    rgb_array = _image_to_rgb_array(image)
    target_layer = _get_target_layer(model)

    gradcam_results: dict[str, dict] = {}

    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        for condition in above_threshold:
            class_idx = ALL_CONDITIONS.index(condition)
            targets = [ClassifierOutputTarget(class_idx)]

            # Grayscale heatmap (H, W) normalized to [0, 1]
            heatmap = cam(input_tensor=input_tensor, targets=targets)[0]

            zone_stats = compute_zone_stats(heatmap, settings.cnn_image_size)
            dominant_zones = get_dominant_zones(zone_stats)
            aligned = check_zone_alignment(condition, dominant_zones)

            gradcam_results[condition] = {
                "heatmap_b64": _heatmap_to_b64(heatmap, rgb_array, condition),
                "zone_stats": zone_stats,
                "dominant_zones": dominant_zones,
                "aligned": aligned,
            }

    semantic_context = _build_semantic_context(gradcam_results)

    return {
        "gradcam_results": gradcam_results,
        "semantic_context": semantic_context,
    }


def _build_semantic_context(gradcam_results: dict[str, dict]) -> str:
    """Build structured text description of GradCAM results for LLM Call 1."""
    if not gradcam_results:
        return "No significant findings detected above threshold."

    lines: list[str] = [
        "GradCAM++ activation analysis for conditions above threshold:\n"
    ]

    for condition, result in gradcam_results.items():
        zone_stats = result["zone_stats"]
        dominant_zones = result["dominant_zones"]
        aligned = result["aligned"]

        dominant_labels = [ZONE_LABELS[z] for z in dominant_zones]
        top_zone_str = " and ".join(dominant_labels) if dominant_labels else "no dominant zone"

        alignment_note = (
            "consistent with expected anatomical distribution"
            if aligned else
            "inconsistent with typical anatomical distribution — interpret with caution"
        )

        zone_detail = ", ".join(
            f"{ZONE_LABELS[z]}: {score:.2f}"
            for z, score in sorted(zone_stats.items(), key=lambda x: x[1], reverse=True)
        )

        lines.append(
            f"- {condition}:\n"
            f"  Dominant activation: {top_zone_str}\n"
            f"  Alignment: {alignment_note}\n"
            f"  Zone scores: {zone_detail}"
        )

    return "\n".join(lines)