"""DenseNet-121 multi-label inference pipeline for NIH ChestX-ray14."""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm

from config.settings import settings


ALL_CONDITIONS: list[str] = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_transform() -> transforms.Compose:
    """Deterministic eval transform matching training pipeline."""
    return transforms.Compose([
        transforms.Resize((settings.cnn_image_size, settings.cnn_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load_thresholds() -> dict[str, float]:
    """Load per-class thresholds from JSON artifact."""
    with open(settings.thresholds_path) as f:
        return json.load(f)


def load_model() -> nn.Module:
    """Load DenseNet-121 weights from disk and set to eval mode."""
    model = timm.create_model(
        "densenet121", pretrained=False,
        num_classes=settings.cnn_num_classes,
    )
    state = torch.load(settings.model_weights_path, map_location=_DEVICE)
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(_DEVICE)
    model.eval()
    return model


def run_inference(
    image: Image.Image,
    model: nn.Module,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Run CNN inference on a validated chest X-ray PIL image.

    Returns all_scores, above_threshold list, and low_confidence_flag.
    """
    if thresholds is None:
        thresholds = _load_thresholds()

    transform = _build_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    all_scores: dict[str, float] = {
        cond: round(float(prob), 4)
        for cond, prob in zip(ALL_CONDITIONS, probs)
    }

    above_threshold: list[str] = [
        cond for cond in ALL_CONDITIONS
        if all_scores[cond] >= thresholds[cond]
    ]

    # Sort above_threshold by confidence descending
    above_threshold.sort(key=lambda c: all_scores[c], reverse=True)

    low_confidence_flag: bool = len(above_threshold) == 0

    return {
        "all_scores": all_scores,
        "above_threshold": above_threshold,
        "low_confidence_flag": low_confidence_flag,
    }