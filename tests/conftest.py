"""Shared pytest fixtures for CNN pipeline tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from core.cnn.inference import load_model


@pytest.fixture(scope="session")
def sample_xray_image() -> Image.Image:
    """Synthetic fixed-seed grayscale-pattern RGB image for pipeline structural tests."""
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 255, size=(224, 224), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@pytest.fixture(scope="session")
def model():
    """Real DenseNet-121 loaded once and shared across the test session."""
    return load_model()