"""Real-weight tests for CLIP-based chest X-ray validator (two-layer)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pytest

from config.settings import settings
from core.clip.validator import validate


@pytest.fixture(scope="module")
def prototype():
    """Real calibrated CLIP prototype (centroid + threshold) from clip_prototype.json."""
    with open(settings.clip_prototype) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def result(sample_xray_image):
    """Single validation run reused across assertions."""
    return validate(sample_xray_image)


def test_cosine_scores_within_valid_range(result):
    """Cosine similarity scores (Layer 1) must lie in [-1, 1]."""
    assert -1.0 <= result.valid_score <= 1.0
    assert -1.0 <= result.invalid_score <= 1.0


def test_distance_is_non_negative(result):
    """Euclidean distance to centroid (Layer 2) cannot be negative."""
    assert result.distance >= 0.0


def test_threshold_matches_calibrated_prototype(result, prototype):
    """Returned threshold must equal the value calibrated in clip_prototype.json."""
    assert result.threshold == round(prototype["threshold"], 4)


def test_centroid_dimension_matches_clip_embedding_size(prototype):
    """Centroid vector length must equal CLIP ViT-B/32 embedding dimension."""
    assert len(prototype["centroid"]) == 512


def test_is_valid_iff_both_layers_passed(result):
    """is_valid must be the logical AND of layer1_passed and layer2_passed."""
    assert result.is_valid == (result.layer1_passed and result.layer2_passed)


def test_layer1_fail_reason_mentions_chest_xray(result):
    """If Layer 1 fails, reason must explain non-resemblance to a chest X-ray."""
    if not result.layer1_passed:
        assert "chest X-ray" in result.reason


def test_layer2_fail_reason_mentions_distribution(result):
    """If Layer 1 passes but Layer 2 fails, reason must reference distribution mismatch."""
    if result.layer1_passed and not result.layer2_passed:
        assert "distribution" in result.reason