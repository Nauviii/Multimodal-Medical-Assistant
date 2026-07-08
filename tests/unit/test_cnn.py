"""Real-weight tests for DenseNet-121 multi-label inference pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pytest

from config.settings import settings
from core.cnn.inference import ALL_CONDITIONS, run_inference


@pytest.fixture(scope="module")
def thresholds():
    """Real per-class thresholds from settings.thresholds_path."""
    with open(settings.thresholds_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def result(model, thresholds, sample_xray_image):
    """Single inference run reused across assertions."""
    return run_inference(sample_xray_image, model, thresholds)


def test_model_loaded_in_eval_mode(model):
    """Loaded model must be in eval mode, not training mode."""
    assert model.training is False


def test_all_scores_covers_all_conditions_in_order(result):
    """all_scores keys must match ALL_CONDITIONS exactly, in fixed order."""
    assert list(result["all_scores"].keys()) == ALL_CONDITIONS


def test_all_scores_are_valid_probabilities(result):
    """Every sigmoid output must lie in [0, 1]."""
    assert all(0.0 <= v <= 1.0 for v in result["all_scores"].values())


def test_above_threshold_matches_threshold_comparison(result, thresholds):
    """Condition is in above_threshold iff its score meets its own threshold."""
    scores = result["all_scores"]
    for cond in ALL_CONDITIONS:
        is_above = scores[cond] >= thresholds[cond]
        assert (cond in result["above_threshold"]) == is_above


def test_above_threshold_sorted_descending(result):
    """above_threshold must be sorted by confidence, highest first."""
    scores = [result["all_scores"][c] for c in result["above_threshold"]]
    assert scores == sorted(scores, reverse=True)


def test_low_confidence_flag_consistency(result):
    """low_confidence_flag is True iff above_threshold is empty."""
    assert result["low_confidence_flag"] == (len(result["above_threshold"]) == 0)