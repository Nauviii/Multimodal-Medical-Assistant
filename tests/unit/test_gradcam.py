"""Real-weight tests for GradCAM++ explainer and 7-zone region mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import base64
import numpy as np
import pytest

from core.gradcam.explainer import run_gradcam
from core.gradcam.region_map import (
    CHEST_REGIONS,
    compute_zone_stats,
    get_dominant_zones,
    check_zone_alignment,
)


# --- region_map: pure functions, no model dependency ---

def test_all_zones_within_normalized_bounds():
    """Every zone coordinate must fall within [0, 1] with x2>x1 and y2>y1."""
    for x1, y1, x2, y2 in CHEST_REGIONS.values():
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0


def test_compute_zone_stats_returns_all_seven_zones():
    """Zone stats dict must contain exactly the 7 defined anatomical zones."""
    heatmap = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    stats = compute_zone_stats(heatmap, image_size=224)
    assert set(stats.keys()) == set(CHEST_REGIONS.keys())


def test_compute_zone_stats_bounded_by_heatmap_range():
    """Zone mean activation cannot exceed the heatmap's own min/max range."""
    heatmap = np.random.default_rng(1).random((224, 224)).astype(np.float32)
    stats = compute_zone_stats(heatmap, image_size=224)
    assert all(heatmap.min() <= v <= heatmap.max() for v in stats.values())


def test_get_dominant_zones_respects_top_k_and_min_activation():
    """Only zones above min_activation are returned, capped at top_k, sorted desc."""
    stats = {"RUZ": 0.9, "LUZ": 0.8, "RMZ": 0.1, "LMZ": 0.05,
             "CAR": 0.2, "RLZ": 0.85, "LLZ": 0.0}
    dominant = get_dominant_zones(stats, top_k=2, min_activation=0.3)
    assert dominant == ["RUZ", "RLZ"]


def test_check_zone_alignment_true_for_expected_zone():
    """Cardiomegaly aligns when its expected zone (CAR) is dominant."""
    assert check_zone_alignment("Cardiomegaly", ["CAR"]) is True


def test_check_zone_alignment_false_for_unexpected_zone():
    """Cardiomegaly does not align when only an unrelated zone is dominant."""
    assert check_zone_alignment("Cardiomegaly", ["RUZ"]) is False


# --- explainer: real model + real GradCAM++ ---

def test_run_gradcam_empty_when_no_conditions_above_threshold(model, sample_xray_image):
    """Empty above_threshold must short-circuit to a no-findings result."""
    result = run_gradcam(sample_xray_image, model, above_threshold=[])
    assert result["gradcam_results"] == {}
    assert "No significant findings" in result["semantic_context"]


def test_run_gradcam_produces_heatmap_per_condition(model, sample_xray_image):
    """Each above-threshold condition yields a valid PNG heatmap and 7-zone stats."""
    conditions = ["Effusion", "Cardiomegaly"]
    result = run_gradcam(sample_xray_image, model, above_threshold=conditions)

    assert set(result["gradcam_results"].keys()) == set(conditions)
    for cond in conditions:
        entry = result["gradcam_results"][cond]
        assert set(entry["zone_stats"].keys()) == set(CHEST_REGIONS.keys())
        assert len(entry["dominant_zones"]) <= 2
        assert isinstance(entry["aligned"], bool)
        decoded = base64.b64decode(entry["heatmap_b64"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_semantic_context_mentions_all_positive_conditions(model, sample_xray_image):
    """semantic_context string must reference every above-threshold condition."""
    conditions = ["Infiltration", "Atelectasis"]
    result = run_gradcam(sample_xray_image, model, above_threshold=conditions)
    for cond in conditions:
        assert cond in result["semantic_context"]