"""7-zone anatomical region mapping for frontal chest X-ray GradCAM analysis.

Zone boundaries follow Felson (1973) Chest Roentgenology conventions and
Hansell et al. (2008) Fleischner Society guidelines for condition-to-region
clinical distribution on PA-view frontal CXR.
"""

import numpy as np

# Zone definitions (normalized coordinates: x1, y1, x2, y2)
# Origin top-left. Radiological convention: patient's right = image left.
CHEST_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "RUZ": (0.00, 0.00, 0.50, 0.33),   # right upper zone (patient right = image left)
    "LUZ": (0.50, 0.00, 1.00, 0.33),   # left upper zone
    "RMZ": (0.00, 0.33, 0.42, 0.63),   # right mid zone
    "LMZ": (0.58, 0.33, 1.00, 0.63),   # left mid zone
    "CAR": (0.33, 0.30, 0.67, 0.73),   # cardiac / mediastinum
    "RLZ": (0.00, 0.63, 0.50, 1.00),   # right lower zone
    "LLZ": (0.50, 0.63, 1.00, 1.00),   # left lower zone
}

# Clinical condition-to-zone mapping
# Based on dominant radiological distribution per condition.
# Reference: Hansell et al. (2008), Felson (1973).
CONDITION_TO_ZONES: dict[str, list[str]] = {
    "Emphysema":          ["RUZ", "LUZ"],
    "Pneumothorax":       ["RUZ", "LUZ"],
    "Infiltration":       ["RMZ", "LMZ"],
    "Mass":               ["RMZ", "LMZ"],
    "Nodule":             ["RMZ", "LMZ"],
    "Cardiomegaly":       ["CAR"],
    "Edema":              ["CAR", "RLZ", "LLZ"],
    "Fibrosis":           ["RLZ", "LLZ"],
    "Effusion":           ["RLZ", "LLZ"],
    "Atelectasis":        ["RLZ", "LLZ"],
    "Consolidation":      ["RLZ", "LLZ"],
    "Pneumonia":          ["RLZ", "LLZ"],
    "Pleural_Thickening": ["RLZ", "LLZ"],
    "Hernia":             ["RLZ"],
}

# Zone full names for semantic context
ZONE_LABELS: dict[str, str] = {
    "RUZ": "right upper zone",
    "LUZ": "left upper zone",
    "RMZ": "right mid zone",
    "LMZ": "left mid zone",
    "CAR": "cardiac/mediastinum",
    "RLZ": "right lower zone",
    "LLZ": "left lower zone",
}


def compute_zone_stats(
    heatmap: np.ndarray,
    image_size: int = 224,
) -> dict[str, float]:
    """Compute mean GradCAM activation per anatomical zone.

    Args:
        heatmap: 2D float32 array (H, W) normalized to [0, 1].
        image_size: pixel size of the square heatmap.

    Returns:
        Dict mapping zone name to mean activation score in [0, 1].
    """
    h = w = image_size
    zone_stats: dict[str, float] = {}

    for zone, (x1, y1, x2, y2) in CHEST_REGIONS.items():
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)
        region = heatmap[py1:py2, px1:px2]
        zone_stats[zone] = float(round(float(region.mean()), 4)) if region.size > 0 else 0.0

    return zone_stats


def get_dominant_zones(
    zone_stats: dict[str, float],
    top_k: int = 2,
    min_activation: float = 0.3,
) -> list[str]:
    """Return top-k zones above min_activation threshold, sorted by activation."""
    candidates = {z: s for z, s in zone_stats.items() if s >= min_activation}
    return sorted(candidates, key=lambda z: candidates[z], reverse=True)[:top_k]


def check_zone_alignment(
    condition: str,
    dominant_zones: list[str],
) -> bool:
    """Check whether dominant GradCAM zones match expected clinical zones."""
    expected = set(CONDITION_TO_ZONES.get(condition, []))
    return any(z in expected for z in dominant_zones)