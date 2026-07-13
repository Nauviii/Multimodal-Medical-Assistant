"""Pydantic response models for API endpoints."""

from pydantic import BaseModel


class TokenResponse(BaseModel):
    """Response body for a successful login."""
    access_token: str
    token_type: str = "bearer"


class GradCAMFindingOut(BaseModel):
    """One condition's GradCAM heatmap and zone activation summary."""
    condition: str
    heatmap_url: str
    dominant_zones: list[str]
    aligned: bool


class LLMConditionOut(BaseModel):
    """One condition's clinical explanation from LLM Call 2."""
    name: str
    explanation: str
    dominant_zones: list[str]


class ImageAnalysisResponse(BaseModel):
    """Full response body for POST /analyze/xray."""
    all_scores: dict[str, float]
    above_threshold: list[str]
    low_confidence_flag: bool
    gradcam_findings: list[GradCAMFindingOut]
    conditions: list[LLMConditionOut]
    clinical_summary: str
    cross_specialty_notes: str | None
    latency_ms: int


class TextQAResponse(BaseModel):
    """Response body for POST /query."""
    answer: str
    cross_specialty_notes: str | None
    latency_ms: int