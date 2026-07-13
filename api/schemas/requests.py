"""Pydantic request models for API endpoints."""

from pydantic import BaseModel


class TextQARequest(BaseModel):
    """Request body for POST /query — a free-text clinical question."""
    query: str


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    interaction_id: str
    is_correct: bool
    comment: str | None = None