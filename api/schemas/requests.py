"""Pydantic request models for API endpoints."""

from pydantic import BaseModel


class TextQARequest(BaseModel):
    """Request body for POST /query — a free-text clinical question, optionally continuing a conversation."""
    query: str
    conversation_id: str | None = None


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    interaction_id: str
    is_correct: bool
    comment: str | None = None