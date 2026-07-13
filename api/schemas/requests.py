"""Pydantic request models for API endpoints."""

from pydantic import BaseModel


class TextQARequest(BaseModel):
    """Request body for POST /query — a free-text clinical question."""
    query: str