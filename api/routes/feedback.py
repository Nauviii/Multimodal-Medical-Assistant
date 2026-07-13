"""POST /feedback — record doctor feedback (agree/disagree + comment) on a past interaction."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import require_role, TokenPayload
from api.schemas.requests import FeedbackRequest
from api.schemas.responses import FeedbackResponse
from scripts.db_session import get_db
from scripts.db_models import Interaction, UserFeedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    body: FeedbackRequest,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> FeedbackResponse:
    """Attach is_correct/comment feedback to an existing interaction; one feedback per interaction."""
    interaction = db.query(Interaction).filter_by(id=body.interaction_id).first()
    if interaction is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Interaction not found")

    if db.query(UserFeedback).filter_by(interaction_id=body.interaction_id).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Feedback already submitted for this interaction")

    feedback = UserFeedback(
        interaction_id=body.interaction_id,
        is_correct=body.is_correct,
        comment=body.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(
        id=feedback.id, interaction_id=feedback.interaction_id,
        is_correct=feedback.is_correct, comment=feedback.comment,
    )