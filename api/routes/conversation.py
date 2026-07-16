"""GET /conversation/{id} — read full transcript from Postgres.
DELETE /conversation/{id} — explicitly close working memory (Redis only)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import require_role, TokenPayload
from api.schemas.responses import (
    ConversationTranscriptResponse, ConversationTurnOut, ConversationCloseResponse,
)
from scripts.db_session import get_db
from scripts.db_models import Interaction, Session as UserSession
from core.memory.session_memory import end_session
from core.memory.conversation_history import get_conversation_transcript

router = APIRouter()


def _check_ownership(conversation_id: str, user: TokenPayload, db: DBSession) -> Interaction:
    """Return the conversation's anchor interaction, or raise 404/403 if inaccessible."""
    anchor = db.query(Interaction).filter_by(id=conversation_id).first()
    if anchor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    if user.role != "admin":
        owner = db.query(UserSession).filter_by(id=anchor.session_id).first()
        if owner is None or owner.user_id != user.sub:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your conversation")

    return anchor


@router.get("/conversation/{conversation_id}", response_model=ConversationTranscriptResponse)
def get_conversation(
    conversation_id: str,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> ConversationTranscriptResponse:
    """Return the full transcript of a conversation; readable even after it has been closed."""
    _check_ownership(conversation_id, user, db)
    transcript = get_conversation_transcript(conversation_id, db)
    return ConversationTranscriptResponse(
        conversation_id=conversation_id,
        turns=[ConversationTurnOut(**t) for t in transcript],
    )


@router.delete("/conversation/{conversation_id}", response_model=ConversationCloseResponse)
def close_conversation(
    conversation_id: str,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> ConversationCloseResponse:
    """Explicitly close a conversation's working memory; permanent Postgres history is unaffected."""
    _check_ownership(conversation_id, user, db)
    end_session(conversation_id)
    return ConversationCloseResponse(conversation_id=conversation_id, closed=True)