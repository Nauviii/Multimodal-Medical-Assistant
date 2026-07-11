"""POST /auth/token — authenticate a user and issue a JWT access token."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import create_access_token, verify_password
from api.schemas.responses import TokenResponse
from scripts.db_session import get_db
from scripts.db_models import User, Session as UserSession

router = APIRouter()


@router.post("/auth/token", response_model=TokenResponse)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[DBSession, Depends(get_db)],
) -> TokenResponse:
    """Verify credentials, open a new Session row, and return a JWT access token."""
    user = db.query(User).filter_by(username=form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")

    session = UserSession(user_id=user.id, role=user.role)
    db.add(session)
    db.commit()
    db.refresh(session)

    token = create_access_token(user.id, user.role, session.id)
    return TokenResponse(access_token=token, token_type="bearer")