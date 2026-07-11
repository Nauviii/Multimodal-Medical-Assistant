"""JWT token creation, decoding, and role-based access control for API routes."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from pydantic import BaseModel

from config.settings import settings

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a plaintext password using the recommended Argon2 configuration."""
    return _password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return _password_hash.verify(password, hashed)


class TokenPayload(BaseModel):
    """Decoded JWT claims identifying the authenticated user and session."""
    sub: str          # user_id
    role: str         # "admin" | "doctor"
    session_id: str


def create_access_token(user_id: str, role: str, session_id: str) -> str:
    """Encode a JWT access token for a newly authenticated session."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": user_id, "role": role, "session_id": session_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> TokenPayload:
    """Decode and validate a JWT access token; raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return TokenPayload(sub=payload["sub"], role=payload["role"], session_id=payload["session_id"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except (jwt.InvalidTokenError, KeyError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def get_current_user(token: Annotated[str, Depends(_oauth2_scheme)]) -> TokenPayload:
    """FastAPI dependency: decode the bearer token into the current user's claims."""
    return decode_access_token(token)


def require_role(*allowed_roles: str):
    """Return a FastAPI dependency that rejects users whose role is not in allowed_roles."""
    def _check(user: Annotated[TokenPayload, Depends(get_current_user)]) -> TokenPayload:
        if user.role not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")
        return user
    return _check