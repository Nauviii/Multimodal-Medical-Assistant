"""Redis-backed patient session memory with sliding TTL for conversational follow-ups."""

import json
import redis
from config.settings import settings

_r = redis.from_url(settings.redis_url, decode_responses=True)

_TOMBSTONE_PREFIX = "closed:"


def _session_key(session_id: str) -> str:
    """Build the Redis key for a session."""
    return f"session:{session_id}"


def _tombstone_key(session_id: str) -> str:
    """Build the Redis key for a conversation's closed marker."""
    return f"{_TOMBSTONE_PREFIX}{session_id}"


def get_session(session_id: str) -> dict | None:
    """Return session state, or None if expired or not found."""
    raw = _r.get(_session_key(session_id))
    return json.loads(raw) if raw else None


def save_session(session_id: str, state: dict) -> None:
    """Write session state and refresh TTL (sliding expiration)."""
    _r.set(_session_key(session_id), json.dumps(state), ex=settings.redis_session_ttl_seconds)


def append_turn(session_id: str, role: str, content: str) -> dict:
    """Append one conversation turn to a session, creating it if absent, and refresh TTL."""
    state = get_session(session_id) or {"conversation": []}
    state["conversation"].append({"role": role, "content": content})
    save_session(session_id, state)
    return state


def end_session(session_id: str) -> None:
    """Explicitly delete working memory and mark the conversation closed (no automatic revival)."""
    _r.delete(_session_key(session_id))
    _r.set(_tombstone_key(session_id), "1")  # permanent — no TTL, survives until explicitly cleared


def is_conversation_closed(session_id: str) -> bool:
    """Return True if the conversation was explicitly closed via end_session."""
    return bool(_r.exists(_tombstone_key(session_id)))