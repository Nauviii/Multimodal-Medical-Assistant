"""Shared fixtures for integration tests: live app, DB session, throwaway authenticated users.

These tests hit real Groq, Pinecone, Redis, and Supabase Postgres — consistent with the
testing approach used throughout this project (test_rag.py, test_llm.py). Each fixture
that creates DB rows registers them for cleanup so repeated test runs don't pollute the
real database with throwaway integration-test data.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app
from scripts.db_session import _SessionLocal
from scripts.db_models import (
    User, Session as UserSession, Interaction,
    CNNResult, GradCAMFinding, RAGLog, LLMOutput, UserFeedback,
)
from api.middleware.auth import hash_password, create_access_token


@pytest.fixture(scope="session")
def client():
    """Shared TestClient wrapping the real app — triggers real lifespan (CNN model, Pinecone index)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db():
    """A DB session for direct setup/assertions/cleanup within a test."""
    session = _SessionLocal()
    yield session
    session.close()


def _make_authenticated_doctor(db, suffix: str = ""):
    """Create a throwaway doctor user + session + JWT token."""
    username = f"integration_test_doctor_{uuid.uuid4().hex[:8]}{suffix}"
    user = User(username=username, hashed_password=hash_password("testpass123"), role="doctor")
    db.add(user); db.commit(); db.refresh(user)

    user_session = UserSession(user_id=user.id, role="doctor")
    db.add(user_session); db.commit(); db.refresh(user_session)

    token = create_access_token(user.id, "doctor", user_session.id)
    return {"user": user, "session": user_session, "token": token,
            "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def doctor(db):
    """A throwaway authenticated doctor, deleted after the test."""
    d = _make_authenticated_doctor(db)
    yield d
    db.delete(d["session"])
    db.delete(d["user"])
    db.commit()


@pytest.fixture
def second_doctor(db):
    """A second throwaway authenticated doctor, for ownership/cross-access tests."""
    d = _make_authenticated_doctor(db, "_2")
    yield d
    db.delete(d["session"])
    db.delete(d["user"])
    db.commit()


def _cleanup_conversation_cascade(db, conversation_id: str) -> None:
    """Delete every interaction in a conversation and their child rows, in FK-safe order."""
    interactions = db.query(Interaction).filter_by(conversation_id=conversation_id).all()
    ids = [i.id for i in interactions]
    if not ids:
        return

    db.query(UserFeedback).filter(UserFeedback.interaction_id.in_(ids)).delete(synchronize_session=False)
    db.query(LLMOutput).filter(LLMOutput.interaction_id.in_(ids)).delete(synchronize_session=False)
    db.query(RAGLog).filter(RAGLog.interaction_id.in_(ids)).delete(synchronize_session=False)
    db.query(GradCAMFinding).filter(GradCAMFinding.interaction_id.in_(ids)).delete(synchronize_session=False)
    db.query(CNNResult).filter(CNNResult.interaction_id.in_(ids)).delete(synchronize_session=False)

    # Self-referential FK (conversation_id -> interactions.id): delete non-anchor rows
    # before the anchor row (the one whose id == conversation_id) to satisfy the constraint.
    non_anchor_ids = [i for i in ids if i != conversation_id]
    if non_anchor_ids:
        db.query(Interaction).filter(Interaction.id.in_(non_anchor_ids)).delete(synchronize_session=False)
    db.query(Interaction).filter_by(id=conversation_id).delete()
    db.commit()


@pytest.fixture
def cleanup_conversation():
    """Returns a function tests call with a conversation_id to register for post-test cleanup."""
    registered: list[str] = []

    def _register(conversation_id: str) -> None:
        registered.append(conversation_id)

    yield _register

    cleanup_db = _SessionLocal()
    for conversation_id in registered:
        _cleanup_conversation_cascade(cleanup_db, conversation_id)
    cleanup_db.close()