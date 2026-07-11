"""SQLAlchemy ORM models for all MedAssist logging tables."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    ForeignKey, Integer, JSON, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Session(Base):
    """Satu sesi login user (admin atau doctor)."""
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, nullable=False)
    role = Column(String, nullable=False)    # "admin" | "doctor"
    started_at = Column(DateTime, default=_now)
    ended_at = Column(DateTime, nullable=True)

    interactions = relationship("Interaction", back_populates="session")


class Interaction(Base):
    """One user request (text or image)."""
    __tablename__ = "interactions"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    interaction_type = Column(String, nullable=False)   # "text" | "image"
    raw_query = Column(Text, nullable=True)       # diisi jika text path
    image_hash = Column(String, nullable=True)     # SHA-256 X-ray asli
    xray_storage_url = Column(String, nullable=True)     # Supabase xray-uploads URL
    timestamp = Column(DateTime, default=_now)
    latency_ms = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="interactions")
    cnn_result = relationship("CNNResult", back_populates="interaction", uselist=False)
    rag_logs = relationship("RAGLog", back_populates="interaction")            # one-to-many
    gradcam_findings = relationship("GradCAMFinding", back_populates="interaction")  # one-to-many
    llm_output = relationship("LLMOutput", back_populates="interaction", uselist=False)
    feedback = relationship("UserFeedback", back_populates="interaction", uselist=False)


class CNNResult(Base):
    """Output CNN per image interaction."""
    __tablename__ = "cnn_results"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    all_scores = Column(JSON, nullable=False)         # {condition: confidence}, semua 14
    above_threshold = Column(JSON, nullable=False)    # list[condition], sorted desc by confidence
    low_confidence_flag = Column(Boolean, default=False)

    interaction = relationship("Interaction", back_populates="cnn_result")


class GradCAMFinding(Base):
    """One GradCAM++ heatmap for one condition within a single image interaction."""
    __tablename__ = "gradcam_findings"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    condition = Column(String, nullable=False)
    heatmap_storage_url = Column(String, nullable=False)  # Supabase gradcam-outputs URL
    dominant_zones = Column(JSON, nullable=False)          # list[str], e.g. ["RLZ", "LLZ"]
    aligned = Column(Boolean, nullable=False)
    zone_stats = Column(JSON, nullable=False)              # {zone: activation_score}, 7 zona

    interaction = relationship("Interaction", back_populates="gradcam_findings")


class RAGLog(Base):
    """One retrieval query executed (text path: 1 row; image path: 1 row per condition)."""
    __tablename__ = "rag_logs"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    condition = Column(String, nullable=True)     # null untuk text path
    query_used = Column(Text, nullable=False)     # query yang dikirim ke Pinecone
    retrieved_ids = Column(JSON, nullable=False)  # list chunk_id
    scores = Column(JSON, nullable=False)         # list similarity score, urutan sama dgn retrieved_ids
    timestamp = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="rag_logs")


class LLMOutput(Base):
    """Output LLM Call 1 and Call 2 (image path) or Q&A answers (text path) per interaction."""
    __tablename__ = "llm_outputs"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    call1_output = Column(JSON, nullable=True)    # {rag_queries: [...]} (image path)
    call2_output = Column(JSON, nullable=True)    # {conditions, clinical_summary, cross_specialty_notes}
    text_response = Column(JSON, nullable=True)   # {answer, cross_specialty_notes} (text path)
    timestamp = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="llm_output")


class UserFeedback(Base):
    """Doctor feedback per interaction."""
    __tablename__ = "user_feedback"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    is_correct = Column(Boolean, nullable=False)   # dokter setuju/tidak
    comment = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="feedback")