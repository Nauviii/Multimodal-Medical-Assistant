"""SQLAlchemy ORM models untuk semua tabel logging MedAssist."""

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


class User(Base):
    """A registered admin or doctor account."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)      # "admin" | "doctor"
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    """Satu sesi login user (admin atau doctor)."""
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)
    started_at = Column(DateTime, default=_now)
    ended_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")


class Interaction(Base):
    """Satu request dari user (text atau image)."""
    __tablename__ = "interactions"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    interaction_type = Column(String, nullable=False)   
    raw_query = Column(Text, nullable=True)       
    image_hash = Column(String, nullable=True)     
    xray_storage_url = Column(String, nullable=True)     
    timestamp = Column(DateTime, default=_now)
    latency_ms = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="interactions")
    cnn_result = relationship("CNNResult", back_populates="interaction", uselist=False)
    rag_logs = relationship("RAGLog", back_populates="interaction")            
    gradcam_findings = relationship("GradCAMFinding", back_populates="interaction")  
    llm_output = relationship("LLMOutput", back_populates="interaction", uselist=False)
    feedback = relationship("UserFeedback", back_populates="interaction", uselist=False)


class CNNResult(Base):
    """Output CNN per image interaction."""
    __tablename__ = "cnn_results"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    all_scores = Column(JSON, nullable=False)        
    above_threshold = Column(JSON, nullable=False)    
    low_confidence_flag = Column(Boolean, default=False)

    interaction = relationship("Interaction", back_populates="cnn_result")


class GradCAMFinding(Base):
    """Satu heatmap GradCAM++ untuk satu kondisi dalam satu image interaction."""
    __tablename__ = "gradcam_findings"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    condition = Column(String, nullable=False)
    heatmap_storage_url = Column(String, nullable=False)  
    dominant_zones = Column(JSON, nullable=False)         
    aligned = Column(Boolean, nullable=False)
    zone_stats = Column(JSON, nullable=False)              

    interaction = relationship("Interaction", back_populates="gradcam_findings")


class RAGLog(Base):
    """Satu query retrieval yang dieksekusi (text path: 1 row; image path: 1 row per kondisi)."""
    __tablename__ = "rag_logs"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    condition = Column(String, nullable=True)     
    query_used = Column(Text, nullable=False)     
    retrieved_ids = Column(JSON, nullable=False)  
    scores = Column(JSON, nullable=False)         
    timestamp = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="rag_logs")


class LLMOutput(Base):
    """Output LLM Call 1 dan Call 2 (image path) atau jawaban Q&A (text path) per interaction."""
    __tablename__ = "llm_outputs"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    call1_output = Column(JSON, nullable=True)    # {rag_queries: [...]} (image path)
    call2_output = Column(JSON, nullable=True)    # {conditions, clinical_summary, cross_specialty_notes}
    text_response = Column(JSON, nullable=True)   # {answer, cross_specialty_notes} (text path)
    timestamp = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="llm_output")


class UserFeedback(Base):
    """Feedback dokter per interaction."""
    __tablename__ = "user_feedback"

    id = Column(String, primary_key=True, default=_uuid)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    is_correct = Column(Boolean, nullable=False) 
    comment = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=_now)

    interaction = relationship("Interaction", back_populates="feedback")