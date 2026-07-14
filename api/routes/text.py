"""POST /query — text Q&A pipeline: guardrail check, retrieval, LLM call, DB logging."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends
from pinecone import Index
from sqlalchemy.orm import Session as DBSession

from config.settings import settings
from api.dependencies import get_pinecone_index
from api.middleware.auth import require_role, TokenPayload
from api.schemas.requests import TextQARequest
from api.schemas.responses import TextQAResponse
from scripts.db_session import get_db
from scripts.db_models import Interaction, RAGLog, LLMOutput
from core.llm.orchestrator import run_text_llm_pipeline

router = APIRouter()


@router.post("/query", response_model=TextQAResponse)
def text_qa(
    body: TextQARequest,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
    index: Annotated[Index, Depends(get_pinecone_index)],
) -> TextQAResponse:
    """Run the text Q&A pipeline and persist the interaction, retrieval log, and LLM output."""
    start = time.perf_counter()

    interaction = Interaction(
        session_id=user.session_id, interaction_type="text", raw_query=body.query,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    bundle = run_text_llm_pipeline(body.query, index, settings.pinecone_namespace)

    if bundle["rag_chunks"]:
        db.add(RAGLog(
            interaction_id=interaction.id, condition=None,
            query_used=bundle["query_used"],
            retrieved_ids=[c["chunk_id"] for c in bundle["rag_chunks"]],
            scores=[c["score"] for c in bundle["rag_chunks"]],
        ))
        db.commit()

    db.add(LLMOutput(
        interaction_id=interaction.id,
        call1_output=None,
        call2_output=None,
        text_response=bundle["answer_output"],
    ))

    interaction.latency_ms = int((time.perf_counter() - start) * 1000)
    db.commit()

    return TextQAResponse(
        answer=bundle["answer_output"]["answer"],
        cross_specialty_notes=bundle["answer_output"]["cross_specialty_notes"],
        latency_ms=interaction.latency_ms,
    )