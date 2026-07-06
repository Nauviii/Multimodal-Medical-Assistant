"""Integration tests for RAG retrieval pipeline against live Pinecone index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from pinecone import Pinecone

from config.settings import settings
from core.rag.retriever import retrieve_for_image_path, retrieve_for_text_path

REQUIRED_CHUNK_KEYS = {"chunk_id", "condition", "section", "source", "text", "score"}


@pytest.fixture(scope="module")
def index():
    """Shared Pinecone index connection for all tests in this module."""
    return Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)


# --- retrieve_for_text_path ---

def test_text_path_returns_chunks(index):
    """Basic retrieval returns non-empty list with correct chunk structure."""
    results = retrieve_for_text_path(
        query="lung collapse partial airway obstruction",
        index=index,
        namespace=settings.pinecone_namespace,
        top_k=4,
        score_threshold=settings.rag_score_threshold,
    )
    assert len(results) > 0
    assert all(REQUIRED_CHUNK_KEYS.issubset(c.keys()) for c in results)


def test_text_path_score_threshold(index):
    """All returned chunks meet the minimum score threshold."""
    results = retrieve_for_text_path(
        query="pleural fluid accumulation costophrenic blunting",
        index=index,
        namespace=settings.pinecone_namespace,
        top_k=4,
        score_threshold=settings.rag_score_threshold,
    )
    assert all(c["score"] >= settings.rag_score_threshold for c in results)


def test_text_path_top_k_respected(index):
    """Number of returned chunks does not exceed top_k."""
    top_k = 3
    results = retrieve_for_text_path(
        query="pneumonia treatment antibiotics",
        index=index,
        namespace=settings.pinecone_namespace,
        top_k=top_k,
        score_threshold=0.0,  # disable threshold to ensure we're testing top_k only
    )
    assert len(results) <= top_k


def test_text_path_text_field_nonempty(index):
    """Each returned chunk has non-empty text field."""
    results = retrieve_for_text_path(
        query="emphysema hyperinflation air trapping",
        index=index,
        namespace=settings.pinecone_namespace,
        top_k=4,
        score_threshold=settings.rag_score_threshold,
    )
    assert all(len(c["text"].strip()) > 0 for c in results)


# --- retrieve_for_image_path ---

def test_image_path_condition_filter(index):
    """Returned chunks only contain the queried condition."""
    rag_queries = [{"condition": "Effusion", "query": "pleural fluid blunting costophrenic angle"}]
    results = retrieve_for_image_path(
        rag_queries=rag_queries,
        index=index,
        namespace=settings.pinecone_namespace,
        score_threshold=0.0,
    )
    assert all(c["condition"] == "Effusion" for c in results)


def test_image_path_multi_condition_no_crossleak(index):
    """Each condition only returns its own chunks; no cross-condition leakage."""
    rag_queries = [
        {"condition": "Pneumonia", "query": "bacterial pneumonia lobar consolidation fever"},
        {"condition": "Pneumothorax", "query": "absent lung markings visceral pleural line"},
    ]
    results = retrieve_for_image_path(
        rag_queries=rag_queries,
        index=index,
        namespace=settings.pinecone_namespace,
        score_threshold=0.0,
    )
    returned_conditions = {c["condition"] for c in results}
    assert returned_conditions.issubset({"Pneumonia", "Pneumothorax"})


def test_image_path_no_duplicate_chunk_ids(index):
    """Deduplication ensures no repeated chunk_ids in results."""
    rag_queries = [
        {"condition": "Edema", "query": "pulmonary edema bilateral perihilar opacity"},
        {"condition": "Cardiomegaly", "query": "enlarged cardiac silhouette cardiothoracic ratio"},
    ]
    results = retrieve_for_image_path(
        rag_queries=rag_queries,
        index=index,
        namespace=settings.pinecone_namespace,
        score_threshold=0.0,
    )
    chunk_ids = [c["chunk_id"] for c in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_image_path_adaptive_top_k_four_conditions(index):
    """With 4 conditions, per-condition top_k=2; max total chunks = 8."""
    rag_queries = [
        {"condition": "Atelectasis",  "query": "lung collapse volume loss"},
        {"condition": "Effusion",     "query": "pleural effusion fluid"},
        {"condition": "Fibrosis",     "query": "pulmonary fibrosis honeycombing"},
        {"condition": "Nodule",       "query": "solitary pulmonary nodule"},
    ]
    results = retrieve_for_image_path(
        rag_queries=rag_queries,
        index=index,
        namespace=settings.pinecone_namespace,
        score_threshold=0.0,
    )
    assert len(results) <= 8  # 4 conditions x top_k=2


def test_image_path_low_confidence_empty_input(index):
    """Empty rag_queries (low_confidence_flag=True path) returns empty list."""
    results = retrieve_for_image_path(
        rag_queries=[],
        index=index,
        namespace=settings.pinecone_namespace,
    )
    assert results == []