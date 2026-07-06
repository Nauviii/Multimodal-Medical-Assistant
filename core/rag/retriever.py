"""Retrieve relevant chunks from Pinecone for image and text Q&A paths."""

from sentence_transformers import SentenceTransformer
from pinecone import Index

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _adaptive_top_k(n_conditions: int) -> int:
    """Return per-condition top_k scaled to number of conditions above threshold."""
    if n_conditions == 1:
        return 4
    if n_conditions <= 3:
        return 3
    return 2


def _query_index(
    query: str,
    index: Index,
    top_k: int,
    namespace: str,
    score_threshold: float,
    condition: str | None = None,
) -> list[dict]:
    """Embed query, run Pinecone query, filter by score threshold, return chunk dicts."""
    vector = _MODEL.encode(query).tolist()
    filter_expr = {"condition": {"$eq": condition}} if condition else None

    response = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        filter=filter_expr,
        include_metadata=True,
    )

    results = []
    for match in response.matches:
        if match.score < score_threshold:
            continue
        results.append({
            "chunk_id":  match.id,
            "condition": match.metadata.get("condition", ""),
            "section":   match.metadata.get("section", ""),
            "source":    match.metadata.get("source", ""),
            "text":      match.metadata.get("text", ""),
            "score":     round(match.score, 4),
        })
    return results


def retrieve_for_image_path(
    rag_queries: list[dict],
    index: Index,
    namespace: str,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Retrieve and deduplicate chunks for all above-threshold conditions (image path).

    Args:
        rag_queries: [{"condition": str, "query": str}, ...] from LLM Call 1 output.
    """
    top_k = _adaptive_top_k(len(rag_queries))
    seen, chunks = set(), []

    for item in rag_queries:
        for chunk in _query_index(
            item["query"], index, top_k, namespace, score_threshold, item["condition"]
        ):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                chunks.append(chunk)

    return chunks


def retrieve_for_text_path(
    query: str,
    index: Index,
    namespace: str,
    top_k: int = 4,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Retrieve chunks without condition filter for general text Q&A path."""
    return _query_index(query, index, top_k, namespace, score_threshold)