"""Embed text chunks and upsert vectors to Pinecone."""

from sentence_transformers import SentenceTransformer
from pinecone import Index

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_and_upsert(
    chunks: list[dict],
    index: Index,
    namespace: str,
    batch_size: int = 100,
) -> int:
    """Embed chunks and upsert to Pinecone; return number of vectors upserted."""
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = _MODEL.encode(texts, batch_size=32, show_progress_bar=False)

    vectors = [
        {
            "id": c["chunk_id"],
            "values": emb.tolist(),
            "metadata": {
                "condition": c["condition"],
                "section":   c["section"],
                "source":    c["source"],
                "text":      c["text"],
            },
        }
        for c, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)

    return len(vectors)