"""Shared FastAPI dependencies: lazily-loaded CNN model and Pinecone index singletons."""

from pinecone import Index, Pinecone

from config.settings import settings
from core.cnn.inference import load_model

_cnn_model = None
_pinecone_index = None


def get_cnn_model():
    """FastAPI dependency: return the cached DenseNet-121 model, loading it on first call."""
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = load_model()
    return _cnn_model


def get_pinecone_index() -> Index:
    """FastAPI dependency: return the cached Pinecone index connection, creating it on first call."""
    global _pinecone_index
    if _pinecone_index is None:
        _pinecone_index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    return _pinecone_index