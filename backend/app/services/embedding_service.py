"""
Embedding service for MemoryOS.

Generates dense vector embeddings from text chunks using Sentence Transformers.
Uses a singleton model instance to avoid reloading the model on every call.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Singleton model holder
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_loaded_model_name: str | None = None


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Return the cached SentenceTransformer model, loading it on first call.

    If a different model_name is requested than what is currently loaded,
    the old model is replaced.
    """
    global _model, _loaded_model_name

    if _model is not None and _loaded_model_name == model_name:
        return _model

    logger.info("Loading embedding model: %s", model_name)
    _model = SentenceTransformer(model_name)
    _loaded_model_name = model_name
    dim = (
        _model.get_embedding_dimension()
        if hasattr(_model, "get_embedding_dimension")
        else _model.get_sentence_embedding_dimension()
    )
    logger.info("Embedding model loaded — dimension: %d", dim)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_embeddings(
    chunks: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[List[float]]:
    """
    Generate embedding vectors for a list of text chunks.

    Parameters
    ----------
    chunks : List[str]
        The text chunks to embed. Must not be modified by this function.
    model_name : str, optional
        The Sentence Transformers model to use. Defaults to all-MiniLM-L6-v2.

    Returns
    -------
    List[List[float]]
        A list of embedding vectors, one per input chunk.
        Each vector is a plain Python list of floats.
        Returns an empty list if *chunks* is empty.

    Raises
    ------
    TypeError
        If *chunks* is not a list.
    ValueError
        If any element in *chunks* is not a string.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"Expected a list of strings, got {type(chunks).__name__}")

    if len(chunks) == 0:
        return []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, str):
            raise ValueError(
                f"All chunks must be strings. Item at index {i} is {type(chunk).__name__}."
            )

    model = _get_model(model_name)
    raw_embeddings = model.encode(chunks, show_progress_bar=False)

    # Convert numpy arrays to plain Python lists of floats
    embeddings: List[List[float]] = [
        embedding.tolist() for embedding in raw_embeddings
    ]

    return embeddings


def get_embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Return the embedding vector dimension for the loaded model."""
    model = _get_model(model_name)
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
