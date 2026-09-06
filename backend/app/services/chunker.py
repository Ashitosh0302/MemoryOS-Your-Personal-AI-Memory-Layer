"""
Document chunking service for MemoryOS.

Splits raw or extracted document text into manageable, overlapping chunks
suitable for vector search, embeddings, and context retrieval.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split extracted document text into smaller chunks using a sliding window.

    Parameters
    ----------
    text : str
        The extracted document text to split.
    chunk_size : int, default=500
        Maximum number of characters per chunk. Must be greater than 0.
    chunk_overlap : int, default=50
        Number of characters to overlap between consecutive chunks.
        Must be non-negative and strictly less than `chunk_size`.

    Returns
    -------
    List[str]
        A list of clean text chunk strings. If `text` is empty or only whitespace,
        returns an empty list.

    Raises
    ------
    ValueError
        If `chunk_size` <= 0, `chunk_overlap` < 0, or `chunk_overlap` >= `chunk_size`.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # Short text remains a single chunk
    if len(cleaned) <= chunk_size:
        return [cleaned]

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    start = 0
    total_len = len(cleaned)

    while start < total_len:
        end = start + chunk_size
        chunk = cleaned[start:end]
        chunks.append(chunk)

        if end >= total_len:
            break

        start += step

    return chunks
