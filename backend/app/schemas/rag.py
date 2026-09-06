"""
RAG (Retrieval-Augmented Generation) schemas for MemoryOS API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class RAGRequest(BaseModel):
    """
    RAG query request schema.
    """

    query: str = Field(
        ...,
        description="The natural-language question to answer.",
        examples=["How much does an electrician charge per hour?"],
    )
    top_k: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Number of relevant chunks to retrieve (1-100).",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string must not be empty or whitespace.")
        return v.strip()


class RAGSourceItem(BaseModel):
    """
    A single retrieved source chunk included in the RAG response.
    """

    id: str = Field(..., description="Unique chunk identifier.")
    text: str = Field(..., description="Original chunk text content.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the chunk."
    )
    distance: Optional[float] = Field(
        default=None, description="Similarity distance score (lower is more similar)."
    )


class RAGResponse(BaseModel):
    """
    RAG response schema containing the generated answer and its sources.
    """

    success: bool = Field(..., description="Whether the request succeeded.")
    query: str = Field(..., description="The original user question.")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved context.")
    sources: List[RAGSourceItem] = Field(
        ..., description="Retrieved chunks used as context for the answer."
    )
    total_sources: int = Field(..., description="Number of source chunks retrieved.")
