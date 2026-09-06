"""
Search schemas for MemoryOS semantic search API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """
    Semantic search request schema.
    """

    query: str = Field(
        ...,
        description="The text query string to search for.",
        examples=["electrician wiring costs"],
    )
    top_k: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Maximum number of relevant chunks to return (1-100).",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string must not be empty or whitespace.")
        return v.strip()


class SearchResultItem(BaseModel):
    """
    Individual chunk search result schema.
    """

    id: str = Field(..., description="Unique chunk identifier.")
    text: str = Field(..., description="Original chunk text content.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the chunk."
    )
    distance: Optional[float] = Field(
        default=None, description="Similarity distance score (lower is more similar)."
    )


class SearchResponse(BaseModel):
    """
    Semantic search response schema.
    """

    success: bool = Field(..., description="Whether the search succeeded.")
    query: str = Field(..., description="The original search query.")
    total_results: int = Field(..., description="Number of results returned.")
    results: List[SearchResultItem] = Field(
        ..., description="List of matching stored chunks."
    )
