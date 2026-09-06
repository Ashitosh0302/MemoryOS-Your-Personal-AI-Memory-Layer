"""
Search API — semantic search endpoint.

POST /api/search
  Accepts: JSON body with query and optional top_k
  Returns: SearchResponse JSON with matching chunks
"""

import logging

from fastapi import APIRouter, status

from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services.embedding_service import get_embeddings
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search",
    description=(
        "Search stored document chunks using a natural-language query. "
        "The query is embedded and matched against the ChromaDB vector store."
    ),
    tags=["Search"],
)
async def search_documents(body: SearchRequest) -> SearchResponse:
    """Embed the query and return the most similar stored chunks."""

    query = body.query
    top_k = body.top_k

    logger.info("Search request: query=%r  top_k=%d", query, top_k)

    # 1. Generate embedding for the query
    query_embedding = get_embeddings([query])[0]

    # 2. Query the vector store
    store = get_vector_store()
    raw_results = store.query_similar(
        query_embedding=query_embedding,
        n_results=top_k,
    )

    # 3. Build response items
    items = [
        SearchResultItem(
            id=r["id"],
            text=r["text"],
            metadata=r.get("metadata", {}),
            distance=r.get("distance"),
        )
        for r in raw_results
    ]

    return SearchResponse(
        success=True,
        query=query,
        total_results=len(items),
        results=items,
    )
