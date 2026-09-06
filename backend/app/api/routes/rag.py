"""
RAG API — retrieval-augmented generation endpoint.

POST /api/rag
  Accepts: JSON body with query and optional top_k
  Returns: RAGResponse JSON with LLM-generated answer and source chunks
"""

import logging

from fastapi import APIRouter, status

from app.schemas.rag import RAGRequest, RAGResponse, RAGSourceItem
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/rag",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask your memory (RAG)",
    description=(
        "Answer a natural-language question using Retrieval-Augmented Generation. "
        "The query is embedded, matched against stored document chunks, and an LLM "
        "generates a grounded answer based on the retrieved context."
    ),
    tags=["RAG"],
)
async def rag_query(body: RAGRequest) -> RAGResponse:
    """Run the full RAG pipeline and return a grounded answer."""

    logger.info("RAG request: query=%r  top_k=%d", body.query, body.top_k)

    service = RAGService()
    result = service.answer_question(question=body.query, top_k=body.top_k)

    sources = [
        RAGSourceItem(
            id=s["id"],
            text=s["text"],
            metadata=s.get("metadata", {}),
            distance=s.get("distance"),
        )
        for s in result.get("sources", [])
    ]

    return RAGResponse(
        success=True,
        query=result["question"],
        answer=result["answer"],
        sources=sources,
        total_sources=result["total_sources"],
    )
