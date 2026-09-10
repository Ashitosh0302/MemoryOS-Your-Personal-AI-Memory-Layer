"""
Documents API — upload endpoint.

POST /api/documents/upload
  Accepts: multipart/form-data  (field name: "file")
  Returns: DocumentUploadResponse JSON
"""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.services.chunker import chunk_text
from app.services.document_processor import (
    process_document,
    sanitize_filename,
    validate_file,
)
from app.services.embedding_service import get_embeddings
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

# Directory where uploaded files are stored (relative to project root)
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data"


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class DocumentUploadResponse(BaseModel):
    success: bool
    filename: str
    file_type: str
    size: int
    pages: int
    text_length: int
    message: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a document",
    description=(
        "Upload a PDF or TXT file (max 10 MB). "
        "The file is saved to backend/data/, text is extracted, chunked, "
        "embedded, and stored in ChromaDB for RAG search. "
        "Only metadata is returned — never the full extracted text."
    ),
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or TXT file to upload"),
) -> DocumentUploadResponse:
    """Receive, validate, save, process, chunk, embed, and store an uploaded document."""

    # ── 1. Read file content into memory ──────────────────────────────────
    content = await file.read()
    size = len(content)

    # ── 2. Validate ────────────────────────────────────────────────────────
    original_name = file.filename or "upload"
    try:
        validate_file(original_name, size)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # ── 3. Sanitize filename ───────────────────────────────────────────────
    safe_name = sanitize_filename(original_name)

    # ── 4. Ensure upload directory exists ─────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5. Save to disk ────────────────────────────────────────────────────
    dest = UPLOAD_DIR / safe_name
    # If a file with the same name already exists, add a numeric suffix
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    dest.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", dest, size)

    # ── 6. Extract text & build metadata ──────────────────────────────────
    try:
        meta, extracted_text = process_document(dest, safe_name)
    except ValueError as exc:
        # Clean up saved file on extraction failure
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Unexpected error processing %s", dest)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # ── 7. Chunk text, generate embeddings & store in ChromaDB ─────────────
    try:
        chunks = chunk_text(extracted_text)
        if chunks:
            embeddings = get_embeddings(chunks)
            metadatas = [
                {"source": safe_name, "chunk_index": i}
                for i in range(len(chunks))
            ]
            store = get_vector_store()
            store.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Successfully indexed %d chunks in ChromaDB for %s", len(chunks), safe_name)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Error indexing document %s into ChromaDB", safe_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document text extracted, but indexing into vector store failed: {exc}",
        )

    return DocumentUploadResponse(
        success=True,
        filename=meta.filename,
        file_type=meta.file_type,
        size=meta.size,
        pages=meta.pages,
        text_length=meta.text_length,
        message="Document uploaded, text extracted, and indexed successfully.",
    )
