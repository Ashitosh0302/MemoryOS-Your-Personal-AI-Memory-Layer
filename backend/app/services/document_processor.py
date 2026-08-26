"""
Document processing service.

Supports PDF (via PyMuPDF) and TXT files only.
Extracts text, preserves metadata, enforces size limits.
"""

import re
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF

from app.core.config import settings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocumentMetadata:
    filename: str
    file_type: str   # "pdf" | "txt"
    size: int        # bytes on disk
    pages: int       # number of pages (TXT always = 1)
    text_length: int # character count of extracted text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(filename: str) -> str:
    """
    Strip directory traversal characters and limit to safe characters.
    Keeps the original stem + extension, replaces everything else with '_'.
    """
    # Take only the basename (no path components)
    name = Path(filename).name
    # Replace anything that is not alphanumeric, dash, underscore, or dot
    name = re.sub(r"[^\w.\-]", "_", name)
    # Collapse multiple consecutive underscores / dots
    name = re.sub(r"_+", "_", name)
    return name or "upload"


def validate_file(filename: str, size: int) -> str:
    """
    Validate extension and size.

    Returns the lowercase extension on success.
    Raises ValueError with a descriptive message on failure.
    """
    ext = Path(filename).suffix.lstrip(".").lower()

    if ext not in settings.allowed_extensions_set:
        allowed = ", ".join(sorted(settings.allowed_extensions_set)).upper()
        raise ValueError(
            f"Unsupported file type '.{ext}'. Allowed types: {allowed}."
        )

    if size > settings.max_upload_bytes:
        raise ValueError(
            f"File size {size / (1024 * 1024):.1f} MB exceeds the "
            f"{settings.MAX_UPLOAD_SIZE_MB} MB limit."
        )

    return ext


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    """
    Extract text from a PDF using PyMuPDF.

    Returns (full_text, page_count).
    Raises ValueError if the file cannot be opened as a PDF.
    """
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF file: {exc}") from exc

    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        raise ValueError("PDF contains no pages.")

    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text())

    doc.close()

    full_text = "\n".join(parts)
    if not full_text.strip():
        raise ValueError("No readable text could be extracted from this PDF.")

    return full_text, page_count


def _extract_txt_text(path: Path) -> tuple[str, int]:
    """
    Read a plain-text file as UTF-8.

    Returns (full_text, 1).
    Raises ValueError if the file cannot be decoded.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback: try latin-1 which never fails
        try:
            content = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise ValueError(f"Cannot read text file: {exc}") from exc

    if not content.strip():
        raise ValueError("Text file is empty or contains only whitespace.")

    return content, 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_document(saved_path: Path, original_filename: str) -> DocumentMetadata:
    """
    Extract text from *saved_path* and return metadata.

    Parameters
    ----------
    saved_path : Path
        Absolute path to the already-saved file.
    original_filename : str
        The sanitized filename (used for metadata only).

    Returns
    -------
    DocumentMetadata
    """
    ext = Path(original_filename).suffix.lstrip(".").lower()
    size = saved_path.stat().st_size

    if ext == "pdf":
        text, pages = _extract_pdf_text(saved_path)
    elif ext == "txt":
        text, pages = _extract_txt_text(saved_path)
    else:
        raise ValueError(f"Unexpected file type: .{ext}")

    return DocumentMetadata(
        filename=original_filename,
        file_type=ext,
        size=size,
        pages=pages,
        text_length=len(text),
    )
