"""
Test suite for MemoryOS RAG API endpoint (POST /api/rag).
All tests mock the RAGService so NO real LLM calls or vector store access occurs.

Run with: python test_rag_api.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure standard output supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def run_test(name: str, fn):
    """Helper to run a test function and log results."""
    print("=" * 60)
    print(f"TEST: {name}")
    try:
        fn()
        print(PASS, name)
        results.append((name, True))
    except AssertionError as err:
        print(FAIL, f"AssertionError: {err}")
        results.append((name, False))
    except Exception as exc:
        print(FAIL, f"Unexpected error: {exc}")
        results.append((name, False))
    print()


# ---------------------------------------------------------------------------
# Mock RAG service response
# ---------------------------------------------------------------------------

MOCK_RAG_RESULT = {
    "question": "How much does an electrician charge?",
    "answer": "An electrician charges between $50 to $100 per hour for residential wiring.",
    "sources": [
        {
            "id": "doc1_chunk1",
            "text": "An electrician charges between $50 to $100 per hour for residential wiring and outlet installation.",
            "metadata": {"source": "electrician.txt", "chunk_index": 0},
            "distance": 0.25,
        },
        {
            "id": "doc1_chunk2",
            "text": "Safety regulations require circuit breakers to be properly rated for the circuit amperage.",
            "metadata": {"source": "electrician.txt", "chunk_index": 1},
            "distance": 0.55,
        },
    ],
    "total_sources": 2,
    "prompt": "Retrieved Context:\n...",
}

MOCK_NO_SOURCES_RESULT = {
    "question": "What is the meaning of life?",
    "answer": "I could not find information about that in your stored memory.",
    "sources": [],
    "total_sources": 0,
    "prompt": None,
}

client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rag_valid_query():
    """Test POST /api/rag with valid query returns expected response shape."""
    with patch("app.api.routes.rag.RAGService") as MockClass:
        mock_instance = MagicMock()
        mock_instance.answer_question.return_value = MOCK_RAG_RESULT
        MockClass.return_value = mock_instance

        payload = {"query": "How much does an electrician charge?", "top_k": 3}
        response = client.post("/api/rag", json=payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["success"] is True
    assert data["query"] == MOCK_RAG_RESULT["question"]
    assert data["answer"] == MOCK_RAG_RESULT["answer"]
    assert isinstance(data["sources"], list)
    assert data["total_sources"] == 2

    # Verify each source has expected fields
    first_source = data["sources"][0]
    assert "id" in first_source
    assert "text" in first_source
    assert "metadata" in first_source
    assert "distance" in first_source

    # Verify RAGService was called correctly
    mock_instance.answer_question.assert_called_once_with(
        question="How much does an electrician charge?",
        top_k=3,
    )


def test_rag_default_top_k():
    """Test that omitting top_k defaults to 5."""
    with patch("app.api.routes.rag.RAGService") as MockClass:
        mock_instance = MagicMock()
        mock_instance.answer_question.return_value = MOCK_RAG_RESULT
        MockClass.return_value = mock_instance

        payload = {"query": "electrician costs"}
        response = client.post("/api/rag", json=payload)

    assert response.status_code == 200

    # Verify default top_k=5 was passed to the service
    mock_instance.answer_question.assert_called_once_with(
        question="electrician costs",
        top_k=5,
    )


def test_rag_empty_query_validation():
    """Test that empty query returns HTTP 422."""
    payload = {"query": "", "top_k": 5}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_rag_whitespace_query_validation():
    """Test that whitespace-only query returns HTTP 422."""
    payload = {"query": "   \n\t   ", "top_k": 5}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_rag_missing_query_validation():
    """Test that missing query field returns HTTP 422."""
    payload = {"top_k": 5}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_rag_invalid_top_k():
    """Test that non-positive top_k returns HTTP 422."""
    payload = {"query": "electrician", "top_k": 0}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422 for top_k=0, got {response.status_code}"

    payload = {"query": "electrician", "top_k": -3}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422 for top_k=-3, got {response.status_code}"


def test_rag_top_k_exceeds_max():
    """Test that top_k > 100 returns HTTP 422."""
    payload = {"query": "electrician", "top_k": 101}
    response = client.post("/api/rag", json=payload)
    assert response.status_code == 422, f"Expected 422 for top_k=101, got {response.status_code}"


def test_rag_no_sources_returned():
    """Test response when the vector store has no matching chunks."""
    with patch("app.api.routes.rag.RAGService") as MockClass:
        mock_instance = MagicMock()
        mock_instance.answer_question.return_value = MOCK_NO_SOURCES_RESULT
        MockClass.return_value = mock_instance

        payload = {"query": "What is the meaning of life?"}
        response = client.post("/api/rag", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_sources"] == 0
    assert data["sources"] == []
    assert "could not find" in data["answer"].lower()


def test_rag_service_exception():
    """Test that a RAGService exception returns HTTP 500."""
    # Use raise_server_exceptions=False so the global exception handler
    # returns a 500 JSON response instead of propagating the error.
    error_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.api.routes.rag.RAGService") as MockClass:
        mock_instance = MagicMock()
        mock_instance.answer_question.side_effect = RuntimeError("LLM API key is not configured.")
        MockClass.return_value = mock_instance

        payload = {"query": "electrician costs"}
        response = error_client.post("/api/rag", json=payload)

    assert response.status_code == 500, f"Expected 500, got {response.status_code}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MEMORYOS STEP 6B — RAG API TEST SUITE")
    print("=" * 60)
    print()

    run_test("Valid RAG Query", test_rag_valid_query)
    run_test("Default top_k Parameter", test_rag_default_top_k)
    run_test("Empty Query Validation", test_rag_empty_query_validation)
    run_test("Whitespace Query Validation", test_rag_whitespace_query_validation)
    run_test("Missing Query Validation", test_rag_missing_query_validation)
    run_test("Invalid top_k Validation", test_rag_invalid_top_k)
    run_test("top_k Exceeds Maximum", test_rag_top_k_exceeds_max)
    run_test("No Sources Returned", test_rag_no_sources_returned)
    run_test("RAGService Exception Handling", test_rag_service_exception)

    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, ok in results if ok)
    total_count = len(results)
    for name, ok in results:
        status_str = PASS if ok else FAIL
        print(f"  {status_str} {name}")
    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count < total_count:
        sys.exit(1)
    else:
        print("\nAll Step 6B tests passed successfully!")


if __name__ == "__main__":
    main()
