"""
Test suite for MemoryOS Semantic Search API (POST /api/search).
Run with: python test_search.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure standard output supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend directory is in sys.path so app modules can be imported
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.services.embedding_service import get_embeddings
from app.services.vector_store import VectorStoreService

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
# Test setup fixture using temporary database
# ---------------------------------------------------------------------------

TEST_DIR = Path(tempfile.mkdtemp(prefix="memoryos_search_test_"))


def setup_test_store():
    """Create a vector store in an isolated temp directory populated with sample documents."""
    store = VectorStoreService(db_dir=str(TEST_DIR))

    test_docs = [
        {
            "id": "doc1_chunk1",
            "text": "An electrician charges between $50 to $100 per hour for residential wiring and outlet installation.",
            "metadata": {"source": "electrician.txt", "chunk_index": 0},
        },
        {
            "id": "doc1_chunk2",
            "text": "Safety regulations require circuit breakers to be properly rated for the circuit amperage.",
            "metadata": {"source": "electrician.txt", "chunk_index": 1},
        },
        {
            "id": "doc2_chunk1",
            "text": "Python is a high-level programming language known for readable syntax and rich ecosystem.",
            "metadata": {"source": "python_guide.pdf", "chunk_index": 0},
        },
        {
            "id": "doc3_chunk1",
            "text": "FastAPI is a modern web framework for building REST APIs with Python 3.8+ based on standard type hints.",
            "metadata": {"source": "fastapi.md", "chunk_index": 0},
        },
    ]

    texts = [d["text"] for d in test_docs]
    embeddings = get_embeddings(texts)
    ids = [d["id"] for d in test_docs]
    metadatas = [d["metadata"] for d in test_docs]

    store.add_chunks(
        ids=ids,
        embeddings=embeddings,
        chunks=texts,
        metadatas=metadatas,
    )
    return store


def cleanup_test_dir():
    """Remove temporary directory after tests."""
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_search_valid_query():
    """Test POST /api/search with valid query returns matching chunks."""
    payload = {
        "query": "How much does an electrician cost for wiring?",
        "top_k": 3,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data.get("success") is True, f"Expected success=True, got {data}"
    assert data.get("query") == payload["query"]
    assert "total_results" in data
    assert isinstance(data.get("results"), list)

    results_list = data["results"]
    assert len(results_list) > 0, "Expected non-empty search results"
    assert len(results_list) <= payload["top_k"]

    # Verify fields in each result item
    first_result = results_list[0]
    assert "id" in first_result
    assert "text" in first_result
    assert "metadata" in first_result
    assert "distance" in first_result


def test_search_relevance():
    """Test semantic search returns most relevant document first."""
    payload = {
        "query": "electrician hourly rate outlet installation",
        "top_k": 2,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200

    data = response.json()
    results_list = data.get("results", [])
    assert len(results_list) > 0

    # The top result should be the electrician chunk
    top_text = results_list[0]["text"]
    assert "electrician" in top_text.lower() or "$50" in top_text


def test_search_empty_query_validation():
    """Test that empty query returns HTTP 422 Unprocessable Entity."""
    payload = {"query": "", "top_k": 5}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_search_whitespace_query_validation():
    """Test that whitespace query returns HTTP 422 Unprocessable Entity."""
    payload = {"query": "   \n\t   ", "top_k": 5}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"


def test_search_invalid_top_k():
    """Test that non-positive top_k returns HTTP 422 Unprocessable Entity."""
    payload = {"query": "electrician", "top_k": 0}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422, f"Expected 422 for top_k=0, got {response.status_code}"

    payload = {"query": "electrician", "top_k": -5}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 422, f"Expected 422 for top_k=-5, got {response.status_code}"


def test_search_default_top_k():
    """Test that omitting top_k defaults to top_k=5."""
    payload = {"query": "python web framework"}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data.get("success") is True
    assert len(data.get("results", [])) <= 5


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MEMORYOS STEP 5D — SEMANTIC SEARCH API TEST SUITE")
    print("=" * 60)
    print()

    # Populate temporary vector store for isolation
    print("Initializing test vector store...")
    test_store = setup_test_store()
    print("Test store ready.\n")

    # Patch global vector store instance to use test store
    import app.services.vector_store as vs_module
    original_store = vs_module._default_store
    vs_module._default_store = test_store

    try:
        run_test("Valid Query Search", test_search_valid_query)
        run_test("Semantic Relevance", test_search_relevance)
        run_test("Empty Query Validation", test_search_empty_query_validation)
        run_test("Whitespace Query Validation", test_search_whitespace_query_validation)
        run_test("Invalid top_k Validation", test_search_invalid_top_k)
        run_test("Default top_k Parameter", test_search_default_top_k)

    finally:
        # Restore original store instance & cleanup
        vs_module._default_store = original_store
        cleanup_test_dir()

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
        print("\nAll Step 5D tests passed successfully!")


if __name__ == "__main__":
    main()
