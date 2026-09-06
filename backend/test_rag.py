"""
Test suite for MemoryOS RAG (Retrieval-Augmented Generation) Service.
Uses a MOCKED LLM to prevent network calls.

Run with: python test_rag.py
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

from app.services.embedding_service import get_embeddings
from app.services.vector_store import VectorStoreService
from app.services.rag_service import (
    RAGService,
    answer_question,
    build_rag_prompt,
    NO_INFO_FALLBACK,
)

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

TEST_DIR = Path(tempfile.mkdtemp(prefix="memoryos_rag_test_"))


def setup_test_store():
    """Create a vector store in an isolated temp directory populated with sample documents."""
    store = VectorStoreService(db_dir=str(TEST_DIR))

    test_docs = [
        {
            "id": "electrician_1",
            "text": "Electrician Ravi charges an hourly rate of $75 for residential wiring and outlet replacement.",
            "metadata": {"source": "electrician_rates.txt", "chunk_index": 0},
        },
        {
            "id": "electrician_2",
            "text": "For emergency service calls after 8 PM, the electrician fee is $120 per hour.",
            "metadata": {"source": "electrician_rates.txt", "chunk_index": 1},
        },
        {
            "id": "python_1",
            "text": "Python is a popular language for data science and AI applications.",
            "metadata": {"source": "python_overview.pdf", "chunk_index": 0},
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
# Mock LLM Client
# ---------------------------------------------------------------------------

class MockLLMRecorder:
    """Mock LLM client that records prompts passed to it and returns pre-set answers."""

    def __init__(self, response_text: str = "Mocked LLM Answer: Electrician charges $75/hr for residential wiring."):
        self.response_text = response_text
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.call_count = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response_text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rag_with_mocked_llm():
    """Test full RAG pipeline with a mocked LLM."""
    store = VectorStoreService(db_dir=str(TEST_DIR))
    mock_llm = MockLLMRecorder("Electrician Ravi charges $75 per hour for residential wiring.")

    service = RAGService(vector_store=store, llm_client=mock_llm)
    question = "What is the electrician hourly rate for residential wiring?"

    result = service.answer_question(question=question, top_k=2)

    # Assertions
    assert mock_llm.call_count == 1, "Expected exactly 1 LLM call"
    assert "MemoryOS" in mock_llm.last_system_prompt, "System prompt missing MemoryOS instructions"
    assert "Electrician Ravi charges an hourly rate" in mock_llm.last_user_prompt, "Context missing from LLM prompt"
    assert question in mock_llm.last_user_prompt, "Question missing from LLM prompt"

    assert result["question"] == question
    assert result["answer"] == "Electrician Ravi charges $75 per hour for residential wiring."
    assert len(result["sources"]) > 0, "Expected source chunks in RAG output"
    assert result["total_sources"] == len(result["sources"])
    assert result["sources"][0]["metadata"]["source"] == "electrician_rates.txt"


def test_rag_prompt_builder():
    """Test build_rag_prompt utility produces expected system and user prompts."""
    chunks = [
        {
            "id": "c1",
            "text": "Circuit breaker installation safety guide.",
            "metadata": {"source": "safety.txt", "chunk_index": 0},
        }
    ]
    sys_prompt, user_prompt = build_rag_prompt("How to install circuit breaker?", chunks)

    assert "Instructions:" in sys_prompt
    assert NO_INFO_FALLBACK in sys_prompt
    assert "Circuit breaker installation safety guide." in user_prompt
    assert "Source: safety.txt Chunk #1" in user_prompt
    assert "Question: How to install circuit breaker?" in user_prompt


def test_insufficient_context_handling():
    """Test behavior when querying an empty vector store (no matching chunks)."""
    empty_dir = Path(tempfile.mkdtemp(prefix="memoryos_empty_rag_"))
    try:
        empty_store = VectorStoreService(db_dir=str(empty_dir))
        mock_llm = MockLLMRecorder()
        service = RAGService(vector_store=empty_store, llm_client=mock_llm)

        result = service.answer_question(question="Where is my passport?", top_k=3)

        assert mock_llm.call_count == 0, "LLM should NOT be called when no context exists"
        assert result["answer"] == NO_INFO_FALLBACK
        assert result["sources"] == []
        assert result["total_sources"] == 0
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)


def test_rag_parameter_validation():
    """Test empty/whitespace questions raise ValueError."""
    service = RAGService(llm_client=MockLLMRecorder())

    try:
        service.answer_question(question="")
        assert False, "Expected ValueError for empty question"
    except ValueError:
        pass

    try:
        service.answer_question(question="    \t  \n ")
        assert False, "Expected ValueError for whitespace question"
    except ValueError:
        pass


def test_convenience_answer_question_function():
    """Test module-level convenience function answer_question with mock LLM."""
    store = VectorStoreService(db_dir=str(TEST_DIR))
    mock_llm = MockLLMRecorder("Python is popular for AI.")

    result = answer_question(
        question="What language is popular for AI?",
        top_k=1,
        vector_store=store,
        llm_client=mock_llm,
    )

    assert result["answer"] == "Python is popular for AI."
    assert len(result["sources"]) == 1


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MEMORYOS STEP 6A — RAG SERVICE TEST SUITE")
    print("=" * 60)
    print()

    print("Initializing test vector store...")
    setup_test_store()
    print("Test store ready.\n")

    try:
        run_test("RAG Pipeline with Mocked LLM", test_rag_with_mocked_llm)
        run_test("RAG Prompt Builder Format", test_rag_prompt_builder)
        run_test("Empty Vector Store / Not Found Fallback", test_insufficient_context_handling)
        run_test("Input Parameter Validation", test_rag_parameter_validation)
        run_test("Module-level answer_question Function", test_convenience_answer_question_function)

    finally:
        cleanup_test_dir()

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
        print("\nAll Step 6A RAG tests passed successfully!")


if __name__ == "__main__":
    main()
