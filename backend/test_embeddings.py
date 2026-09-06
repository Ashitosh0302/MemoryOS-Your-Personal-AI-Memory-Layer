"""
Test suite for MemoryOS embedding service.
Run with: python test_embeddings.py
"""

import copy
import os
import sys
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

from app.services.embedding_service import get_embeddings, get_embedding_dimension

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


# ── Test 1: Normal list produces same number of embeddings ────────────────────
def test_normal_chunks_produce_embeddings():
    chunks = [
        "MemoryOS is a personal AI memory layer.",
        "It helps users capture and retrieve information.",
        "Documents are broken into chunks for semantic search.",
    ]
    embeddings = get_embeddings(chunks)

    assert isinstance(embeddings, list), f"Expected list, got {type(embeddings)}"
    assert len(embeddings) == len(chunks), (
        f"Expected {len(chunks)} embeddings, got {len(embeddings)}"
    )

    print(f"  Input: {len(chunks)} chunks")
    print(f"  Output: {len(embeddings)} embeddings")


run_test("Normal list produces same number of embeddings", test_normal_chunks_produce_embeddings)


# ── Test 2: Each embedding is a numeric vector ───────────────────────────────
def test_embeddings_are_numeric_vectors():
    chunks = ["The quick brown fox jumps over the lazy dog."]
    embeddings = get_embeddings(chunks)

    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"

    vec = embeddings[0]
    assert isinstance(vec, list), f"Expected list, got {type(vec)}"
    assert len(vec) > 0, "Embedding vector is empty"

    for i, val in enumerate(vec):
        assert isinstance(val, float), (
            f"Element {i} is {type(val).__name__}, expected float"
        )

    print(f"  Vector length: {len(vec)}")
    print(f"  Sample values: [{vec[0]:.6f}, {vec[1]:.6f}, ... {vec[-1]:.6f}]")


run_test("Each embedding is a numeric vector of floats", test_embeddings_are_numeric_vectors)


# ── Test 3: Embeddings have consistent dimension ─────────────────────────────
def test_consistent_embedding_dimension():
    chunks = [
        "Short text.",
        "A somewhat longer piece of text with more words and context to process.",
        "Electrician: Ravi Kumar. Date: 12 August 2026. Electrical wiring costs.",
        "x",  # single character
    ]
    embeddings = get_embeddings(chunks)
    expected_dim = get_embedding_dimension()

    dimensions = [len(vec) for vec in embeddings]
    for i, dim in enumerate(dimensions):
        assert dim == expected_dim, (
            f"Chunk {i} has dimension {dim}, expected {expected_dim}"
        )

    print(f"  Model dimension: {expected_dim}")
    print(f"  All {len(chunks)} embeddings match expected dimension")


run_test("Embeddings have consistent dimension", test_consistent_embedding_dimension)


# ── Test 4: Empty input is handled safely ─────────────────────────────────────
def test_empty_input_handled():
    embeddings = get_embeddings([])

    assert isinstance(embeddings, list), f"Expected list, got {type(embeddings)}"
    assert len(embeddings) == 0, f"Expected 0 embeddings, got {len(embeddings)}"

    print("  Empty list -> []")


run_test("Empty input is handled safely", test_empty_input_handled)


# ── Test 5: Different text produces different embeddings ──────────────────────
def test_different_text_different_embeddings():
    chunks = [
        "The weather is sunny and warm today.",
        "Quantum mechanics describes subatomic particle behavior.",
    ]
    embeddings = get_embeddings(chunks)

    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"

    # Compare element-by-element — vectors should NOT be identical
    identical = all(
        abs(a - b) < 1e-9
        for a, b in zip(embeddings[0], embeddings[1])
    )
    assert not identical, "Embeddings for semantically different text should differ"

    # Compute cosine similarity to show they are meaningfully different
    dot = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
    norm_a = sum(a * a for a in embeddings[0]) ** 0.5
    norm_b = sum(b * b for b in embeddings[1]) ** 0.5
    cosine_sim = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    assert cosine_sim < 0.99, (
        f"Cosine similarity {cosine_sim:.4f} is suspiciously high for different text"
    )

    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print("  Vectors are distinct as expected")


run_test("Different text produces different embeddings", test_different_text_different_embeddings)


# ── Test 6: Original chunks are not modified ──────────────────────────────────
def test_original_chunks_not_modified():
    original_chunks = [
        "MemoryOS helps you remember everything.",
        "Upload documents and search them later.",
        "Powered by local embeddings for privacy.",
    ]
    chunks_copy = copy.deepcopy(original_chunks)

    _ = get_embeddings(original_chunks)

    assert original_chunks == chunks_copy, (
        "The original chunks list was modified by get_embeddings!"
    )

    print("  Chunks before and after embedding are identical")


run_test("Original chunks are not modified", test_original_chunks_not_modified)


# ── Test 7: Type validation ───────────────────────────────────────────────────
def test_type_validation():
    # Non-list input
    try:
        get_embeddings("not a list")  # type: ignore
        assert False, "Should raise TypeError for non-list input"
    except TypeError:
        pass

    # List with non-string elements
    try:
        get_embeddings(["valid", 123, "also valid"])  # type: ignore
        assert False, "Should raise ValueError for non-string element"
    except ValueError:
        pass

    print("  Non-list input raises TypeError")
    print("  Non-string element raises ValueError")


run_test("Type validation rejects invalid input", test_type_validation)


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("RESULTS SUMMARY")
all_pass = True
for name, passed in results:
    status_icon = PASS if passed else FAIL
    print(f"  {status_icon}  {name}")
    if not passed:
        all_pass = False

print()
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
sys.exit(0 if all_pass else 1)
