"""
Test suite for MemoryOS document chunking service.
Run with: python test_chunker.py
"""

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
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunker import chunk_text

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


# ── Test 1: Normal text is split into multiple chunks ─────────────────────────
def test_normal_text_splits_into_multiple_chunks():
    sample_text = (
        "MemoryOS is an intelligent personal AI memory layer. "
        "It helps users capture, organize, and retrieve information effortlessly. "
        "By breaking large documents into structured chunks, MemoryOS ensures "
        "that semantic search and retrieval produce accurate results. "
        "Each chunk preserves contextual meaning while remaining within optimal "
        "token boundaries for language models and embedding generation."
    )
    chunk_size = 80
    chunk_overlap = 20

    chunks = chunk_text(sample_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert isinstance(chunks, list), f"Expected list, got {type(chunks)}"
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    for i, c in enumerate(chunks):
        assert len(c) <= chunk_size, f"Chunk {i} exceeded chunk_size: {len(c)} > {chunk_size}"
        assert len(c) > 0, f"Chunk {i} is empty"

    print(f"  Input length: {len(sample_text.strip())} chars")
    print(f"  Generated {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")


run_test("Normal text splits into multiple chunks", test_normal_text_splits_into_multiple_chunks)


# ── Test 2: Short text remains one chunk ──────────────────────────────────────
def test_short_text_remains_one_chunk():
    short_text = "Quick note: Meeting at 3 PM."
    chunk_size = 500
    chunk_overlap = 50

    chunks = chunk_text(short_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert isinstance(chunks, list), f"Expected list, got {type(chunks)}"
    assert len(chunks) == 1, f"Expected exactly 1 chunk, got {len(chunks)}"
    assert chunks[0] == short_text, f"Expected '{short_text}', got '{chunks[0]}'"

    print(f"  Input: '{short_text}'")
    print(f"  Output: 1 chunk of length {len(chunks[0])}")


run_test("Short text remains one chunk", test_short_text_remains_one_chunk)


# ── Test 3: Empty text is handled safely ──────────────────────────────────────
def test_empty_text_handled_safely():
    # Empty string
    chunks_empty = chunk_text("")
    assert chunks_empty == [], f"Expected [], got {chunks_empty}"

    # Whitespace only
    chunks_whitespace = chunk_text("   \n\t  \n   ")
    assert chunks_whitespace == [], f"Expected [], got {chunks_whitespace}"

    # Non-string input (None)
    chunks_none = chunk_text(None)  # type: ignore
    assert chunks_none == [], f"Expected [], got {chunks_none}"

    print("  Empty string -> []")
    print("  Whitespace string -> []")
    print("  None input -> []")


run_test("Empty and whitespace text handled safely", test_empty_text_handled_safely)


# ── Test 4: Chunk overlap works correctly ─────────────────────────────────────
def test_chunk_overlap_works():
    sample_text = (
        "Electrician: Ravi Kumar. Date: 12 August 2026. "
        "Electrical wiring - Rs 8,000. Fan installation - Rs 2,000. "
        "Switchboard replacement - Rs 1,500. Total estimated cost: Rs 11,500."
    )
    chunk_size = 60
    chunk_overlap = 15

    chunks = chunk_text(sample_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 2, f"Need at least 2 chunks to test overlap, got {len(chunks)}"

    for i in range(len(chunks) - 1):
        prev_tail = chunks[i][-chunk_overlap:]
        next_head = chunks[i + 1][:chunk_overlap]
        assert prev_tail == next_head, (
            f"Overlap mismatch between chunk {i} and {i+1}: "
            f"'{prev_tail}' != '{next_head}'"
        )

    print(f"  Verified exact {chunk_overlap}-character overlap across all {len(chunks)} chunks")


run_test("Chunk overlap works across consecutive chunks", test_chunk_overlap_works)


# ── Test 5: No important text is lost ─────────────────────────────────────────
def test_no_important_text_lost():
    sample_path = PROJECT_ROOT / "sample_data" / "electrician.txt"
    if sample_path.exists():
        text = sample_path.read_text(encoding="utf-8").strip()
    else:
        text = (
            "Electrician: Ravi Kumar\n"
            "Date: 12 August 2026\n\n"
            "Electrical wiring - ₹8,000\n"
            "Fan installation - ₹2,000\n"
            "Switchboard replacement - ₹1,500\n\n"
            "Total estimated cost: ₹11,500."
        )

    chunk_size = 50
    chunk_overlap = 15

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Reconstruct text from chunks by appending each non-overlapping suffix
    reconstructed = chunks[0]
    for c in chunks[1:]:
        reconstructed += c[chunk_overlap:]

    assert reconstructed == text, (
        "Reconstructed text does not match original text! Text was lost."
    )

    # Verify key phrases exist in the chunks
    key_phrases = [
        "Electrician: Ravi Kumar",
        "12 August 2026",
        "₹8,000",
        "₹2,000",
        "₹1,500",
        "₹11,500",
    ]
    for phrase in key_phrases:
        found = any(phrase in chunk for chunk in chunks) or phrase in reconstructed
        assert found, f"Key phrase '{phrase}' was lost during chunking!"

    print(f"  Reconstruction matched original {len(text)} characters exactly.")
    print("  All key phrases verified intact.")


run_test("No important text is lost (with reconstruction check)", test_no_important_text_lost)


# ── Test 6: Parameter validation & edge cases ─────────────────────────────────
def test_parameter_validation():
    sample = "Valid text for testing parameter exceptions."

    # chunk_size <= 0
    try:
        chunk_text(sample, chunk_size=0)
        assert False, "Should raise ValueError for chunk_size=0"
    except ValueError:
        pass

    # chunk_overlap < 0
    try:
        chunk_text(sample, chunk_size=100, chunk_overlap=-1)
        assert False, "Should raise ValueError for negative chunk_overlap"
    except ValueError:
        pass

    # chunk_overlap >= chunk_size
    try:
        chunk_text(sample, chunk_size=50, chunk_overlap=50)
        assert False, "Should raise ValueError when chunk_overlap >= chunk_size"
    except ValueError:
        pass

    # chunk_overlap == 0 (zero overlap should work cleanly)
    chunks_no_overlap = chunk_text("ABCDEFGHIJ", chunk_size=4, chunk_overlap=0)
    assert chunks_no_overlap == ["ABCD", "EFGH", "IJ"], f"Unexpected: {chunks_no_overlap}"

    print("  Invalid chunk_size rejected as ValueError")
    print("  Negative chunk_overlap rejected as ValueError")
    print("  chunk_overlap >= chunk_size rejected as ValueError")
    print("  chunk_overlap=0 handled correctly")


run_test("Parameter validation and edge cases", test_parameter_validation)


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
