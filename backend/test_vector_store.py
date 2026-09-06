"""
Test suite for MemoryOS ChromaDB vector store service.
Run with: python test_vector_store.py
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
from app.services.vector_store import VectorStoreService, get_vector_store

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


# Temporary directory helper for isolated test DBs
TEST_DIR = Path(tempfile.mkdtemp(prefix="memoryos_chroma_test_"))


def cleanup_test_dir():
    """Safely remove temporary directory after tests."""
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR, ignore_errors=True)
        except Exception:
            pass


# ── TEST 1: ChromaDB collection can be created successfully ──────────────────
def test_1_create_collection():
    db_path = TEST_DIR / "test1_db"
    store = VectorStoreService(db_dir=db_path, collection_name="test_collection")

    assert store.collection is not None, "Collection was not created"
    assert store.collection.name == "test_collection", (
        f"Collection name mismatch: got {store.collection.name}"
    )
    print("  Collection 'test_collection' initialized successfully")


run_test("TEST 1: A ChromaDB collection can be created successfully", test_1_create_collection)


# ── TEST 2: Chunks with embeddings can be added successfully ─────────────────
def test_2_add_chunks():
    db_path = TEST_DIR / "test2_db"
    store = VectorStoreService(db_dir=db_path, collection_name="memory_documents")

    chunks = [
        "Ravi Kumar - Senior Electrician with 10 years experience.",
        "Electrical wiring and circuit repair rate: $50/hour.",
    ]
    embeddings = get_embeddings(chunks)
    metadatas = [
        {"filename": "electrician.txt", "chunk_index": 0},
        {"filename": "electrician.txt", "chunk_index": 1},
    ]

    added_ids = store.add_chunks(chunks=chunks, embeddings=embeddings, metadatas=metadatas)

    assert isinstance(added_ids, list), "Expected list of IDs"
    assert len(added_ids) == 2, f"Expected 2 IDs, got {len(added_ids)}"
    print(f"  Added {len(added_ids)} chunks with IDs: {added_ids}")


run_test("TEST 2: Chunks with embeddings can be added successfully", test_2_add_chunks)


# ── TEST 3: Stored chunk count is correct ─────────────────────────────────────
def test_3_stored_chunk_count():
    db_path = TEST_DIR / "test3_db"
    store = VectorStoreService(db_dir=db_path, collection_name="memory_documents")

    assert store.get_count() == 0, f"Expected initial count 0, got {store.get_count()}"

    chunks = ["Chunk A", "Chunk B", "Chunk C"]
    embeddings = get_embeddings(chunks)

    store.add_chunks(chunks=chunks, embeddings=embeddings)
    count = store.get_count()

    assert count == 3, f"Expected count 3, got {count}"
    print(f"  Initial count: 0 -> After adding 3 chunks: {count}")


run_test("TEST 3: Stored chunk count is correct", test_3_stored_chunk_count)


# ── TEST 4: Similarity query returns stored chunks ───────────────────────────
def test_4_similarity_query():
    db_path = TEST_DIR / "test4_db"
    store = VectorStoreService(db_dir=db_path, collection_name="memory_documents")

    chunks = [
        "Electrician Ravi Kumar charges $50/hour for home electrical services.",
        "Plumber Anil Sharma handles pipe repair and water supply installation.",
        "Software developer builds React web applications and Python APIs.",
    ]
    embeddings = get_embeddings(chunks)
    store.add_chunks(chunks=chunks, embeddings=embeddings)

    query_chunk = ["How much does the electrician charge for wiring?"]
    query_emb = get_embeddings(query_chunk)[0]

    matches = store.query_similar(query_embedding=query_emb, n_results=2)

    assert isinstance(matches, list), f"Expected list, got {type(matches)}"
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"

    top_text = matches[0]["text"]
    assert "Electrician" in top_text or "Ravi" in top_text, (
        f"Top query match expected electrician info, got: '{top_text}'"
    )
    print(f"  Top query match: '{top_text[:60]}...'")
    print(f"  Returned {len(matches)} similar chunks successfully")


run_test("TEST 4: Similarity query returns stored chunks", test_4_similarity_query)


# ── TEST 5: Returned results contain original chunk text and metadata ────────
def test_5_results_contain_text_and_metadata():
    db_path = TEST_DIR / "test5_db"
    store = VectorStoreService(db_dir=db_path, collection_name="memory_documents")

    chunk_text_input = "Document processing with PyMuPDF extracts full text cleanly."
    embeddings = get_embeddings([chunk_text_input])
    meta_input = {"source": "sample.pdf", "page": 1, "author": "MemoryOS"}

    store.add_chunks(chunks=[chunk_text_input], embeddings=embeddings, metadatas=[meta_input])

    query_emb = get_embeddings(["pdf text extraction"])[0]
    results_list = store.query_similar(query_embedding=query_emb, n_results=1)

    assert len(results_list) == 1, "Expected 1 result"
    res = results_list[0]

    assert "id" in res and res["id"], "Result missing 'id'"
    assert "text" in res, "Result missing 'text'"
    assert res["text"] == chunk_text_input, f"Text mismatch: got '{res['text']}'"
    assert "metadata" in res, "Result missing 'metadata'"
    assert res["metadata"].get("source") == "sample.pdf", f"Metadata mismatch: {res['metadata']}"
    assert res["metadata"].get("page") == 1, f"Metadata mismatch: {res['metadata']}"
    assert "distance" in res, "Result missing 'distance'"

    print(f"  Result ID: {res['id']}")
    print(f"  Result text: '{res['text']}'")
    print(f"  Result metadata: {res['metadata']}")


run_test("TEST 5: Returned results contain original chunk text and metadata", test_5_results_contain_text_and_metadata)


# ── TEST 6: Empty database is handled safely ──────────────────────────────────
def test_6_empty_database_handled_safely():
    db_path = TEST_DIR / "test6_db"
    store = VectorStoreService(db_dir=db_path, collection_name="memory_documents")

    assert store.get_count() == 0, "Expected count 0 for empty database"

    # Add empty list should return []
    empty_ids = store.add_chunks(chunks=[], embeddings=[])
    assert empty_ids == [], f"Expected [], got {empty_ids}"

    # Query empty collection should return [] without crashing
    query_emb = [0.1] * 384
    query_res = store.query_similar(query_embedding=query_emb, n_results=5)

    assert isinstance(query_res, list), f"Expected list, got {type(query_res)}"
    assert len(query_res) == 0, f"Expected 0 results, got {len(query_res)}"
    print("  Empty list insertion returns []")
    print("  Querying empty collection returns [] safely")


run_test("TEST 6: Empty database is handled safely", test_6_empty_database_handled_safely)


# ── TEST 7: Data can be retrieved after creating a new vector-store instance ─
def test_7_persistence_across_instances():
    db_path = TEST_DIR / "test7_db"

    # First instance: add data
    store1 = VectorStoreService(db_dir=db_path, collection_name="memory_documents")
    chunks = ["Persistent chunk 1", "Persistent chunk 2"]
    embeddings = get_embeddings(chunks)
    metadatas = [{"tag": "persisted_1"}, {"tag": "persisted_2"}]
    store1.add_chunks(chunks=chunks, embeddings=embeddings, metadatas=metadatas)
    count1 = store1.get_count()
    assert count1 == 2, f"Instance 1 failed to store chunks, count={count1}"

    # Second instance pointing to same db_path: retrieve data
    store2 = VectorStoreService(db_dir=db_path, collection_name="memory_documents")
    count2 = store2.get_count()

    assert count2 == 2, (
        f"Instance 2 expected 2 chunks from persistent DB, got {count2}"
    )

    query_emb = get_embeddings(["Persistent chunk 1"])[0]
    res = store2.query_similar(query_embedding=query_emb, n_results=1)

    assert len(res) == 1, "Expected 1 result from store2"
    assert res[0]["text"] == "Persistent chunk 1", (
        f"Persistent text mismatch: got '{res[0]['text']}'"
    )
    assert res[0]["metadata"].get("tag") == "persisted_1", (
        f"Persistent metadata mismatch: {res[0]['metadata']}"
    )

    print(f"  Instance 1 count: {count1}")
    print(f"  Instance 2 count: {count2}")
    print(f"  Successfully retrieved persisted chunk: '{res[0]['text']}'")


run_test("TEST 7: Data can be retrieved after creating a new vector-store service instance using the same database directory", test_7_persistence_across_instances)


# Cleanup test temp directory
cleanup_test_dir()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
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
