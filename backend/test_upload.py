"""
Test suite for MemoryOS document upload API.
Run with: python test_upload.py
"""
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"


def multipart_upload(url: str, filepath: str, field: str = "file") -> tuple:
    """Upload a file using multipart/form-data. Returns (status_code, response_dict)."""
    boundary = b"----MemoryOSTestBoundary"
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_data = f.read()
    body = (
        b"--" + boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode()
        + b"\r\n"
        + b"Content-Type: application/octet-stream\r\n\r\n"
        + file_data
        + b"\r\n"
        + b"--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


results = []

# ── Test 1: Health ──────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Health Check")
try:
    r = urllib.request.urlopen(f"{BASE}/api/health")
    data = json.loads(r.read())
    ok = data.get("status") == "healthy"
    print(PASS if ok else FAIL, "status =", data)
    results.append(("Health check", ok))
except Exception as e:
    print(FAIL, e)
    results.append(("Health check", False))

# ── Test 2: TXT upload ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: TXT Upload")
with tempfile.NamedTemporaryFile(
    suffix=".txt", delete=False, mode="w", encoding="utf-8"
) as f:
    f.write("Hello MemoryOS!\nThis is a plain-text test document.\nIt has three lines.")
    txt_path = f.name

try:
    status, data = multipart_upload(f"{BASE}/api/documents/upload", txt_path)
    ok = status == 200 and data.get("success") is True and data.get("file_type") == "txt"
    print(PASS if ok else FAIL, f"HTTP {status}")
    print(json.dumps(data, indent=2))
    results.append(("TXT upload", ok))
finally:
    os.unlink(txt_path)

# ── Test 3: PDF upload ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 3: PDF Upload")
try:
    import fitz  # PyMuPDF

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "MemoryOS PDF test document.\nPage one content.\nSecond line.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two content. More text here for the test.")
    doc.save(pdf_path)
    doc.close()

    status, data = multipart_upload(f"{BASE}/api/documents/upload", pdf_path)
    ok = status == 200 and data.get("success") is True and data.get("file_type") == "pdf"
    print(PASS if ok else FAIL, f"HTTP {status}")
    print(json.dumps(data, indent=2))
    results.append(("PDF upload", ok))
finally:
    os.unlink(pdf_path)

# ── Test 4: Unsupported file type ───────────────────────────────────────────
print()
print("=" * 60)
print("TEST 4: Unsupported File Type (.jpg)")
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    f.write(b"fake image data")
    jpg_path = f.name

try:
    status, data = multipart_upload(f"{BASE}/api/documents/upload", jpg_path)
    ok = status == 422
    print(PASS if ok else FAIL, f"HTTP {status} (expected 422)")
    print("Detail:", data.get("detail", data))
    results.append(("Unsupported type rejected", ok))
finally:
    os.unlink(jpg_path)

# ── Test 5: Empty TXT file ──────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 5: Empty TXT File")
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
    f.write("   ")  # only whitespace
    empty_path = f.name

try:
    status, data = multipart_upload(f"{BASE}/api/documents/upload", empty_path)
    ok = status == 422
    print(PASS if ok else FAIL, f"HTTP {status} (expected 422)")
    print("Detail:", data.get("detail", data))
    results.append(("Empty file rejected", ok))
finally:
    os.unlink(empty_path)

# ── Test 6: File over 10 MB ─────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 6: File Over 10 MB")
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
    f.write(b"A" * (11 * 1024 * 1024))  # 11 MB
    big_path = f.name

try:
    status, data = multipart_upload(f"{BASE}/api/documents/upload", big_path)
    ok = status == 422
    print(PASS if ok else FAIL, f"HTTP {status} (expected 422)")
    print("Detail:", data.get("detail", data))
    results.append(("Oversized file rejected", ok))
finally:
    os.unlink(big_path)

# ── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("RESULTS SUMMARY")
all_pass = True
for name, passed in results:
    status_icon = "\u2705" if passed else "\u274c"
    print(f"  {status_icon}  {name}")
    if not passed:
        all_pass = False

print()
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
sys.exit(0 if all_pass else 1)
