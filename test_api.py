"""
Quick integration tests for the shakun API.
Run:  python test_api.py
Needs: pip install requests openpyxl
"""

import io, json, requests, sys
import pandas as pd

BASE = "http://127.0.0.1:8000"

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_excel_bytes(rows=50) -> bytes:
    """Generate a synthetic transactions Excel file in memory."""
    import random, datetime
    random.seed(42)
    base = datetime.date(2024, 1, 1)
    data = {
        "date":        [(base + datetime.timedelta(days=i)).isoformat() for i in range(rows)],
        "description": [f"Vendor {chr(65 + i % 26)}" for i in range(rows)],
        "amount":      [round(random.gauss(5000, 800), 2) for _ in range(rows)],
        "category":    [random.choice(["ops", "payroll", "infra", "misc"]) for _ in range(rows)],
    }
    # Inject obvious outliers
    data["amount"][10] = 99999.00
    data["amount"][30] = -50000.00

    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_health():
    print("\n[1] Health check")
    r = requests.get(BASE + "/")
    check("GET / returns 200", r.status_code == 200)

def test_upload():
    print("\n[2] Upload Excel")
    xlsx = make_excel_bytes()
    r = requests.post(BASE + "/upload/", files={"file": ("transactions.xlsx", xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    ok = r.status_code == 200
    check("POST /upload/ returns 200", ok, r.text[:120])
    if ok:
        d = r.json()
        check("rows == 50", d["rows"] == 50)
        check("amount_column detected", d["amount_column"] == "amount")
        check("anomalies_detected >= 1", d["anomalies_detected"] >= 1, f"got {d['anomalies_detected']}")
    return ok

def test_anomalies():
    print("\n[3] Anomalies endpoint")
    r = requests.get(BASE + "/anomalies/")
    ok = r.status_code == 200
    check("GET /anomalies/ returns 200", ok)
    if ok:
        d = r.json()
        check("response has 'anomalies' key", "anomalies" in d)
        check("count > 0", d["count"] > 0, f"count={d['count']}")

def test_summary():
    print("\n[4] Summary endpoint")
    r = requests.get(BASE + "/summary/")
    ok = r.status_code == 200
    check("GET /summary/ returns 200", ok)
    if ok:
        d = r.json()
        check("has 'rows' key", "rows" in d)
        check("has 'describe' key", "describe" in d)

def test_visualise():
    print("\n[5] Visualise endpoint")
    r = requests.get(BASE + "/visualise")
    check("GET /visualise returns 200", r.status_code == 200)
    check("response is HTML", "text/html" in r.headers.get("content-type", ""))
    check("contains plotly", "plotly" in r.text.lower())

def test_chat():
    print("\n[6] Chat endpoint")
    payload = {"query": "What is the total amount?"}
    r = requests.post(BASE + "/chat/", json=payload)
    ok = r.status_code == 200
    check("POST /chat/ returns 200", ok, r.text[:120])
    if ok:
        d = r.json()
        check("has 'answer' key", "answer" in d)
        check("answer is not empty", bool(d.get("answer")))

def test_bad_upload():
    print("\n[7] Bad file type rejected")
    r = requests.post(BASE + "/upload/", files={"file": ("notes.txt", b"hello", "text/plain")})
    check("unsupported file returns 400", r.status_code == 400)

def test_no_file_yet():
    """Simulate cold-start — anomalies before upload should 400."""
    # Can't easily test this after upload without restarting the server,
    # so just verify the endpoint is reachable.
    print("\n[8] Endpoints reachable (smoke)")
    for path in ["/anomalies/", "/summary/"]:
        r = requests.get(BASE + path)
        check(f"GET {path} reachable (200 or 400)", r.status_code in (200, 400))


if __name__ == "__main__":
    print(f"Testing against {BASE}")
    try:
        requests.get(BASE + "/", timeout=3)
    except Exception:
        print(f"\n❌  Cannot reach {BASE}. Is the server running?\n   uvicorn main:app --reload")
        sys.exit(1)

    test_health()
    uploaded = test_upload()
    if uploaded:
        test_anomalies()
        test_summary()
        test_visualise()
        test_chat()
    else:
        print("  ⚠  Skipping downstream tests (upload failed).")
    test_bad_upload()
    test_no_file_yet()

    print("\nDone.\n")