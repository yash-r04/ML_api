from fastapi import FastAPI, UploadFile, File
from sklearn.ensemble import IsolationForest
import io
import pdfplumber
import camelot
import tempfile
import pandas as pd

app = FastAPI(title="shakuntala")


def read_excel(content: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content))

def read_pdf(content: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Try structured tables first
    tables = camelot.read_pdf(tmp_path, pages="all")
    if len(tables) > 0:
        return tables[0].df

    # Fallback: extract raw text
    rows = []
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            rows.extend(page.extract_text().split("\n"))
    return pd.DataFrame({"raw_text": rows})

def detect_anomalies(df: pd.DataFrame):
    # Pick correct money/amount column
    if "amount" in df.columns:
        col = "amount"
    elif "money" in df.columns:
        col = "money"
    else:
        return df, []

    df[col] = pd.to_numeric(df[col], errors="coerce")

    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[[col]])

    anomalies = df[df["anomaly_score"] == -1]
    return df, anomalies

#endpoint 1: it is to find analomy in the uploaded pdf
@app.post("/find_anomaly/")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()

    # Detect file type
    if file.filename.endswith((".xlsx", ".xls")):
        df = read_excel(contents)
    elif file.filename.endswith(".pdf"):
        df = read_pdf(contents)
    else:
        return {"error": "Unsupported file type. Please upload Excel or PDF."}

    df, anomalies = detect_anomalies(df)

    anomalies_json = anomalies.to_dict(orient="records")

    return {
        "filename": file.filename,
        "total_records": len(df),
        "anomalies_detected": len(anomalies),
        "flagged_transactions": anomalies_json
    }
