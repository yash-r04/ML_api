import os
import io
import tempfile
import logging
import pandas as pd
import numpy as np
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import time
from dotenv import load_dotenv

# Check for camelot table extractor availability
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# Initialize Environment Variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("shakun")

app = FastAPI(
    title="shakun",
    description="Upload Excel/PDF financial data → anomaly detection + Hugging Face LLM querying",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    log.warning("HF_TOKEN is missing. Chat endpoints will fail until configured.")

# Global state dataframes to keep memory simplified for prototyping
RAW_DF = None
ANOMALIES = None

def find_amount_column(df: pd.DataFrame) -> str:
    """Dynamically identity the core financial transactional column."""
    keywords = ["amount", "money", "value", "amt", "debit", "credit", "price", "total"]
    for col in df.columns:
        if any(kw in str(col).lower() for kw in keywords):
            return col
    # Fallback to look for numeric formats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def extract_tables_from_pdf(file_path: str) -> pd.DataFrame:
    """Fallback engine using pdfplumber to safely extract data matrices."""
    all_data = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                all_data.extend(table)
    
    if not all_data:
        # Emergency string scrape if tables are not natively structured inside the vector tree
        text_dump = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_dump += (page.extract_text() or "") + "\n"
        return pd.DataFrame([{"raw_text": text_dump}])

    headers = all_data[0]
    # Handle potentially blank dynamic headers
    headers = [h if h else f"Col_{i}" for i, h in enumerate(headers)]
    rows = all_data[1:]
    
    # Filter rows matching headers length to prevent structural layout mismatching
    cleaned_rows = [r for r in rows if len(r) == len(headers)]
    return pd.DataFrame(cleaned_rows, columns=headers)

def find_category_column(df: pd.DataFrame):
    """Find a categorical grouping column to enable per-category anomaly detection."""
    for col in df.columns:
        if any(kw in str(col).lower() for kw in ["category", "type", "kind", "dept", "department"]):
            if df[col].dtype == object and df[col].nunique() < 50:
                return col
    return None

def run_iqr_anomaly_detection(df: pd.DataFrame, multiplier: float = 2.5) -> pd.DataFrame:
    """Per-category IQR anomaly detection.

    Using a global IQR on multi-category financial data causes massive false positives
    because different categories (e.g. Logistics ~$9K vs CapEx ~$420K) have completely
    different normal ranges. We group by category first, then apply IQR within each group.
    Falls back to global IQR if no category column exists.
    multiplier=2.5 is less aggressive than the classic 1.5 — appropriate for industrial
    ledgers where legitimate high-value transactions are common.
    """
    df_processed = df.copy()
    amt_col = find_amount_column(df_processed)

    if not amt_col:
        df_processed["anomaly_flag"] = 0
        df_processed["anomaly_score"] = 0.0
        return df_processed

    # Coerce to clean floats
    df_processed[amt_col] = pd.to_numeric(
        df_processed[amt_col].astype(str).str.replace(r"[^\d\.]", "", regex=True),
        errors="coerce"
    ).fillna(0.0)

    cat_col = find_category_column(df_processed)
    anomaly_flags = pd.Series(0, index=df_processed.index)
    anomaly_scores = pd.Series(0.0, index=df_processed.index)

    groups = df_processed.groupby(cat_col) if cat_col else [(None, df_processed)]

    for _, group in groups:
        q1 = group[amt_col].quantile(0.25)
        q3 = group[amt_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        is_anomaly = (group[amt_col] < lower_bound) | (group[amt_col] > upper_bound)
        anomaly_flags[group.index[is_anomaly]] = 1

        mean_val = group[amt_col].mean()
        std_dev = group[amt_col].std()
        if std_dev and std_dev > 0:
            scores = ((group[amt_col] - mean_val) / std_dev).abs().round(2)
        else:
            scores = pd.Series(0.0, index=group.index)
        anomaly_scores[group.index] = scores

    df_processed["anomaly_flag"] = anomaly_flags
    df_processed["anomaly_score"] = anomaly_scores
    return df_processed

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload Excel, CSV, or transactional PDF statements to isolate transaction metadata structures."""
    global RAW_DF, ANOMALIES
    filename = file.filename.lower()
    
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Parse step based on file signatures
        if filename.endswith(".csv"):
            RAW_DF = pd.read_csv(tmp_path)
        elif filename.endswith((".xls", ".xlsx")):
            RAW_DF = pd.read_excel(tmp_path)
        elif filename.endswith(".pdf"):
            if CAMELOT_AVAILABLE:
                try:
                    tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
                    if len(tables) > 0:
                        RAW_DF = pd.concat([t.df for t in tables], ignore_index=True)
                    else:
                        RAW_DF = extract_tables_from_pdf(tmp_path)
                except Exception:
                    RAW_DF = extract_tables_from_pdf(tmp_path)
            else:
                RAW_DF = extract_tables_from_pdf(tmp_path)
        else:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="Unsupported file format type extensions passed.")

        os.unlink(tmp_path)

        if RAW_DF is None or RAW_DF.empty:
            raise HTTPException(status_code=400, detail="The parser could not extract any structured rows.")

        # Strip spaces from column headers
        RAW_DF.columns = [str(c).strip() for c in RAW_DF.columns]

        # Execute Forensics
        RAW_DF = run_iqr_anomaly_detection(RAW_DF)
        ANOMALIES = RAW_DF[RAW_DF["anomaly_flag"] == 1]
        amt_col = find_amount_column(RAW_DF)  # Re-derive after processing so it's in scope

        # 🚨 KEY ALIGNMENT MATCH FOR FRONTEND HTML EXPECTATIONS
        return {
            "status": "Success",
            "filename": file.filename,
            "rows": len(RAW_DF),                  # Matches data.rows
            "anomalies_detected": len(ANOMALIES), # Matches data.anomalies_detected
            "columns": list(RAW_DF.columns),       # Matches data.columns
            "amount_column": amt_col               # Matches data.amount_column
        }

    except Exception as e:
        log.error(f"Upload Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File Processing Pipeline Aborted: {str(e)}")

@app.post("/chat/")
async def chat(query: dict):
    """Ask natural language questions about your transactions using a Hugging Face pipeline."""
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        raise HTTPException(status_code=400, detail="No data available. Please upload a file first.")

    question = query.get("query", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Provide a 'query' key with your question.")

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN configuration is missing on the server.")

    # 1. Capture core metrics safely
    total_rows = len(RAW_DF)
    amt_col = find_amount_column(RAW_DF)
    if amt_col:
        numeric_series = pd.to_numeric(
            RAW_DF[amt_col].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce"
        ).fillna(0.0)
        total_spend = round(numeric_series.sum(), 2)
    else:
        total_spend = 0

    df_anomalies = RAW_DF[RAW_DF["anomaly_flag"] == 1].copy() if "anomaly_flag" in RAW_DF.columns else pd.DataFrame()
    anomalies_count = len(df_anomalies)

    # 2. Build sample (Strictly keep the context matrix light)
    df_sample = RAW_DF.head(10).copy()  # Grab context baselines
    df_anomalies_capped = df_anomalies.head(15).copy()  # Grab an anomaly cohort sample
    df_combined = pd.concat([df_sample, df_anomalies_capped]).drop_duplicates().copy()

    # 🚨 HARD PROTECTION: Completely wipe heavy raw text vectors if fallback triggered
    if "raw_text" in df_combined.columns:
        df_combined = df_combined.drop(columns=["raw_text"], errors="ignore")

    # Filter columns to pass only core structured fields
    essential_keywords = {"amount", "money", "value", "amt", "debit", "credit", "price", 
                          "date", "timestamp", "time", "category", "type", "description", "id", "anomaly_flag"}
    
    columns_to_keep = [col for col in df_combined.columns if any(kw in str(col).lower() for kw in essential_keywords)]
    
    if columns_to_keep:
        df_combined = df_combined[columns_to_keep]
    else:
        df_combined = df_combined.iloc[:, :6]
        if "anomaly_flag" in RAW_DF.columns and "anomaly_flag" not in df_combined.columns:
            df_combined["anomaly_flag"] = RAW_DF["anomaly_flag"]

    # 🚨 CELL INTERCEPT: Chop single cell character limits to prevent wordy descriptions from swelling
    for col in df_combined.columns:
        if df_combined[col].dtype == object:
            df_combined[col] = df_combined[col].astype(str).str.slice(0, 40)

    # 🚨 HARD STRING BOUNDARY: Prevent multi-kilobyte strings from hitting the gateway
    markdown_table = df_combined.to_markdown(index=False)
    if len(markdown_table) > 15000:
        markdown_table = markdown_table[:15000] + "\n... [Table context clipped horizontally/vertically for size] ..."

    system_prompt = (
        "You are 'shakun AI', an expert financial forensic analyst. Use the provided metrics and "
        "condensed ledger slice to answer the user's question accurately. Be concise, insightful, "
        "and highly analytical. Focus heavily on rows flagged with anomaly_flag = 1."
    )
    
    user_context = (
        f"--- METRIC AGGREGATIONS ---\n"
        f"- Total Transactions in Database: {total_rows}\n"
        f"- Total Capital Throughput: ${total_spend:,}\n"
        f"- Critical Anomalies Isolated: {anomalies_count}\n\n"
        f"--- LEDGER SUBSET (Normal Baselines + Flagged Anomalies Sample) ---\n"
        f"{markdown_table}\n\n"
        f"User Question: {question}\n"
        f"Forensic Answer:"
    )

    try:
    
        client = InferenceClient(
            api_key=HF_TOKEN
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are shakun AI, an expert financial forensic analyst. "
                        "Analyze transaction anomalies and answer concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": user_context,
                },
            ],
            max_tokens=400,
            temperature=0.6,
        )

        answer_text = response.choices[0].message.content

        return {
            "query": question,
            "answer": answer_text,
            "aggregator": "Qwen3-32B"
        }

    except Exception as e:
        log.error(f"Hugging Face Chat Failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline exception: {str(e)}"
        )

@app.get("/visualise", response_class=HTMLResponse)   # Alias used by the dashboard JS
@app.get("/visualize/", response_class=HTMLResponse)
async def visualize():
    """Generates an interactive diagnostic layout visualizing transaction anomalies over time."""
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        return "<h3>No data uploaded yet. Please use /upload/ first.</h3>"

    amt_col = find_amount_column(RAW_DF)
    if not amt_col:
        return "<h3>Could not identify a valid numeric amount scale for visualization.</h3>"

    plot_df = RAW_DF.copy()
    # Ensure amount column is numeric
    plot_df[amt_col] = pd.to_numeric(
        plot_df[amt_col].astype(str).str.replace(r"[^\d\.]", "", regex=True), errors="coerce"
    ).fillna(0.0)

    # Find a date column if one exists for a meaningful X axis
    date_col = None
    for col in plot_df.columns:
        if any(kw in str(col).lower() for kw in ["date", "time", "timestamp"]):
            try:
                plot_df[col] = pd.to_datetime(plot_df[col], errors="coerce")
                if plot_df[col].notna().sum() > 0:
                    date_col = col
                    break
            except Exception:
                pass
    x_axis = plot_df[date_col] if date_col else plot_df.index

    # Build hover text with all available transaction fields
    def build_hover(row):
        lines = []
        for col in plot_df.columns:
            if col not in ["anomaly_flag", "anomaly_score"] and str(row[col]).strip() not in ("", "nan"):
                lines.append(f"<b>{col}:</b> {str(row[col])[:60]}")
        lines.append(f"<b>anomaly_score:</b> {row.get('anomaly_score', '')}")
        return "<br>".join(lines)

    plot_df["_hover"] = plot_df.apply(build_hover, axis=1)

    normal = plot_df[plot_df["anomaly_flag"] == 0]
    flagged = plot_df[plot_df["anomaly_flag"] == 1]

    fig = go.Figure()

    # Base line — all transactions
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=plot_df[amt_col],
        mode="lines",
        name="All Transactions",
        line=dict(color="#4f86f7", width=1.5),
        opacity=0.5,
        hoverinfo="skip",
    ))

    # Normal dots
    fig.add_trace(go.Scatter(
        x=(normal[date_col] if date_col else normal.index),  # series, not string
        y=normal[amt_col],
        mode="markers",
        name="Normal",
        marker=dict(color="#2ecf8a", size=5, opacity=0.7),
        hovertemplate=normal["_hover"] + "<extra>Normal</extra>",
    ))

    # Anomaly X markers
    fig.add_trace(go.Scatter(
        x=(flagged[date_col] if date_col else flagged.index),  # series, not string
        y=flagged[amt_col],
        mode="markers+text",
        name="Anomaly",
        marker=dict(color="#e8445a", size=12, symbol="x", line=dict(width=2, color="#e8445a")),
        text=["⚠"] * len(flagged),
        textposition="top center",
        hovertemplate=flagged["_hover"] + "<extra>⚠ ANOMALY</extra>",
    ))

    fig.update_layout(
        title="Transaction Flow — Anomalies Highlighted",
        xaxis_title=date_col if date_col else "Transaction Index",
        yaxis_title=amt_col,
        template="plotly_dark",
        plot_bgcolor="#0d0f14",
        paper_bgcolor="#161b27",
        font=dict(color="#e4e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )

    html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")
    return HTMLResponse(content=html_content)

@app.get("/")
async def root():
    return {"message": "Welcome to shakun AI financial forensic gateway. Endpoints active: /upload/, /chat/, /visualize/"}

@app.get("/anomalies/")
async def get_anomalies():
    """Returns isolated anomalous transaction records formatted for the UI table."""
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        return {"count": 0, "anomalies": []}
    
    df_anomalies = RAW_DF[RAW_DF["anomaly_flag"] == 1].copy()
    
    # Clean up native numeric types to standard serializable python formats
    if not df_anomalies.empty:
        df_anomalies = df_anomalies.fillna("")
        records = df_anomalies.to_dict(orient="records")
    else:
        records = []
        
    return {
        "count": len(records),
        "anomalies": records
    }

@app.get("/summary/")
async def get_summary():
    """Returns dataset structural profiling details for the summary view card."""
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    # Standardize data types dictionary string maps
    dtypes_map = {str(col): str(dtype) for col, dtype in RAW_DF.dtypes.items()}

    # Include descriptive statistics for numeric columns so the Summary card is actually useful
    numeric_cols = RAW_DF.select_dtypes(include=[np.number]).columns.tolist()
    describe_map = {}
    if numeric_cols:
        desc = RAW_DF[numeric_cols].describe().round(2)
        describe_map = desc.to_dict()

    return {"dtypes": dtypes_map, "describe": describe_map}