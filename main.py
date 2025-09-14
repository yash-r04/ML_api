from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import io, tempfile, os
import pdfplumber, camelot

from sklearn.ensemble import IsolationForest
import plotly.express as px

from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

app = FastAPI(title="shakun")
MODEL_NAME = "google/tapas-small-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(MODEL_NAME)
model = TapasForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline("table-question-answering", model=model, tokenizer=tokenizer)

RAW_DF = None
ANOMALIES = None

def read_excel(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.BytesIO(content))

def read_pdf(content: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(tmp_path, pages="all")
        if len(tables) > 0:
            return tables[0].df

        rows = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                rows.extend(page.extract_text().split("\n"))
        return pd.DataFrame({"raw_text": rows})

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def detect_anomalies(df: pd.DataFrame):
    col = None
    for c in df.columns:
        if str(c).lower() in ["amount", "money", "value", "transaction", "amt"]:
            col = c
            break

    if col is None:
        df["anomaly_flag"] = 0
        return df, pd.DataFrame()

    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[[col]])
    df["anomaly_flag"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

    anomalies = df[df["anomaly_flag"] == 1]
    return df, anomalies

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global RAW_DF, ANOMALIES
    contents = await file.read()

    if file.filename.endswith((".xlsx", ".xls")):
        df = read_excel(contents)
    elif file.filename.endswith(".pdf"):
        df = read_pdf(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload Excel/PDF.")

    df, anomalies = detect_anomalies(df)

    RAW_DF = df.copy()
    ANOMALIES = anomalies.copy()

    return {
        "status": "file uploaded and analyzed",
        "rows": len(df),
        "columns": list(df.columns),
        "anomalies_detected": len(anomalies)
    }


@app.get("/anomalies/")
async def get_anomalies():
    global ANOMALIES
    if ANOMALIES is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")
    return ANOMALIES.to_dict(orient="records")


@app.get("/visualise")
async def visualize():
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")

    df = RAW_DF.copy().reset_index(drop=True)

    col = None
    for c in df.columns:
        if str(c).lower() in ["amount", "money", "value", "transaction", "amt"]:
            col = c
            break
    if col is None:
        raise HTTPException(status_code=400, detail="No numeric transaction column found.")

    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    x_axis = "date" if "date" in df.columns else df.index

    hover_cols = [c for c in df.columns if c not in ["anomaly_flag", "anomaly_score"]]
    fig = px.line(df, x=x_axis, y=col, title="Money Flow Over Time", hover_data=hover_cols)

    if "anomaly_flag" in df.columns:
        anomalies = df[df["anomaly_flag"] == 1]
        fig.add_scatter(
            x=anomalies[x_axis],
            y=anomalies[col],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            name="Anomalies",
            hovertext=["<br>".join([f"{c}: {row[c]}" for c in hover_cols]) for _, row in anomalies.iterrows()],
            hoverinfo="text",
        )

    fig.update_layout(template="plotly_white")
    html = fig.to_html(full_html=True)
    return HTMLResponse(content=html)


@app.post("/chat/")
async def chat(query: dict):
    global RAW_DF
    if RAW_DF is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")
    if "raw_text" in RAW_DF.columns:
        raise HTTPException(status_code=400, detail="PDF is text-only, no structured table.")

    question = query.get("query")
    if not question:
        raise HTTPException(status_code=400, detail="No query provided.")

    try:
        # 🔑 Convert everything to strings (important for TAPAS)
        df_str = RAW_DF.astype(str)

        answers = qa_pipeline(table=df_str, query=question)

        if isinstance(answers, list) and len(answers) > 0:
            best_answer = answers[0]["answer"]
            return {
                "query": question,
                "answer": best_answer,
                "all_answers": answers
            }
        else:
            return {"query": question, "answer": "No answer found."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


