from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.ensemble import IsolationForest
from fastapi.responses import HTMLResponse
import io,os, json,pdfplumber
import numpy as np
import faiss
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import camelot, tempfile
import pandas as pd
from typing import List
from transformers import pipeline
import plotly.express as px

app = FastAPI(title="shakuntala")

@app.on_event("startup")
def load_models():
    """Load all machine learning models once at startup."""
    print("Loading models...")
    MODELS["embedding"] = SentenceTransformer("all-MiniLM-L6-v2")
    MODELS["qa"] = pipeline("text2text-generation", model="google/flan-t5-small")
    print("Models loaded successfully.")

RAW_DF = None
vector_index = None
documents = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

FAISS_INDEX = None
DOCS: List[str] = []
RAW_DF = None  
def read_excel(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content))
    except Exception:
        pd.read_csv(io.BytesIO(content))

def read_pdf(content: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        try:
            tables = camelot.read_pdf(tmp_path, pages="all")
            if len(tables) > 0:
                return tables[0].df
        
        except Exception:
            pass
        
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

    # Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[[col]])

    df["anomaly_flag"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

    anomalies = df[df["anomaly_flag"] == 1]

    return df, anomalies

def df_to_chunks(df: pd.DataFrame) -> List[str]:
    chunks = []
    if df.empty:
        return chunks

    if list(df.columns) == ["raw_text"]:
        # PDF raw text case
        for _, row in df.iterrows():
            chunks.append(f"text: {row['raw_text']} | anomaly: 0")
    else:
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = str(row[col])
                parts.append(f"{col}: {val}")
            chunks.append(" | ".join(parts))
    return chunks

def build_faiss_index(chunks: List[str]):
    global FAISS_INDEX, DOCS
    embeddings = EMBEDDING_MODEL.encode(chunks, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    FAISS_INDEX = index
    DOCS = chunks.copy()
    
def search_index(query: str, top_k: int = 5):
    if FAISS_INDEX is None:
        raise RuntimeError("No index built yet.")
    q_emb = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = FAISS_INDEX.search(q_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(DOCS):
            results.append({"id": int(idx), "score": float(score), "text": DOCS[idx]})
    return results

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    
    
#--------------------end points, dont put functions there


#endpoint 1: it is to find analomy in the uploaded pdf
@app.post("/find_anomaly/")
async def analyze_file(file: UploadFile = File(...)):
    global RAW_DF
    contents = await file.read()

    if file.filename.endswith((".xlsx", ".xls")):
        df = read_excel(contents)
    elif file.filename.endswith(".pdf"):
        df = read_pdf(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload Excel or PDF.")

    df, anomalies = detect_anomalies(df)

    RAW_DF = df.copy()

    anomalies_json = anomalies.to_dict(orient="records")

    return {
        "filename": file.filename,
        "total_records": len(df),
        "anomalies_detected": len(anomalies),
        "flagged_transactions": anomalies_json
    }

    
@app.post("/upload/")
async def upload_and_index(file: UploadFile = File(...)):
    global RAW_DF, vector_index, documents

    contents = await file.read()

    # Parse Excel/PDF into DataFrame
    if file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    elif file.filename.endswith(".pdf"):
        df = read_pdf(contents)  # <-- your pdf reader function
    else:
        return {"error": "Unsupported file type"}

    RAW_DF = df.copy()

    # Build text chunks for RAG
    chunks = df.astype(str).agg(" | ".join, axis=1).tolist()
    documents = chunks

    # Build FAISS index
    build_faiss_index(chunks)   # this function updates global FAISS_INDEX, DOCS

    return {
        "status": "file indexed",
        "filename": file.filename,
        "rows": len(df),
        "total_chunks": len(chunks)
    }
    
class ChatRequest(BaseModel):
    query: str


@app.post("/chat/")
async def chat(req: ChatRequest):
    results = search_index(req.query, top_k=3)
    context = "\n".join([r["text"] for r in results])

    qa_pipeline = pipeline(
        "text2text-generation", 
        model="google/flan-t5-small" 
    )

    prompt = f"""
    Context: {context}

    Question: {req.query}

    Answer the question based on the context provided.
    """

    output = qa_pipeline(prompt, max_length=100)
    
    answer = output[0]['generated_text']


    return {"answer": answer, "sources": results}

@app.get("/visualise")
async def visualize():
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        raise HTTPException(status_code=400, detail="No file uploaded yet. Please upload first.")
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
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid numeric values in column.")
    if "date" in df.columns:
        x_axis = "date"
    elif "time" in df.columns:
        x_axis = "time"
    else:
        x_axis = df.index
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
    fig.update_layout(
        xaxis_title="Date" if isinstance(x_axis, str) else "Transaction Index",
        yaxis_title=col.capitalize(),
        template="plotly_white",
    )
    html = fig.to_html(full_html=True)
    return HTMLResponse(content=html)