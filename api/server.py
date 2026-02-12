from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
from rag.pipeline import init_pipeline, run_query_and_log, MISSING_EVIDENCE_MSG
from dotenv import load_dotenv

# Local dev environment loading
try:
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    dotenv_path = os.path.join(BASEDIR, '../.env') 
    load_dotenv(dotenv_path)
except:
    pass

app = FastAPI(title="CS 5542 Lab 4 RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lab45542-frontend-streamlit.onrender.com/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key verification for backend
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
async def verify_token(internal_api_key: str = Header(None)):
    if internal_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# Initialize once on server start
init_pipeline(data_dir="data", logs_dir="logs", log_file="query_metrics.csv")

class QueryIn(BaseModel):
    query_id: str
    question: str
    retrieval_mode: str = "mm"     # "mm" or "text_only"
    top_k: int = 8                # optional (you can wire this later)

@app.get("/status", dependencies=[Depends(verify_token)])
def health():
    return {"status": "ok", "message": "RAG API working"}

@app.post("/query", dependencies=[Depends(verify_token)])
def query(q: QueryIn) -> Dict[str, Any]:
    # run_query_and_log expects a dict with query_id, question, gold_evidence_ids
    query_item = {"query_id": q.query_id, "question": q.question, "gold_evidence_ids": []}

    out = run_query_and_log(query_item, retrieval_mode=q.retrieval_mode)

    # Return everything Streamlit needs
    return {
        "answer": out["answer"],
        "evidence": out["ctx"]["evidence"],
        "metrics": {
            "latency_ms": out["latency_ms"],
            "p5": out["p5"],
            "r10": out["r10"],
            "faithful": out["faithful"],
            "missing_evidence_behavior": out["meb"],
        },
        "missing_evidence_msg": MISSING_EVIDENCE_MSG,
    }
