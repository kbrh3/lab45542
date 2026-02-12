from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware

from rag.pipeline import init_pipeline, run_query_and_log, MISSING_EVIDENCE_MSG

app = FastAPI(title="CS 5542 Lab 4 RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lab45542-frontend-streamlit.onrender.com/"], # Update after deploying frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize once on server start
init_pipeline(data_dir="data", logs_dir="logs", log_file="query_metrics.csv")

class QueryIn(BaseModel):
    query_id: str
    question: str
    retrieval_mode: str = "mm"     # "mm" or "text_only"
    top_k: int = 8                # optional (you can wire this later)

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG API working"}

@app.post("/query")
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
