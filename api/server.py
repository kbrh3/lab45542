from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add the parent directory to sys.path to resolve the 'rag' module
BASEDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(BASEDIR, '..')))

from rag import init_pipeline, run_query_and_log, MISSING_EVIDENCE_MSG
from dotenv import load_dotenv

from agent.runner import run_agent
from agent.types import AgentResponse

# Local dev environment loading
try:
    dotenv_path = os.path.join(BASEDIR, '../.env') 
    load_dotenv(dotenv_path)
except:
    pass

app = FastAPI(title="CS 5542 Lab 4 RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://policypulse5542.streamlit.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key verification for backend
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")
async def verify_token(internal_api_key: str = Header(None)):
    if internal_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# Initialize once on server start
init_pipeline(data_dir="data")

class QueryIn(BaseModel):
    query_id: str
    question: str
    retrieval_mode: str = "mm"     # "mm" or "text_only"
    top_k: int = 8                # optional (you can wire this later)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/status", dependencies=[Depends(verify_token)])
def health():
    return {"status": "ok", "message": "RAG API working"}

@app.post("/query", dependencies=[Depends(verify_token)])
def query(q: QueryIn) -> Dict[str, Any]:
    # run_query_and_log expects a dict with query_id, question, gold_evidence_ids
    query_item = {"query_id": q.query_id, "question": q.question, "gold_evidence_ids": []}

    print(f"--- DEBUG: api/server.py -> query() called. Question: {q.question}")
    
    out = run_query_and_log(query_item, retrieval_mode=q.retrieval_mode)

    print(f"--- DEBUG: api/server.py -> query() got result with keys: {list(out.keys())}")

    # Return everything Streamlit needs
    return {
        "pipeline_version": "snowflake_only_v2",
        "retriever_source": "snowflake_retriever_v2",
        "retriever_database": os.getenv("SNOWFLAKE_DATABASE", "UNKNOWN"),
        "retriever_schema": os.getenv("SNOWFLAKE_SCHEMA", "UNKNOWN"),
        "retriever_table": "BILLS",
        "dynamic_sql_from_bills": True,
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

class AgentQueryIn(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    max_steps: int = 5

@app.post("/agent_query", dependencies=[Depends(verify_token)])
def agent_query(q: AgentQueryIn) -> AgentResponse:
    try:
        return run_agent(user_message=q.message, history=q.history, max_steps=q.max_steps)
    except Exception as e:
        # Unhandled fatal error fallback to prevent crash
        return {
            "answer": f"Agent crashed unexpectedly: {str(e)}",
            "evidence": [],
            "metrics": {"agent_ok": False, "error_type": "fatal"},
            "missing_evidence_msg": MISSING_EVIDENCE_MSG,
            "tool_trace": [],
            "errors": [str(e)]
        }
