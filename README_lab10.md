PolicyPulse - Lab 10
Project Description

PolicyPulse is a Retrieval Augmented Generation (RAG) web application designed to extract and reason over legislative insights. It has evolved into a production-ready system utilizing a decoupled architecture with a FastAPI backend and a Streamlit frontend.

The system exclusively leverages a Snowflake data warehouse for enterprise-level legislative context, enabling scalable querying over large structured datasets. It also features an autonomous Agent Mode powered by the Gemini API, allowing multi-step reasoning and dynamic tool execution.

System Architecture

The system follows a two-tier architecture:

Frontend (Streamlit)
Interactive UI for querying and viewing results.
Backend (FastAPI)
Handles API requests, retrieval orchestration, and agent execution.
RAG Pipeline
Snowflake-based retrieval (SQL + keyword matching)
Context construction
Grounded answer generation with evidence filtering
Agent Layer (Gemini)
Multi-step reasoning loop
Tool execution (RAG retrieval, SQL queries, summarization)
Structured outputs with tool traces
Data Source
Snowflake warehouse (POLICYPULSE_DB.PUBLIC.BILLS)
Data Flow

User → Streamlit → FastAPI → RAG / Agent → Snowflake → Response → UI

Setup Instructions

Prerequisites: Python 3.12.10

1. Clone and set up environment
git clone https://github.com/kbrh3/lab45542.git
cd lab45542

python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\activate
2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
3. Configure Environment Variables
cp .env.example .env

Edit .env:

BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=your_internal_secret_key

# Gemini API (required for Agent Mode)
GEMINI_API_KEY=your_gemini_api_key

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PAT=your_programmatic_access_token
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=POLICYPULSE_DB
SNOWFLAKE_SCHEMA=PUBLIC

# Optional evaluation
ENABLE_ERAG_EVAL=true

Note: Snowflake requires a Programmatic Access Token (PAT), not a password.

How to Run
Backend (Terminal 1)
uvicorn api.server:app --host 0.0.0.0 --port 8000

Check:

http://localhost:8000/health
Frontend (Terminal 2)
streamlit run app/main.py

Open:

http://localhost:8501
Deployment
Backend (Render):
https://policypulse-s1i6.onrender.com
Frontend (Streamlit):
(Add your deployed link if applicable)

The system is fully cloud-deployed and does not require local Snowflake setup for usage.

Dataset

The system queries legislative data from:

POLICYPULSE_DB.PUBLIC.BILLS

This includes:

bill numbers
titles
descriptions
status updates
committee assignments

All retrieval is performed dynamically via SQL (ILIKE-based keyword matching).

Example Queries
Standard RAG Queries
"What bills are related to education funding?"
"Summarize recent healthcare policies."
"Which bills are currently in committee?"
Agent Mode Queries
"Find education related bills and summarize their status."
"How many bills were passed in 2024 related to public safety?"
"Compare trends in healthcare legislation over time."
Agent Mode

Agent Mode enables multi-step reasoning using the Gemini API.

Features:
Dynamic tool selection (RAG + SQL)
Iterative reasoning (up to 5 steps)
Structured outputs:
answer
evidence
metrics
tool_trace
Example:

Query:

How many education bills were passed in 2024?

Agent:

Retrieves relevant bills
Filters by date
Aggregates results
Returns grounded answer
Evaluation

The system includes an eRAG evaluation suite that tracks:

Precision@K
Recall@K
Latency
Faithfulness
Missing-evidence behavior

Results are logged to:

artifacts/runs/<run_id>/query_metrics.csv
Recent Codebase Cleanup
Removed legacy PDF-based retrieval pipeline
Eliminated TF-IDF indexing system
Transitioned fully to Snowflake retrieval
Deleted root-level scratch/test files
Refactored test suite with mocking (no Snowflake dependency required)
Simplified pipeline state initialization
Known Limitations
Snowflake access requires valid PAT and network permissions
Gemini API requires active billing credits
Retrieval uses keyword matching (not embeddings)
No labeled ground truth dataset for precision evaluation
Agent performance depends on model availability and API limits
GitHub Structure
app/            → Streamlit frontend
api/            → FastAPI backend
rag/            → Retrieval pipeline
agent/          → Agent reasoning system
tests/          → Smoke tests
artifacts/      → Evaluation logs
Quick Start
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run app/main.py
Contributors
(Add your team names here)
Final Notes

PolicyPulse demonstrates a scalable, explainable AI system combining:

Retrieval Augmented Generation (RAG)
Enterprise data warehousing (Snowflake)
Multi step agent reasoning (Gemini)
