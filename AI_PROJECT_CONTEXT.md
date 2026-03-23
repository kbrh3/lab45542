# Project Overview: Lab 45542 RAG Application

## 1. Description & Purpose
This project is a Retrieval-Augmented Generation (RAG) web application originally centered on extracting insights from local PDFs and image queries (multimodal reasoning). Its primary domains of instruction include SQL error detection, LLM hallucinations, and conceptual architecture extraction. 

Recently, the application has evolved into a fully production-ready system with three key advancements:
1. **Separation of Concerns**: Split into an independent Streamlit UI and a FastAPI backend.
2. **Data Source Refactoring**: Transitioned from purely local PDF extractions to utilizing **Snowflake** via a dedicated `snowflake_retriever.py` for enterprise-level context data retrieval. 
3. **Agentic Capabilities**: Integrates an autonomous "Agent Mode" powered by the Gemini API (`/agent_query`), providing an interface where the AI agent invokes multi-step tools, scores metrics, handles ambiguities, and maintains robust operational tracing.

Additionally, the project supports a deterministic evaluation suite (`eRAG`), calculating Precision@K retrieval scores and dynamically logging outputs to auto-generated CSV artifacts without mandatory LLM dependency.

---

## 2. Architecture & Stack
The system is decoupled into two cloud-deployable services (often hosted on Render's free tier):

1. **FastAPI Backend (`api/server.py`)**: 
   - Exposes REST endpoints (`/query`, `/agent_query`, and `/health`).
   - Orchestrates the RAG retrieval pipeline and the AI agent tool loops.
   - Secured via an internal `internal-api-key` header.
2. **Streamlit Frontend (`app/main.py`)**:
   - The user interface application.
   - Supports toggling between standard REST queries and interactive LLM "Agent Mode" chat streams.
3. **Core LLM/RAG Logics**:
   - `rag/`: Manages multimodal TF-IDF embedding generation, indexing, evaluating fallback mechanisms (missing evidence), scoring (Recall/Precision), and metric logging.
   - `agent/`: Orchestrates the complex multi-step logical looping framework, transforming tool calls into natural answers, parsing JSON, and preventing state issues.

---

## 3. How to Run Locally

*Note: Requires Python 3.12.10*

**1. Environment Setup & Installation**
```bash
git clone https://github.com/kbrh3/lab45542.git
cd lab45542

# Create and activate virtual environment
python -m venv .venv

# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**2. Configure Secrets**
Copy `.env.example` to `.env`. This `.env` file must not be committed to Git. Include the following values:
```env
BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=your_internal_secret_key
GEMINI_API_KEY=your_gemini_key # Mandatory if using Agent Mode

# Snowflake parameters:
SNOWFLAKE_ACCOUNT=your_snowflake_account
SNOWFLAKE_USER=your_snowflake_user
SNOWFLAKE_PASSWORD=your_snowflake_password
# Etc...

ENABLE_ERAG_EVAL=true # (Optional) Turn on eRAG score tracking
```

**3. Start the Backend API (Terminal 1)**
```bash
# From the root application folder:
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**4. Start the Frontend UI (Terminal 2)**
```bash
# Ensure venv is activated
streamlit run app/main.py
```
After launching, navigate to `http://localhost:8501`. Enable the "Agent Mode" toggle in the sidebar to chat directly with the configured AI assistant.

---

## 4. File Organization & Directory Structure

The repository is modular and relies heavily on isolated responsibilities.

### Main Components
*   `app/` **(Frontend Application)**
    *   `main.py`: The UI dashboard. Handles rendering logic, the chat history loop, toggling modes, and API HTTP requests formatting.
    *   `requirements.txt`: Lightweight frontend dependencies (Streamlit, Requests).
*   `api/` **(Backend Application)**
    *   `server.py`: FastAPI definitions, API routing, exception handling, and CORS middleware logic.
    *   `requirements.txt`: API-specific requirements (FastAPI, Uvicorn).
*   `rag/` **(RAG Pipeline Data Processors)**
    *   `pipeline.py`: Central hub executing metrics, retrieval loops, multimodality fusion, and formatting.
    *   `retriever.py`: Base retrieval implementations bridging text documents and multimedia figures.
    *   `snowflake_retriever.py`: Script dedicated strictly to querying remote Snowflake data lakes.
    *   `indexer.py`, `evaluator.py`, `generator.py`, `data_loader.py`: Utilities handling vector similarities, LLM answer parsing, evaluation calculations, and raw data ingestion.
    *   `logger.py`, `config.py`, `state.py`, `models.py`: Structural definitions mapping variables, caching application states, and orchestrating outputs.
*   `agent/` **(Intelligent LLM Multi-step Framework)**
    *   `coordinator.py`: Core logic for managing the AI's execution steps and overall agent orchestration state.
    *   `runner.py`: Dispatches API tools and evaluates condition loops.
    *   `tool_registry.py` & `tools/`: Implementations of independent tools the AI agent can activate.
    *   `llm_client.py`: Client interface wrapping the raw LLM (Gemini) endpoint integration.
    *   `schema_adapter.py`, `prompts.py`, `types.py`: Validation definitions, rigid schema structures, and deterministic prompt instruction files.

### Context Files & Tools
*   `data/` **(Evidence Source)**
    *   `pdfs/`: Text-heavy static documents like thesis/research papers.
    *   `figures/`: Screenshot and diagram assets evaluated via multimodal retrieval.
*   `artifacts/runs/` **(Evaluation Output Loggers)**
    *   Dynamically created on query executions. Stores `query_metrics.csv` capturing `Precision@5`, `Recall@10`, `eRAG scores`, and latency checks down to the millisecond.
*   `tests/` **(Stability Checkers)**
    *   `test_smoke.py`: Headless environment verifications employing python `unittest.mock` for pipeline logic checks omitting API dependencies.

### Configuration & Documentation
*   `.env` / `.env.example`: Secret tokens structure.
*   `README.md`: Centralized project documentation mapping endpoints and intent.
*   `AGENT_USAGE.md`: Tracks tasks handled securely via Agentic-AI.
*   `reproduce.sh`: A shell script enforcing identical environments and test executions across different workspaces.
*   `render.yaml`: Standardized host blueprint required for Render cloud platform.
*   `requirements.txt`: Root aggregation of standard Python package dependencies.
