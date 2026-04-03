# Project Overview: Lab 45542 RAG Application Architecture

## 1. What the Program Does (Project Purpose)
This project is a **Retrieval-Augmented Generation (RAG) web application** originally designed to extract insights from local PDF documents and image queries (multimodal reasoning) regarding topics like SQL error detection, LLM hallucinations, and conceptual architecture extraction.

Recently, it has evolved into a production-ready cloud application with advanced capabilities:
- **Enterprise Retrieval**: It has transitioned from purely local file extraction to querying data seamlessly from **Snowflake** data lakes for enterprise-level context.
- **Autonomous Agent Mode**: It includes an "Agent Mode" powered by the Gemini API. Instead of simply fetching documents, the agent can invoke multi-step tools, reason through complex queries, evaluate ambiguities, and provide a traced chain of its logical loops.
- **Deterministic Evaluation Suite (eRAG)**: Automatically calculates retrieval accuracy (Precision@K, Recall) and logs performance metrics (like latency and faithfulness) directly to CSV artifacts without strictly requiring an LLM.

---

## 2. Project Architecture
The system is built on a decoupled, two-tier architecture that is designed to be independently deployable (e.g., using Render and Streamlit Community Cloud).

### High-Level Flow:
```text
┌────────────────────┐        HTTPS / JSON        ┌─────────────────────┐
│ Streamlit Frontend │  ──────────────────────►   │   FastAPI Backend   │
│   (app/main.py)    │  ◄──────────────────────   │   (api/server.py)   │
└─────────┬──────────┘                            └─────────┬───────────┘
          │ toggles                                         │ handles requests
          ▼                                                 ▼
   [ UI Modes ]                                   ┌─────────────────────┐
   1. Standard RAG                                │ Core AI / RAG Logic │
   2. Agent Mode Chat                             │ - rag/ (Retriever)  │
                                                  │ - agent/ (LLM Loop) │
                                                  └─────────┬───────────┘
                                                            │ fetches context
                                                            ▼
                                                  [ External Services ]
                                                  - Snowflake (Data)
                                                  - Gemini API (LLM/Agent)
```

### Core Technical Components:
*   **Frontend UI (`app/`)**: Built with **Streamlit**. It provides the main user dashboard, handles chat loops, toggles between standard vs. agent queries, and manages HTTPS requests to the backend.
*   **Backend API (`api/`)**: Built with **FastAPI** and served via Uvicorn. Exposes REST endpoints (`/query`, `/agent_query`, `/health`). Handles API routing, CORS, and secures communication using an `internal-api-key`.
*   **RAG Pipeline (`rag/`)**: Handles embedding generation, indexing, TF-IDF scoring, multimodal fusion (text + images), evaluating fallback mechanisms for missing evidence, and metric logging. Includes `snowflake_retriever.py` dedicated to querying remote Snowflake databases.
*   **Agent Framework (`agent/`)**: Intelligent multi-step framework orchestrating the AI's execution pipeline. Features a `coordinator.py` to manage state, `runner.py` for condition loops, and a tool registry where independent plugins can be executed by the LLM.

---

## 3. How to Use the Program (Local Setup)

**Prerequisites:** Python 3.12.10

### Step 1: Clone and Environment Setup
```bash
git clone https://github.com/kbrh3/lab45542.git
cd lab45542

# Create and activate virtual environment
python -m venv .venv

# Activate (Mac/Linux):
source .venv/bin/activate
# Activate (Windows PowerShell):
# .venv\Scripts\activate

# Install dependencies globally to the project
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the placeholder configuration file to establish your local secrets. 
```bash
cp .env.example .env
```
Open `.env` and fill out the required credentials:
```env
BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=your_internal_secret_key
GEMINI_API_KEY=your_gemini_api_key        # Mandatory for Agent Mode

# Snowflake Data Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password

# Optional: Track retrieval Quality
ENABLE_ERAG_EVAL=true 
```

### Step 3: Start the Backend (Terminal 1)
With your `.venv` activated, launch the FastAPI server:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
*(You can verify it's running by checking `http://localhost:8000/health`)*

### Step 4: Start the Frontend (Terminal 2)
Open a second terminal, ensure your `.venv` is activated, and launch the Streamlit interface:
```bash
streamlit run app/main.py
```

### Step 5: Interactive Agent Mode Chat
1. The Streamlit UI will open in your browser at `http://localhost:8501`.
2. In the left-hand sidebar under **Mode Selection**, toggle the **Agent Mode** button to On.
3. Chat with the agent by asking complex questions like, *"What is the overall SQLENS pipeline?"*
4. Explore the parsed responses! The UI will expand to show you the LLM's natural answer, the exact `evidence` referenced, runtime `metrics` (down to the millisecond), and a chronological `tool_trace`.

*(Note: Data evaluated includes legacy PDFs in `data/pdfs/` and multimodal image queries in `data/figures/`, but modern agent tasks execute primarily by streaming from your defined Snowflake cluster).*
