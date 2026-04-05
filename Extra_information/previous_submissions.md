# Previous Submissions Archive

This document archives all previous READMEs and lab instructions, ordered from most recent to oldest.

---

# Submission 3 — PolicyPulse (Recent Update)
*(From `README_submission_3.md`)*

Hi! Welcome to **PolicyPulse**. I've been working on this project as I learn more about building AI applications. This README explains what the project is, how it works, and how to get it running. 

## 1. Project Overview
**PolicyPulse** is an AI-powered system designed to answer questions about legislation. You can ask it questions about different bills or laws, and it will try to find the right information to give you a clear answer. It's essentially a smart search engine specifically for legislative data!

## 2. How the System Works
The system is built using three main pieces that talk to each other:
- **The Frontend (Streamlit):** This is the user interface where you type your questions. It's simple and easy to use.
- **The Backend (FastAPI):** This is the "brain" of the app. It takes your question from the frontend, figures out what to do with it, and searches for the answer.
- **The Database (Snowflake):** This is where all the legislative data lives. Instead of looking through messy local files, the backend speaks to Snowflake to retrieve the exact text of the bills needed to answer your question.

## 3. What Was Fixed and Improved
When I first started jumping into the code, the system was a bit messy. It originally had issues where it returned incorrect data from old, outdated PDF files. Here is how I helped fix and improve it:
- **Upgraded to Snowflake:** I removed the old PDF-based data entirely. Now, the system uses a Snowflake-based retrieval system, which is much more structured and accurate.
- **Traced the Backend:** I spent time tracing the real execution path of the backend so I could see exactly how data was moving through the app and where things were getting stuck.
- **Added Debugging Tools:** I added tools to help monitor what the system is doing under the hood, making it easier to catch errors during retrieval.
- **Improved the UI:** I made the Streamlit frontend look and feel a lot nicer!
- **Added a Safe Fallback:** The system used to try and guess answers even if it found nothing. I added a safe fallback feature so that if no relevant legislative evidence is found, it will just honestly tell you that instead of making things up.

## 4. Known Issues & Improvements
- **Limited Data:** Right now, there isn't a massive amount of data in the Snowflake database yet. Because of this, the AI might not know the answer to every obscure legislative question you throw at it.
- **Adding More Data:** The biggest priority is loading more bills into Snowflake so the system can answer a wider variety of questions.
- **Faster Retrieval:** As the database grows, we'll need to figure out ways to make sure the app stays fast and responsive. 
- **Better Handling of Complex Questions:** Sometimes users ask multi-part questions, and the system could do a better job of breaking those down before searching the database.

---

# Lab 9 — Deployment Setup
*(From `README.md`)*

**Recommended Architecture:** Streamlit Community Cloud (Frontend) + Render (Backend)

**1. Backend Deployment (Render):**
- Connect your GitHub repository to Render and deploy the `lab45542-backend-api` service using the existing blueprint.
- Add your required `BACKEND_API_KEY` (and `GEMINI_API_KEY`, `USE_SNOWFLAKE`, `SNOWFLAKE_*` if using Agent mode) to the Render environment variables.
- **Note:** Snowflake may use a programmatic access token instead of a standard password if preferred.
- Copy your deployed backend URL.

**2. Frontend Deployment (Streamlit Cloud):**
- Create a new app on [Streamlit Cloud](https://share.streamlit.io) pointing to `app/main.py`.
- In **Advanced Settings > Secrets**, configure your connections:
  ```toml
  BACKEND_URL = "https://your-backend-url.onrender.com"
  BACKEND_API_KEY = "your-secret-key-here"
  ```
- Deploy the app.

**3. Post-Deployment:**
- Update the `allow_origins` list in `api/server.py` directly with your newly generated Streamlit URL to prevent CORS blocking.

---

# Lab 6 — Agent Integration
*(From `README.md` and `agent/README.md`)*

- Task 1: Antigravity IDE analysis completed
- Screenshot included in docx version of report
- See task1_antigravity_report.md
- lab 6 Demo video = https://youtu.be/zo2IeqDRkGI

## Agent Package Details
This package contains the core scaffolding for the AI Agent layer. The agent is responsible for coordinating tool executions and answering user queries based strictly on tool evidence.
It provides compatibility with the existing RAG API response schema, meaning the agent layer can be seamlessly dropped into the existing application without breaking Streamlit UI parsing.
- `prompts.py`: Defines the system instructions and behavioral guidelines for the AI agent (e.g. failing safely, returning correct schema).
- `types.py`: Defines types (like `AgentResponse` and `ToolTraceItem`) to ensure data integrity during tool execution and serialization.

## Agent Endpoint
A new AI agent layer is available at `/agent_query`.

**Example Request:**
```bash
curl -X POST http://localhost:8000/agent_query \
  -H "internal-api-key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the overall SQLENS pipeline?", "max_steps": 5}'
```

**How to run Agent Mode (UI)**
Agent Mode utilizes the `/agent_query` endpoint on the backend, converting the classic Streamlit app into an interactive Chat interface capable of multi-step tool execution.

**Note:** Gemini/Agent mode requires extra optional dependencies and API key. reproduce.sh and smoke tests run without them.

1. Ensure both the backend (`uvicorn api.server:app`) and frontend (`streamlit run app/main.py`) are actively running as described above.
2. Provide a valid `GEMINI_API_KEY` in the `.env` root file so the LLM routing logic successfully resolves.
3. Open the UI at `http://localhost:8501`.
4. In the left-hand sidebar under **Mode Selection**, toggle the **Agent Mode** button to On.
5. Watch the agent process your message in real-time. It will return an expandable interface parsing the `evidence`, `metrics`, `errors` and chronological `tool_trace` payloads naturally!

---

# Lab 4 — RAG Application Integration, Deployment & Monitoring
*(From main `README.md`)*

## Deployed Application
| Service | URL |
|---------|-----|
| **Streamlit Frontend** | <https://lab45542-frontend-streamlit.onrender.com> |
| **FastAPI Backend** | <https://lab45542-backend-api.onrender.com> |

> **Note:** Both services run on Render's free tier and may spin down after inactivity. The first request after idle can take 30–60 seconds while the container restarts.

## Dataset Description
The project-aligned dataset comes from our team's Lab 3 work on **SQL error detection and hallucination reduction** in LLM-generated outputs. It includes:
- **2–3 PDF documents** covering the SQLENS pipeline, ALIGNRAG, and FACT frameworks (`data/pdfs/`).
- **7 multimodal assets** (annotated screenshots of pipeline diagrams, causal graphs, and architecture overviews) (`data/figures/`).

Captions for each image are defined in `rag/pipeline.py` → `caption_map` and are used during TF-IDF-based multimodal retrieval.

### Evaluation Queries (Q1–Q5)
| ID | Type | Question |
|----|------|----------|
| Q1 | Typical project query | What is the overall SQLENS pipeline and what happens in each step? |
| Q2 | Typical project query | What semantic error types are shown in the causal graph and what signals are used to detect them? |
| Q3 | Typical project query | How does FACT reduce inconsistent hallucinations, and what kinds of hallucinations does it target? |
| Q4 | Multimodal evidence | Using the figure of the SQLENS pipeline, list the pipeline stages in order. |
| Q5 | Missing-evidence / ambiguous | Who won the FIFA World Cup in 2050? |

## Automatic Evaluation Logging

Every query automatically appends a row to `artifacts/runs/<run_id>/query_metrics.csv` with these fields:

| Field | Description |
|-------|-------------|
| `timestamp` | UTC ISO-8601 timestamp |
| `query_id` | Q1–Q5 identifier |
| `retrieval_mode` | `mm` (multimodal) or `text_only` |
| `top_k_evidence` | Number of evidence items fused |
| `latency_ms` | End-to-end pipeline latency |
| `Precision@5` | Fraction of top-5 results in gold set |
| `Recall@10` | Fraction of gold items found in top-10 |
| `evidence_ids_returned` | JSON list of retrieved evidence IDs |
| `gold_evidence_ids` | JSON list of expected evidence IDs |
| `faithfulness_pass` | Yes/No — heuristic grounding check |
| `missing_evidence_behavior` | Pass/Fail — correct handling when no evidence exists |

*(When **`ENABLE_ERAG_EVAL=true`** is set, three additional columns are appended: `erag_P_1`, `erag_P_3`, and `erag_P_5` tracking retrieval concept precision).*

## Results Snapshot
| Query | Retrieval Mode | Latency (ms) | Faithfulness | Missing-Evidence |
|-------|---------------|--------------|-------------|-----------------|
| Q1 | mm | ~15–30 | Yes | Pass |
| Q2 | mm | ~15–30 | Yes | Pass |
| Q3 | mm | ~15–30 | Yes | Pass |
| Q4 | mm | ~15–30 | Yes | Pass |
| Q5 | mm | ~10–20 | Yes | Pass |

Q5 correctly returns *"Not enough evidence in the retrieved context."* since no document covers FIFA World Cup 2050.

## Reflection
This lab transformed our Lab 3 notebook-based RAG pipeline into a production-ready two-service application. Key takeaways:
1. **Separation of concerns** — Splitting the Streamlit UI from the FastAPI backend made each component independently deployable and testable.
2. **Automatic logging** — CSV-based metric logging per query provides an audit trail without requiring external infrastructure.
3. **Missing-evidence handling** — Defining explicit behavior for out-of-scope queries (Q5) forced us to think about reliability boundaries, not just happy-path retrieval.
4. **Deployment realities** — Render's free tier cold-start latency highlighted the importance of lightweight initialization and user-facing status messages.

---

# eRAG Reproduction Steps
*(From `related_work/erag/README_erag_steps.md`)*

This folder contains scripts to run an eRAG-style (evaluative RAG) verification on our base retriever. 

## Setup and Execution (Windows PowerShell)
1. **Configure Data**: Make sure the actual vector database or TF-IDF matrices (if running the mock DB) are available to the script.
2. **Review Gold Set**: Update `gold_erag.json` if necessary to match the actual knowledge base you are testing against. It expects simple queries and a list of `required_concepts`.
3. **Create a virtual environment and activate it**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
4. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements_erag.txt
   ```
5. **Execute Evaluation**: 
   Run from this directory:
   ```powershell
   python erag_run_on_our_retrieval.py
   ```
6. **Analyze Output**: Check the output to see if the required concepts were actually retrieved by `rag/retriever.py:build_context`.
