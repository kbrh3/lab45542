

# Lab 6 — Agent Integration

- Task 1: Antigravity IDE analysis completed
- Screenshot included in docx version of report
- See task1_antigravity_report.md
- lab 6 Demo video = https://youtu.be/zo2IeqDRkGI

# CS 5542 — Lab 4: RAG Application Integration, Deployment & Monitoring
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

---

## Repository Structure

```
lab45542/
├── app/
│   ├── main.py               # Streamlit frontend
│   └── requirements.txt      # Frontend dependencies
├── api/
│   ├── server.py              # FastAPI backend (REST API)
│   └── requirements.txt      # Backend dependencies
├── rag/
│   └── pipeline.py            # Core RAG pipeline, metrics, logging
├── data/
│   ├── pdfs/                  # PDF documents
│   └── figures/               # Multimodal image assets
├── artifacts/
│   └── runs/                  # Automatic evaluation logs (per-run directories)
│       └── <run_id>/query_metrics.csv
├── render.yaml                # Render deployment blueprint
├── .gitignore
└── README.md
```

---

## Architecture

```
┌────────────────────┐        HTTPS / JSON        ┌─────────────────────┐
│  Streamlit Frontend │  ──────────────────────►  │  FastAPI Backend     │
│  (app/main.py)      │  ◄──────────────────────  │  (api/server.py)     │
└────────────────────┘                            └────────┬────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────────┐
                                                  │  RAG Pipeline        │
                                                  │  (rag/pipeline.py)   │
                                                  │  - PDF extraction    │
                                                  │  - TF-IDF indexing   │
                                                  │  - Multimodal fusion │
                                                  │  - Evaluation metrics│
                                                  │  - CSV logging       │
                                                  └─────────────────────┘
```

- **Frontend → Backend** communication is secured via an `internal-api-key` header.
- **Logging** is automatic: every `/query` call appends a row to a per-run metrics file `artifacts/runs/<run_id>/query_metrics.csv` with timestamp, retrieval mode, latency, Precision@5, Recall@10, evidence IDs, faithfulness, and missing-evidence behavior. (Note: The old `logs/query_metrics.csv` fallback is still supported.)

## Agent Endpoint

A new AI agent layer is available at `/agent_query`.

**Example Request:**
```bash
curl -X POST http://localhost:8000/agent_query \\
  -H "internal-api-key: your-secret-key-here" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "What is the overall SQLENS pipeline?", "max_steps": 5}'
```

**Example Response:**
```json
{
  "answer": "The SQLENS pipeline consists of...",
  "evidence": [...],
  "metrics": {
    "agent_latency_ms": 1205.5,
    "agent_steps": 2,
    "agent_ok": true
  },
  "missing_evidence_msg": "...",
  "tool_trace": [...],
  "errors": []
}
```

---

## Local Run (Recommended)

**Prerequisites:** Python 3.12.10

**1. Clone & setup virtual environment:**
```bash
git clone https://github.com/kbrh3/lab45542.git
cd lab45542

# Create and activate venv
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Upgrade pip and install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**2. Add data:**
Place PDF files in `data/pdfs/` and screenshot images in `data/figures/`.

**3. Configure Environment Variables:**
Create your local environment configuration by copying the example file:

```bash
cp .env.example .env
```
*(Note: `.env` is gitignored; never commit secrets! Open your new `.env` file and replace the placeholder values with your actual credentials).*

**4. Start Backend (Terminal 1):**
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
*Backend URL:* http://localhost:8000
*(Quick check: `curl http://localhost:8000/health`)*

**5. Start Frontend (Terminal 2):**
```bash
# Make sure .venv is activated here too!
streamlit run app/main.py
```
*Frontend URL:* http://localhost:8501

### 6. How to run Agent Mode

Agent Mode utilizes the `/agent_query` endpoint on the backend, converting the classic Streamlit app into an interactive Chat interface capable of multi-step tool execution.

**Note:** Gemini/Agent mode requires extra optional dependencies and API key. reproduce.sh and smoke tests run without them.

**Setup Instructions:**
1. Ensure both the backend (`uvicorn api.server:app`) and frontend (`streamlit run app/main.py`) are actively running as described above.
2. Provide a valid `GEMINI_API_KEY` in the `.env` root file so the LLM routing logic successfully resolves.
3. Open the UI at `http://localhost:8501`.
4. In the left-hand sidebar under **Mode Selection**, toggle the **Agent Mode** button to On.
5. The interface will switch to a chat window. Try asking:
   - *"What is the overall SQLENS pipeline?"*
   - *"How does FACT reduce inconsistent hallucinations?"*
6. Watch the agent process your message in real-time. It will return an expandable interface parsing the `evidence`, `metrics`, `errors` and chronological `tool_trace` payloads naturally!

---

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

---

## Results Snapshot

| Query | Retrieval Mode | Latency (ms) | Faithfulness | Missing-Evidence |
|-------|---------------|--------------|-------------|-----------------|
| Q1 | mm | ~15–30 | Yes | Pass |
| Q2 | mm | ~15–30 | Yes | Pass |
| Q3 | mm | ~15–30 | Yes | Pass |
| Q4 | mm | ~15–30 | Yes | Pass |
| Q5 | mm | ~10–20 | Yes | Pass |

Q5 correctly returns *"Not enough evidence in the retrieved context."* since no document covers FIFA World Cup 2050.

---

## Failure Analysis

### Failure 1: Retrieval Failure (Low Precision on Q4 — Multimodal Query)

**Observed behavior:** When asking Q4 ("Using the figure of the SQLENS pipeline, list the pipeline stages in order"), text-only retrieval mode returns PDF pages that mention the pipeline but does not surface the actual diagram image. Precision@5 drops because the gold evidence is an image asset.

**Root cause:** TF-IDF over text pages cannot match visual content. Without multimodal fusion (`retrieval_mode=mm`), the image caption index is bypassed entirely.

**Proposed fix:** Always enable multimodal retrieval for queries that reference figures, diagrams, or visual elements. A more robust solution would use a query classifier to auto-detect multimodal intent and force `retrieval_mode=mm`.

### Failure 2: Grounding / Missing-Evidence Failure (Q5 — Ambiguous Case)

**Observed behavior:** If the score threshold in `generate_answer()` is set too low (e.g., < 0.01), Q5 ("Who won the FIFA World Cup in 2050?") returns a spurious extractive snippet from an unrelated document page instead of the expected missing-evidence message.

**Root cause:** TF-IDF assigns small but non-zero scores to unrelated documents (e.g., a page containing the word "2050" in a different context). If the relevance threshold is not strict enough, the system treats noise as valid evidence and generates an unfaithful answer.

**Proposed fix:** The current threshold of `0.05` in `generate_answer()` correctly filters this case. For production systems, a calibrated confidence threshold (e.g., learned from a validation set) or a separate reranker stage would provide more reliable missing-evidence detection.

---

## Reflection

This lab transformed our Lab 3 notebook-based RAG pipeline into a production-ready two-service application. Key takeaways:

1. **Separation of concerns** — Splitting the Streamlit UI from the FastAPI backend made each component independently deployable and testable.
2. **Automatic logging** — CSV-based metric logging per query provides an audit trail without requiring external infrastructure.
3. **Missing-evidence handling** — Defining explicit behavior for out-of-scope queries (Q5) forced us to think about reliability boundaries, not just happy-path retrieval.
4. **Deployment realities** — Render's free tier cold-start latency highlighted the importance of lightweight initialization and user-facing status messages.

---

## Deployment (Render)

The `render.yaml` blueprint defines both services. To deploy:

1. Push this repo to GitHub.
2. In the Render dashboard, create a new **Blueprint** and point it at the repo.
3. Set the required environment variables (`BACKEND_URL`, `BACKEND_API_KEY`) in each service's settings.
4. Deploy.

Both services use Python with their respective `requirements.txt` files.
