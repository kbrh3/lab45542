# Reproducibility Audit Report (REPRO_AUDIT.md)

## Environment & Pinned Dependencies
The pipeline environment is specified centrally to ensure consistent runs.
- **Python Version:** 3.10+
- **Combined Dependencies:** `requirements.txt` (Root-level)
- **Backend Dependencies:** `api/requirements.txt`
- **Frontend Dependencies:** `app/requirements.txt`

The dependencies in `requirements.txt` are deduplicated to ensure identically locked versions whenever `pip install -r requirements.txt` is run.

## Determinism Controls
To guarantee identical pipeline runs across different local environments, the following determinism controls are implemented in `rag/pipeline.py`:
- **Seed Handling:** NumPy random operations rely on a fixed seed (`np.random.seed(0)`).
- **File Ordering:** Files extracted from `data/pdfs/` and `data/figures/` use `sorted(glob.glob(...))` to ensure identical loading orders across Mac/Linux/Windows operating systems.
- **Data Splits:** The evaluation set (`mini_gold`) is hardcoded in the pipeline, ensuring evaluations always execute against identically ordered expectations.

## One-Command Execution
The entire environment setup, test verification, and test execution can be automated using:
```bash
bash reproduce.sh
```
This script handles:
1. Virtual environment creation (`.venv`).
2. Dependency installation.
3. Smoke test validation.
4. A single evaluation query run.

## Artifacts Produced
Artifacts are automatically generated for each pipeline execution to maintain an isolated audit trail.
- **Path:** `artifacts/runs/<run_id>/query_metrics.csv`
- **Format:** Comma-Separated Values (`CSV`)
- **Note:** `<run_id>` corresponds to the timestamp of the run execution (e.g. `20260309_165500`). 

## Logs Produced
Logs from the standard evaluation (`reproduce.sh`) capture:
- `timestamp`: UTC ISO-8601 of the run.
- `query_id`: Identifier of the evaluated question (e.g., Q1).
- `retrieval_mode`: Text only or multimodal (`mm`).
- `top_k_evidence`, `latency_ms`, `Precision@5`, `Recall@10`.
- `evidence_ids_returned` and `gold_evidence_ids`.
- `faithfulness_pass` and `missing_evidence_behavior`.

## Smoke Test Description
The smoke test (`tests/test_smoke.py`) validates the core extraction and retrieval mechanisms of the RAG pipeline **without requiring external network calls or API keys**.
- Dynamically creates tiny PDF (`fitz`) and Image (`Pillow`) assets in memory.
- Verifies `init_pipeline()` successfully indexes both file types locally.
- Tests `run_pipeline()` with an arbitrary query.
- Employs `unittest.mock` to intercept and bypass LLM calls within `agent/runner.py`.
- **Duration:** < 1 second.

## Known Limitations / Nondeterminism
- **Dependency Version Drift:** While dependencies are documented, some sub-dependencies might shift if underlying package versions aren't perfectly pinned (`==x.y.z`).
- **Cold-Start Latency:** Depending on local machine limits (or Render.com container cold-starts for deployed versions), initial pipeline logs might show slightly sporadic latency measurements for the first query executed.
- **LLM API Calls:** For requests routed through the actual LLM (when a `GEMINI_API_KEY` is provided during interactive app usage), generative responses naturally possess temperature-based nondeterminism not covered by the `run_pipeline` test functions.

---
**Audit Log Details:**
- **Commit Hash:** `<INSERT_COMMIT_HASH_HERE>`
- **Run ID Example:** `<INSERT_RUN_ID_EXAMPLE_HERE>`
