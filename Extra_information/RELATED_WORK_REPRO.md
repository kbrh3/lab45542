# Related Work Reproduction

This document tracks reproduction efforts for various relevant literature, techniques, and benchmarks that we are applying to our own RAG pipeline.

## 1. eRAG (Evaluative RAG) Reproduction

### Repo/paper summary (what eRAG does)
eRAG (evaluative RAG) is a framework designed to robustly evaluate Retrieval-Augmented Generation (RAG) systems. It evaluates retrieval by checking if the downstream generated text contains the necessary semantic concepts required to answer a query. This evaluates the retrieval and generation holistically rather than assuming simple passage overlap is sufficient.

### What we attempted (install, toy run, run on our retrieval outputs)
We attempted to reproduce the eRAG evaluation and apply it to our project's specific retrieval components. This included defining a gold set of queries tailored to our project with simple `expected_concepts` (in `gold_erag.json`), writing a driver script (`erag_run_on_our_retrieval.py`) that executes our multi-modal extraction pipeline, and parsing the documents to pipe into `erag.eval()`.

### What worked (scripts ran, metrics produced)
The core `erag.eval()` loop successfully mapped our retrieved documents using our deterministic string concatenation generator (`text_generator`) and boolean substring matcher (`downstream_metric`). The script successfully generated outputs like Precision@1, Precision@3, and Precision@5, capturing the quality of our retrieval against the `expected_concepts`.

### What failed / gaps
We encountered a few notable friction points during the reproduction:
- **Missing OS Dependencies**: We hit a `RuntimeError` due to paths to our vector-DB/documents being hard-coded incorrectly. 
- **Missing Python Dependencies**: The `erag` package failed to list `numpy` as a required dependency, prompting us to manually pin `numpy==2.4.2` and install it.
- **Metric Name Mismatch**: The package failed when requesting unsupported metric strings like `"success"`. We bypassed this by explicitly requesting `{"P_1", "P_3", "P_5"}`.

### Differences vs reported results
We only focused on reproducing the mechanistic evaluation loop of eRAG (the capability to assess a custom retrieval system via deterministic concepts). We did not recreate their full Kendall $\tau$ correlation experiments against human judgments, nor did we test heavily on their reported datasets (like NaturalQuestions or MS MARCO), as we applied it specifically to our own domain data (SQLENS/ALIGNRAG/FACT papers).

### Improvement integrated into our system
We have successfully integrated the eRAG evaluative system natively into our application's `rag/pipeline.py` script. 
By default, this is disabled to protect existing runs, but users can now leverage the `ENABLE_ERAG_EVAL` configuration flag to compute deterministic concept-recall metrics on-the-fly and save them directly to our `query_metrics.csv` evaluation artifact.

**How to run our reproduction script:**
```powershell
# Setup virtual environment
cd related_work/erag
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
python -m pip install --upgrade pip
pip install -r requirements_erag.txt

# Execute eRAG eval on our dataset
python erag_run_on_our_retrieval.py
```
> The textual output metrics are printed directly to the console.

**How to enable optional logging of eRAG metrics in our main pipeline:**
```powershell
# From the repository root
$env:ENABLE_ERAG_EVAL="true"

# Running any query against the application (or via agent) will now log erag_P_1, erag_P_3, erag_P_5 
# in the metrics CSV file located at: artifacts/runs/<run_id>/query_metrics.csv
```
*(Example run ID generated with this enabled: `20260310_205906`)*

More techniques will be added here as we continue to benchmark and test our system against other standards.
