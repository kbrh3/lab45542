#!/usr/bin/env bash

# reproduce.sh
# Automates the setup, testing, and a single reproducible run of the pipeline.

set -e # Exit on error

echo "=========================================="
echo " Starting Reproducibility Script"
echo "=========================================="

# 1. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment '.venv'..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment '.venv' already exists."
fi

# Activate venv
echo "      Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# 2. Install dependencies
echo "[2/4] Installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Smoke Tests
echo "[3/4] Running smoke tests (requires NO external API keys)..."
python -m unittest tests/test_smoke.py

# 4. Generate Reproducibility Run (Artifacts)
echo "[4/4] Executing a single pipeline run..."

# Inline Python script to run a query and trigger artifact generation
python3 -c "
import sys
import os
from rag.pipeline import run_query_and_log, get_mini_gold

try:
    print('      Initializing pipeline...')
    # Fetch the first evaluation query
    query_item = get_mini_gold()[0]
    
    print(f'      Running query: {query_item[\"question\"]}')
    result = run_query_and_log(query_item, retrieval_mode='mm')
    
    # We successfully ran the query. Metrics are logged!
except Exception as e:
    print(f'      Error running pipeline: {e}')
    sys.exit(1)
"

# Locate the most recent run artifacts
RUNS_DIR="artifacts/runs"
if [ -d "$RUNS_DIR" ]; then
    LATEST_RUN=$(ls -td $RUNS_DIR/* | head -1)
    echo "=========================================="
    echo " Reproducibility run completed successfully!"
    echo " Artifacts and metrics were written to:"
    echo " -> $LATEST_RUN/query_metrics.csv"
    echo "=========================================="
else
    echo "=========================================="
    echo " Reproducibility run completed but no artifacts directory found."
    echo "=========================================="
fi
