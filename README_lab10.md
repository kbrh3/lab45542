# PolicyPulse - Lab 10

## Project Description
PolicyPulse is a Retrieval-Augmented Generation (RAG) web application designed to extract and reason over legislative insights. It has evolved into a production-ready system utilizing a decoupled architecture with a FastAPI backend and a Streamlit frontend. The system exclusively leverages a Snowflake data warehouse for enterprise-level legislative context, making it incredibly scalable for massive text databases. It also features an autonomous "Agent Mode" powered by the Gemini API, which allows the AI to invoke multi-step tools, and an eRAG suite to evaluate retrieval metrics dynamically.

## Setup Instructions

**Prerequisites:** Python 3.12.10

1. **Clone the repository and set up your virtual environment:**
```bash
git clone https://github.com/kbrh3/lab45542.git
cd lab45542

# Create and activate virtual environment
python -m venv .venv

# Activate on Mac/Linux:
source .venv/bin/activate
# Activate on Windows PowerShell:
# .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure Environment Variables:**
Copy the example configuration to create your local secrets manager:
```bash
cp .env.example .env
```
Open `.env` and fill in your credentials:
```env
BACKEND_URL=http://localhost:8000
BACKEND_API_KEY=your_internal_secret_key
GEMINI_API_KEY=your_gemini_api_key        # Mandatory for Agent Mode

# Snowflake Data Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PAT=your_pat_secret             # Programmatic Access Token
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# Validation tracking (Optional)
ENABLE_ERAG_EVAL=true 
```

## How to Run Backend + Frontend

You will need to open two separate terminal windows. Ensure your `.venv` is activated in **both** terminals.

**1. Start the Backend API (Terminal 1)**
Run the FastAPI server from the root directory:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
*(Verify it's running by checking `http://localhost:8000/health`)*

**2. Start the Frontend UI (Terminal 2)**
Launch the Streamlit web interface:
```bash
streamlit run app/main.py
```
*(The UI will open in your browser at `http://localhost:8501`)*

## Dataset Info
The application retrieves real, enterprise-level legislative records and bills specifically located from a remote **Snowflake Database** (`POLICYPULSE_DB.PUBLIC.BILLS` table). Previous generations of this application relied on static, local PDFs, but moving to an enterprise warehouse decoupled the backend from bulky storage allocations and dramatically improved semantic routing.

## Example Queries

To test out standard RAG functionality or the interactive Agent Mode, try these queries in the Streamlit interface:

1. **Legislative Data Extraction (Snowflake):** 
   - *"What bills are related to education funding in 2023?"*
   - *"Summarize the recent healthcare policies regarding prescription prices."*
2. **Interactive Agent Tasks:**
   - *"Find the status of the Clean Energy Transition Act and list similar bills targeting zero emissions."*

## Recent Codebase Organization & Cleanup
To make the repository significantly more readable and maintainable, we recently executed a massive codebase cleanup:
- **Deprecated Legacy Workflows**: Purged the entire `data/` directory containing old PDFs, along with `rag/data_loader.py` and `rag/indexer.py` local index generation scripts. The application now exclusively relies on dynamic Snowflake querying.
- **Root Repository Cleanup**: Removed 10+ obsolete temporary files, root scratch logs, and stale Git dumps (`output_log.txt`, `test_verify.py`, etc.). Old documentation iterations were effectively consolidated into an archived `previous_submissions.md` document. 
- **Refactored Test Suite**: Updated `tests/test_smoke.py` away from testing local dummy document PDFs logic. Implemented secure global `Mock()` structures testing our decoupled `snowflake_retriever` interactions natively without demanding live warehouse credentials in headless CI/CD stages. 
- **Streamlined State Mapping**: Optimized the pipeline build contexts `rag/__init__.py` and `rag/state.py` to bypass heavy matrix initialization patterns, preserving performance integrity while retaining strict FastAPI streaming definitions.
