# Streamlit app goes here
import streamlit as st
import requests
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

st.set_page_config(page_title="CS5542 Lab 4 RAG App", layout="wide")
st.title("CS 5542 — Lab 4 RAG App")

# API_URL = st.sidebar.text_input("FastAPI base URL", value="http://127.0.0.1:8000")

st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox("retrieval_mode", ["mm", "text_only"])
top_k = st.sidebar.slider("top_k (display)", 1, 30, 8)

st.sidebar.header("Logging")
st.sidebar.caption("Logging happens in the backend: logs/query_metrics.csv")

# Your Q1–Q5 selector (ids must match logs)
MINI_GOLD = {
    "Q1": "What is the overall SQLENS pipeline and what happens in each step?",
    "Q2": "What semantic error types are shown in the causal graph and what signals are used to detect them?",
    "Q3": "How does FACT reduce inconsistent hallucinations, and what kinds of hallucinations does it target?",
    "Q4": "Using the figure of the SQLENS pipeline, list the pipeline stages in order.",
    "Q5": "Who won the FIFA World Cup in 2050?",
}

query_id = st.sidebar.selectbox("query_id", list(MINI_GOLD.keys()))
use_gold = st.sidebar.checkbox("Use gold question text", value=True)

question = st.text_area("Question", value=(MINI_GOLD[query_id] if use_gold else ""), height=120)

run = st.button("Run")

colA, colB = st.columns([2, 1])

if run and question.strip():
    header = {
        "internal-api-key": BACKEND_API_KEY,
    }
    payload = {
        "query_id": query_id,
        "question": question,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
    }

    try:
        r = requests.post(f"{BACKEND_URL}/query", json=payload, headers=header, timeout=60)
        r.raise_for_status()
        data = r.json()

        with colA:
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            st.subheader("Evidence (Top-K)")
            evidence = data.get("evidence", [])[:top_k]
            for ev in evidence:
                st.markdown(f"**{ev.get('citation_tag','')}** — *{ev.get('modality','')}* — score={ev.get('fused_score',0):.3f}")
                st.write((ev.get("text") or "")[:500])
                if ev.get("modality") == "image" and ev.get("path"):
                    st.image(ev["path"], caption=ev.get("id",""), use_container_width=True)
                st.divider()

        with colB:
            st.subheader("Metrics")
            st.json(data.get("metrics", {}))
            st.success("Logged to logs/query_metrics.csv (backend)")

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.caption("Make sure FastAPI is running and API_URL is correct.")

