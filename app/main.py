import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Local dev environment loading
try:
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    dotenv_path = os.path.join(BASEDIR, '../.env') 
    load_dotenv(dotenv_path)
except:
    pass

# Secrets
BACKEND_URL = os.getenv("BACKEND_URL")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

if not BACKEND_URL:
    st.error("BACKEND_URL is not set. Add it to .env or Render environment variables.")
    st.stop()

BACKEND_URL = BACKEND_URL.rstrip('/')

if not BACKEND_API_KEY:
    st.error("BACKEND_API_KEY is not set. Add it to .env or Render environment variables.")
    st.stop()

def check_backend():
    header = {
        "internal-api-key": BACKEND_API_KEY,
    }
    try:
        r = requests.get(f'{BACKEND_URL}/status', timeout=2, headers=header)
        r.raise_for_status()
        return True
    except:
        return False
api_awake = check_backend()


# Begin page content
st.set_page_config(page_title="CS5542 Lab 4 RAG App", layout="wide")
st.title("CS 5542 — Lab 4 RAG App")
if not api_awake:
    st.warning(f"Backend is offline (Render free tier limitations). Reboot it by following this link: {BACKEND_URL}")

# Agent Mode state
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# Sidebar
st.sidebar.header("Mode Selection")
agent_mode_on = st.sidebar.toggle("Agent Mode", value=False)

st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox("retrieval_mode", ["mm", "text_only"])
top_k = st.sidebar.slider("top_k (display)", 1, 30, 8)

st.sidebar.header("Logging")
st.sidebar.caption("Logging happens in the backend: logs/query_metrics.csv")

if agent_mode_on:
    # -------------------------------------------------------------
    # Agent Mode UI
    # -------------------------------------------------------------
    st.subheader("Agent Mode (Experimental)")
    
    if st.sidebar.button("Clear chat"):
        st.session_state.agent_history = []
        st.rerun()

    # Render history
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If the msg has additional payload components like evidence, render them
            if msg.get("evidence"):
                with st.expander(f"View {len(msg['evidence'])} Evidence Items"):
                    for ev in msg["evidence"][:top_k]:
                        st.markdown(f"**{ev.get('citation_tag','')}** — *{ev.get('modality','')}* — score={ev.get('fused_score',0):.3f}")
                        st.write((ev.get("text") or "")[:500])
                        
                        path = ev.get("path")
                        if ev.get("modality") == "image" and path:
                            if path.startswith("http") or os.path.exists(path):
                                st.image(path, caption=ev.get("id",""), use_container_width=True)
                            else:
                                st.caption("Image path available only in local runtime.")
                        st.divider()
            
            if msg.get("metrics"):
                with st.expander("Metrics"):
                    st.json(msg["metrics"])
                    
            if msg.get("tool_trace"):
                with st.expander("Tool Trace"):
                    st.json(msg["tool_trace"])
                    
            if msg.get("errors"):
                with st.expander("Errors", expanded=True):
                    for err in msg["errors"]:
                        st.error(err)

    # Chat Input
    if user_input := st.chat_input("Ask the agent something..."):
        st.session_state.agent_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.spinner("Agent thinking..."):
            header = {"internal-api-key": BACKEND_API_KEY}
            
            # Send the history strictly excluding special extra UI keys to the backend
            clean_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.agent_history
            ]
            
            payload = {
                "message": user_input,
                "history": clean_history[:-1], # everything except the message we just sent
                "max_steps": 5
            }
            
            try:
                r = requests.post(f"{BACKEND_URL}/agent_query", json=payload, headers=header, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("answer", "")
                    
                    # Construct assistant message with all metadata
                    assistant_msg = {
                        "role": "assistant",
                        "content": answer,
                        "evidence": data.get("evidence", []),
                        "metrics": data.get("metrics", {}),
                        "tool_trace": data.get("tool_trace", []),
                        "errors": data.get("errors", [])
                    }
                    st.session_state.agent_history.append(assistant_msg)
                    
                    # Render the assistant message immediately so it doesn't wait for the hard rerun
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if assistant_msg["evidence"]:
                            with st.expander(f"View {len(assistant_msg['evidence'])} Evidence Items"):
                                for ev in assistant_msg["evidence"][:top_k]:
                                    st.markdown(f"**{ev.get('citation_tag','')}** — *{ev.get('modality','')}* — score={ev.get('fused_score',0):.3f}")
                                    st.write((ev.get("text") or "")[:500])
                                    path = ev.get("path")
                                    if ev.get("modality") == "image" and path:
                                        if path.startswith("http") or os.path.exists(path):
                                            st.image(path, caption=ev.get("id",""), use_container_width=True)
                                        else:
                                            st.caption("Image path available only in local runtime.")
                                    st.divider()
                        if assistant_msg["metrics"]:
                            with st.expander("Metrics"):
                                st.json(assistant_msg["metrics"])
                        if assistant_msg["tool_trace"]:
                            with st.expander("Tool Trace"):
                                st.json(assistant_msg["tool_trace"])
                        if assistant_msg["errors"]:
                            with st.expander("Errors", expanded=True):
                                for err in assistant_msg["errors"]:
                                    st.error(err)
                                    
                else:
                    st.error(f"Backend returned error {r.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to agent backend: {e}")

else:
    # -------------------------------------------------------------
    # Classic RAG UI
    # -------------------------------------------------------------
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
                    path = ev.get("path")
                    if ev.get("modality") == "image" and path:
                        if path.startswith("http") or os.path.exists(path):
                            st.image(path, caption=ev.get("id",""), use_container_width=True)
                        else:
                            st.caption("Image path available only in local runtime.")
                    st.divider()
    
            with colB:
                st.subheader("Metrics")
                st.json(data.get("metrics", {}))
                st.success("Logged to logs/query_metrics.csv (backend)")
    
        except Exception as e:
            st.error(f"API call failed")
            print(e)


