import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Run configuration setup before any other Streamlit calls
st.set_page_config(page_title="PolicyPulse", layout="wide")

# Local dev environment loading
try:
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    dotenv_path = os.path.join(BASEDIR, '../.env') 
    load_dotenv(dotenv_path)
except Exception:
    pass

# Secrets
BACKEND_URL = os.getenv("BACKEND_URL")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

if not BACKEND_URL:
    st.error("The backend URL configuration is missing. Please ensure BACKEND_URL is defined in your environment variables.")
    st.stop()

BACKEND_URL = BACKEND_URL.rstrip('/')

if not BACKEND_API_KEY:
    st.error("The backend API key configuration is missing. Please ensure BACKEND_API_KEY is defined in your environment variables.")
    st.stop()

def check_backend():
    header = {
        "internal-api-key": BACKEND_API_KEY,
    }
    url = f'{BACKEND_URL}/health'
    try:
        r = requests.get(url, timeout=2, headers=header)
        r.raise_for_status()
        return True, None
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else None
        response_text = e.response.text if e.response is not None else None
        
        err_msg = f"Connection Attempt to URL: {url}\n"
        if status_code is not None:
            err_msg += f"Status Code: {status_code}\nServer Response: {response_text}\n"
        err_msg += f"Exception Details: {str(e)}"
        
        return False, err_msg

api_awake, backend_error_msg = check_backend()

# Begin page content
st.title("PolicyPulse")

st.markdown("PolicyPulse helps users ask questions about legislation using retrieval and agent-based reasoning. The system supports both a direct retrieval workflow and a multi-step agent workflow, with backend integration for policy data and Snowflake-supported querying.")

with st.expander("About Modes", expanded=False):
    st.info("""
    * **Classic Mode**: Returns answers directly from the retrieval pipeline based on the provided context.
    * **Agent Mode**: Supports multi-step reasoning and tool use for more complex questions, leveraging an interactive chat wrapper.
    """)

if not api_awake:
    st.warning(f"Note: The backend service is currently offline. This may occur due to free tier hosting limitations on Render. Please click [here]({BACKEND_URL}) to wake the service.")
    st.error("Backend health check failed. The system is temporarily unavailable.")
    with st.expander("Technical Details"):
        st.code(backend_error_msg, language="text")

# Agent Mode state
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# Sidebar
st.sidebar.header("System Status")
if api_awake:
    st.sidebar.success("Backend: Online")
else:
    st.sidebar.error("Backend: Offline")

show_debug = st.sidebar.checkbox("Show Debug Details", value=False)
st.sidebar.divider()

st.sidebar.header("Mode Selection")
agent_mode_on = st.sidebar.toggle("Enable Agent Mode", value=False)

st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode", 
    options=["mm", "text_only"],
    help="Select 'mm' for multimodal retrieval or 'text_only' for standard text retrieval."
)
top_k = st.sidebar.slider(
    "Number of Evidence Items", 
    min_value=1, 
    max_value=30, 
    value=8,
    help="Adjust the number of retrieved evidence chunks to display and process."
)

st.sidebar.header("Evaluation Settings")
st.sidebar.caption("System evaluation logs are automatically saved to the backend metrics database.")

def render_evidence(evidence_list, display_limit):
    for i, ev in enumerate(evidence_list[:display_limit]):
        with st.container(border=True):
            st.markdown(f"**{ev.get('citation_tag','Source')}** | *{ev.get('modality','Unknown Modality').title()}* | Relevance Score: **{ev.get('fused_score',0):.3f}**")
            st.caption(f"ID: {ev.get('id', 'N/A')}")
            
            text_content = ev.get("text") or ""
            if len(text_content) > 500:
                st.write(text_content[:500] + "...")
            else:
                st.write(text_content)
            
            path = ev.get("path")
            if ev.get("modality") == "image" and path:
                if path.startswith("http") or os.path.exists(path):
                    st.image(path, caption=ev.get("id",""), use_container_width=True)
                else:
                    st.caption("Image asset available on backend.")

if agent_mode_on:
    # -------------------------------------------------------------
    # Agent Mode UI
    # -------------------------------------------------------------
    st.subheader("Agent Mode")
    st.caption("Intended for more complex multi-step reasoning questions.")
    
    if st.sidebar.button("Clear conversation history"):
        st.session_state.agent_history = []
        st.rerun()

    # Render history
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If the msg has additional payload components like evidence, render them
            if msg.get("evidence"):
                with st.expander(f"Supporting Evidence ({len(msg['evidence'])} Items Retrieved)"):
                    render_evidence(msg["evidence"], top_k)
            
            if msg.get("metrics"):
                with st.expander("Metrics"):
                    st.json(msg["metrics"])
                    
            if msg.get("tool_trace"):
                with st.expander("Agent Tool Trace"):
                    st.json(msg["tool_trace"])
                    
            if msg.get("errors"):
                with st.expander("System Errors", expanded=True):
                    for err in msg["errors"]:
                        st.error(err)

    # Chat Input
    if user_input := st.chat_input("Enter your policy question..."):
        st.session_state.agent_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.spinner("Processing request via agent reasoning..."):
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
                url_agent = f"{BACKEND_URL}/agent_query"
                r = requests.post(url_agent, json=payload, headers=header, timeout=120)
                
                if show_debug:
                    with st.expander("Debug: Agent API Details", expanded=False):
                        st.write("**Endpoint:**", url_agent)
                        st.write("**Status Code:**", r.status_code)
                        st.write("**Payload:**")
                        st.json(payload)
                        st.write("**Response:**")
                        try:
                            st.json(r.json())
                        except Exception:
                            st.text(r.text)

                if r.status_code == 200:
                    data = r.json()
                    try:
                        answer = json.loads(data.get("answer", ""))['answer']
                    except Exception:
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
                            with st.expander(f"Supporting Evidence ({len(assistant_msg['evidence'])} Items Retrieved)"):
                                render_evidence(assistant_msg["evidence"], top_k)
                        if assistant_msg["metrics"]:
                            with st.expander("Metrics"):
                                st.json(assistant_msg["metrics"])
                        if assistant_msg["tool_trace"]:
                            with st.expander("Agent Tool Trace"):
                                st.json(assistant_msg["tool_trace"])
                        if assistant_msg["errors"]:
                            with st.expander("System Errors", expanded=True):
                                for err in assistant_msg["errors"]:
                                    st.error(err)
                                    
                else:
                    st.error("The system encountered an error fulfilling your request. Please try again or check the technical details.")
                    if show_debug:
                        with st.expander("Technical Details", expanded=True):
                            st.write(f"Status Code: {r.status_code}")
                            st.code(r.text, language="text")
            except Exception as e:
                st.error("Unable to reach the backend system. Please ensure the backend is online.")
                if show_debug:
                    with st.expander("Technical Details", expanded=True):
                        st.write("Connection failed:")
                        st.exception(e)

else:
    # -------------------------------------------------------------
    # Classic RAG UI
    # -------------------------------------------------------------
    st.subheader("Classic Retrieval Workflow")
    st.caption("Intended for direct retrieval-based questions.")
    
    st.write("Please enter a policy question below, or select a pre-defined evaluation question from the sidebar.")
    
    # Your Q1–Q5 selector (ids must match logs)
    MINI_GOLD = {
        "Q1": "What bills are related to education?",
        "Q2": "What is the latest action on healthcare-related bills?",
        "Q3": "Which bills are currently in committee?",
        "Q4": "Summarize recent legislation related to public safety",
        "Q5": "Who won the FIFA World Cup in 2050?",
    }
    
    query_id = st.sidebar.selectbox(
        "Question ID", 
        options=list(MINI_GOLD.keys()),
        help="Select a benchmark evaluation query to pre-fill the question area."
    )
    
    use_gold = st.sidebar.checkbox(
        "Use Gold Question Text", 
        value=True,
        help="Automatically populate the query input with the selected benchmark question."
    )
    
    question = st.text_area("Question", value=(MINI_GOLD[query_id] if use_gold else ""), height=120)
    
    run = st.button("Run Query")
    
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
            url = f"{BACKEND_URL}/query"
            r = requests.post(url, json=payload, headers=header, timeout=60)
            
            if show_debug:
                with st.expander("Debug: API Details", expanded=False):
                    st.write("**Endpoint:**", url)
                    st.write("**Status Code:**", r.status_code)
                    st.write("**Payload:**")
                    st.json(payload)
                    st.write("**Response:**")
                    try:
                        st.json(r.json())
                    except Exception:
                        st.text(r.text)
            
            r.raise_for_status()
            data = r.json()
    
            with colA:
                st.subheader("Answer")
                st.write(data.get("answer", ""))
    
                st.subheader(f"Supporting Evidence (Top {top_k})")
                evidence = data.get("evidence", [])
                for ev in evidence[:top_k]:
                    with st.container(border=True):
                        st.markdown(f"**{ev.get('citation_tag','Source')}** | *{ev.get('modality','Unknown Modality').title()}* | Relevance Score: **{ev.get('fused_score',0):.3f}**")
                        st.caption(f"ID: {ev.get('id', 'N/A')}")
                        text_content = ev.get("text") or ""
                        if len(text_content) > 500:
                            st.write(text_content[:500] + "...")
                        else:
                            st.write(text_content)
                        
                        path = ev.get("path")
                        if ev.get("modality") == "image" and path:
                            if path.startswith("http") or os.path.exists(path):
                                st.image(path, caption=ev.get("id",""), use_container_width=True)
                            else:
                                st.caption("Image asset available on backend.")
    
            with colB:
                st.subheader("Metrics")
                st.json(data.get("metrics", {}))
                st.success("Evaluation parameters successfully integrated into metrics logs.")
    
        except requests.exceptions.RequestException as e:
            st.error("A system error occurred while retrieving policy data. Please try again.")
            
            if show_debug:
                with st.expander("Technical Details", expanded=True):
                    status_code = e.response.status_code if e.response is not None else None
                    response_text = e.response.text if e.response is not None else None
                    
                    err_msg = f"Target Endpoint: {url}\nPayload Data: {json.dumps(payload, indent=2)}\n"
                    if status_code is not None:
                        err_msg += f"Status Code: {status_code}\nServer Response: {response_text}\n"
                    err_msg += f"Exception Details: {str(e)}"
                    
                    st.code(err_msg, language="text")
                    st.exception(e)
            else:
                st.info("Enable 'Show Debug Details' in the sidebar for technical information.")
            print(e)
