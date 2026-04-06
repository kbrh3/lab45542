import json
import time
import uuid
from typing import List, Dict, Any, Optional

from agent.prompts import SYSTEM_PROMPT
from agent.types import AgentResponse, ToolTraceItem
from agent.tool_registry import execute_tool
from agent.llm_client import chat
from agent.schema_adapter import openai_tools_to_gemini, extract_tool_calls
from tool_schemas import TOOLS

try:
    from rag.pipeline import MISSING_EVIDENCE_MSG
except Exception:
    MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."

def run_agent(user_message: str, history: Optional[List[Dict[str, str]]] = None, max_steps: int = 5) -> AgentResponse:
    t0 = time.time()
    
    # Initialize the history if not provided
    messages = history or []
    if not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
    messages.append({"role": "user", "content": user_message})
    
    # State tracking
    tool_trace: List[ToolTraceItem] = []
    errors: List[str] = []
    evidence_list: List[Dict[str, Any]] = []
    rag_metrics: Dict[str, Any] = {}
    
    # Convert tools
    gemini_tools = openai_tools_to_gemini(TOOLS)
    
    step_count = 0
    final_answer = ""
    agent_ok = True
    
    try:
        while step_count < max_steps:
            step_count += 1
            
            # 1. Call LLM
            response = chat(messages, tools=gemini_tools)
            
            if not response.get("ok"):
                errors.append(f"LLM Error: {response.get('error')}")
                agent_ok = False
                final_answer = "I'm sorry, my configuration seems to be missing or I couldn't reach the language model."
                break
                
            tool_calls = extract_tool_calls(response)
            content = response.get("content", "")
            
            # Only append the assistant message content if there are no tool calls
            if not tool_calls and content:
                messages.append({"role": "assistant", "content": content})
            
            if not tool_calls:
                # Done! Agent didn't call any tools, so we have our final answer
                final_answer = content
                break
                
            # 2. Execute Tools
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                
                tool_res, trace_item = execute_tool(name, args)
                tool_trace.append(trace_item)
                
                if not trace_item["ok"]:
                    errors.append(f"Tool Error ({name}): {trace_item['error']}")
                    msg_content = f"Tool '{name}' failed with error: {trace_item['error']}"
                else:
                    # Capture useful metadata from specific tools if needed
                    if name == "rag_query":
                        # Collect evidence and metrics
                        data = tool_res.get("data", {})
                        if "evidence" in data:
                            evidence_list.extend(data["evidence"])
                        if "metrics" in data:
                            rag_metrics.update(data["metrics"])
                            
                    # Format success message for the LLM
                    # Convert the result dict to a JSON string
                    msg_content = f"Tool '{name}' returned:\n{json.dumps(tool_res.get('data', {}))}"
                    
                # Append tool result to messages as a "tool" role
                messages.append({"role": "context", "name": name, "content": msg_content})
                
        # Compile final metrics
        latency = (time.time() - t0) * 1000.0
        combined_metrics = {
            "agent_latency_ms": latency,
            "agent_steps": step_count,
            "agent_ok": agent_ok
        }
        # Merge any RAG metrics
        if rag_metrics:
            for k, v in rag_metrics.items():
                if k not in combined_metrics:
                    combined_metrics[k] = v
                    
        # If we hit max steps without natural termination
        if step_count >= max_steps and not final_answer:
            final_answer = "I reached my maximum number of working steps before I could finish the task."
            errors.append("Agent hit max_steps limit.")
            
        if final_answer.strip() == "":
            final_answer = "I couldn't produce an answer. Please try rephrasing your question."
            
        resp: AgentResponse = {
            "answer": final_answer,
            "evidence": evidence_list,
            "metrics": combined_metrics,
            "missing_evidence_msg": MISSING_EVIDENCE_MSG,
            "tool_trace": tool_trace,
            "errors": errors
        }
        
        return resp
        
    except Exception as e:
        # Failsafe Catch-All
        return {
            "answer": f"Agent encountered a fatal error: {str(e)}",
            "evidence": evidence_list,
            "metrics": {"agent_ok": False, "error_type": "fatal", "agent_steps": step_count, "agent_latency_ms": (time.time() - t0) * 1000.0},
            "missing_evidence_msg": MISSING_EVIDENCE_MSG,
            "tool_trace": tool_trace,
            "errors": errors + [str(e)]
        }
