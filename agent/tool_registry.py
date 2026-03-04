import time
import json
from typing import Dict, Any, Tuple, Callable
from tools import rag_query, snowflake_query, bills_analytics, summarize_text
from .types import ToolTraceItem

# Registry mapping tool names to their corresponding functions
TOOL_REGISTRY: Dict[str, Callable] = {
    "rag_query": rag_query,
    "snowflake_query": snowflake_query,
    "bills_analytics": bills_analytics,
    "summarize_text": summarize_text,
}

def execute_tool(tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], ToolTraceItem]:
    """
    Executes a tool from the registry safely, tracking latency and producing a trace.
    
    Args:
        tool_name (str): The name of the tool to execute.
        args (dict): The arguments to pass to the tool.
        
    Returns:
        tuple[dict, ToolTraceItem]: The tool's resulting dictionary and its trace item.
    """
    t0 = time.time()
    
    if tool_name not in TOOL_REGISTRY:
        latency = (time.time() - t0) * 1000.0
        err_msg = f"unknown tool: {tool_name}"
        trace: ToolTraceItem = {
            "tool_name": tool_name,
            "args": args,
            "ok": False,
            "latency_ms": latency,
            "error": err_msg
        }
        return {"ok": False, "error": err_msg}, trace
        
    tool_func = TOOL_REGISTRY[tool_name]
    
    try:
        # Execute the underlying tool function
        # We assume tools return a dictionary format similar to {"ok": bool, "data": ...}
        # Based on tools.py, the tools are already catching exceptions internally,
        # but we wrap them again as a guarantee.
        result = tool_func(**args)
        
        # Ensure it is JSON Serializable
        try:
            json.dumps(result)
        except TypeError:
            result = {"ok": result.get("ok", False), "data": str(result.get("data", ""))}
            
        latency = (time.time() - t0) * 1000.0
        
        trace: ToolTraceItem = {
            "tool_name": tool_name,
            "args": args,
            "ok": result.get("ok", False),
            "latency_ms": latency,
            "error": result.get("error") if not result.get("ok") else None
        }
        
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        err_msg = str(e)
        trace: ToolTraceItem = {
            "tool_name": tool_name,
            "args": args,
            "ok": False,
            "latency_ms": latency,
            "error": err_msg
        }
        result = {"ok": False, "error": err_msg}
        
    return result, trace
