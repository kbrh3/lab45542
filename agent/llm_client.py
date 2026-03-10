import os
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def chat(messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Sends a chat request to Gemini.
    
    Args:
        messages (list[dict]): Standard chat messages with "role" and "content".
            (e.g., [{"role": "user", "content": "Hello"}])
        tools (list[dict], optional): Tool schemas to pass to the model.
        
    Returns:
        dict: A dictionary containing the response content, tool calls, and status ok.
    """
    if not HAS_GENAI:
        return {
            "ok": False,
            "error": "Informational Note: Gemini/Agent mode requires extra optional dependencies and API key. reproduce.sh and smoke tests run without them."
        }
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "GEMINI_API_KEY not configured."
        }
        
    try:
        genai.configure(api_key=api_key)
        
        # Convert standard Dictionaries into Gemini's expected format if needed
        # We assume standard user/assistant format. Gemini natively uses 'user' and 'model'
        gemini_history = []
        last_message = ""
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            # map 'assistant' to 'model' for Gemini
            if role == "assistant":
                role = "model"
                
            content = msg.get("content", "")
            
            # The API generates based on the last message provided to 'generate_content'
            # The rest needs to be put in a history context if there's more than 1
            if i == len(messages) - 1:
                last_message = content
            else:
                gemini_history.append({"role": role, "parts": [content]})
                
        # Initialize
        meta = {}
        try:
            model = genai.GenerativeModel('gemini-1.5-flash', tools=tools)
        except Exception as e:
            model = genai.GenerativeModel('gemini-1.5-flash')
            meta["tools_enabled"] = False
            meta["tools_error"] = str(e)
        
        chat_session = model.start_chat(history=gemini_history)
        
        response = chat_session.send_message(last_message)
        
        # Parse output for tools
        tool_calls = []
        content = ""
        
        for part in response.parts:
            if hasattr(part, 'function_call') and part.function_call:
                # Part has a function call
                args = {}
                for key, val in part.function_call.args.items():
                    args[key] = val
                    
                tool_calls.append({
                    "name": part.function_call.name,
                    "arguments": args
                })
            elif hasattr(part, 'text') and part.text:
                content += part.text
                
        result = {
            "ok": True,
            "content": content.strip() if content else "",
            "tool_calls": tool_calls
        }
        
        if meta:
            result["meta"] = meta
            
        return result
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }
