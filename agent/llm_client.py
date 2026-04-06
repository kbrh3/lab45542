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
                
        # Fallback models safely prioritized
        # fallback_models = ['gemini-2.5-flash', 'gemini-3-flash-preview', 'gemini-2.0-flash']
        fallback_models = ['gemini-3.1-flash-lite-preview']
        
        # Startup logs
        sdk_version = getattr(genai, '__version__', 'unknown')
        print(f"--- DEBUG: Gemini SDK matching google.generativeai (Version: {sdk_version})")
        print("--- DEBUG: API Version in use: likely v1beta based on SDK defaults.")
        
        response = None
        meta = {}
        
        for model_name in fallback_models:
            print(f"--- DEBUG: Attempting chat with model: {model_name}")
            try:
                try:
                    model = genai.GenerativeModel(model_name, tools=tools)
                except Exception as e:
                    model = genai.GenerativeModel(model_name)
                    meta[f"{model_name}_tools_error"] = str(e)
                
                chat_session = model.start_chat(history=gemini_history)
                response = chat_session.send_message(last_message)
                
                print(f"--- DEBUG: Selected model: {model_name} successfully executed generateContent.")
                break  # Complete success
            except Exception as e:
                err_str = str(e)
                print(f"--- DEBUG: Model {model_name} failed: {err_str}")
                if "404" in err_str or "not found" in err_str.lower():
                    continue  # Try next fallback automatically
                else:
                    # Depending on error, it might be quota, but we fallback anyway
                    continue
        
        if not response:
            raise ValueError(f"All fallback models failed to generate content. Last error: {err_str}")
        
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
