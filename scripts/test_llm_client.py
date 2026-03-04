import os
import sys
import json

# Setup import path for direct script execution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from agent.llm_client import chat

def main():
    print("=== Testing LLM Client ===")
    
    # 1. Test Without API Key (Fail Safe)
    print("\n--- 1. Testing without API Key ---")
    original_key = os.environ.pop("GEMINI_API_KEY", None)
    
    res1 = chat([{"role": "user", "content": "What is the capital of France?"}])
    print(f"Result without Key: {json.dumps(res1, indent=2)}")
    
    # 2. Test Package Import Check
    print("\n--- 2. Testing Package Check (If Missing) ---")
    import agent.llm_client
    
    # Temporarily spoof missing package
    agent.llm_client.HAS_GENAI = False 
    
    res2 = chat([{"role": "user", "content": "Hello?"}])
    print(f"Result without Package: {json.dumps(res2, indent=2)}")
    
    # Restore package check
    agent.llm_client.HAS_GENAI = True
    
    # 3. Test With API Key (If Available)
    print("\n--- 3. Testing with API Key (if available) ---")
    if original_key:
        os.environ["GEMINI_API_KEY"] = original_key
        res3 = chat([{"role": "user", "content": "What is 2 + 2?"}])
        print(f"Result with Key: {json.dumps(res3, indent=2)}")
    else:
        print("Skipping active API key test because no native GEMINI_API_KEY was found in environment.")
        
    print("\nTest execution completed with zero crashes.")

if __name__ == "__main__":
    main()
