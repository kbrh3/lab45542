import sys
import os
import json

# Setup import path for direct script execution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from agent.tool_registry import execute_tool
from tools import snowflake_query

def main():
    print("=== Testing Tool Registry execute_tool ===")
    
    # 1. Test RAG Query
    print("\n--- 1. Testing 'rag_query' ---")
    rag_args = {
        "query": "What is the overall SQLENS pipeline and what happens in each step?",
        "retrieval_mode": "mm",
        "top_k": 5
    }
    rag_res, rag_trace = execute_tool("rag_query", rag_args)
    print(f"RAG Trace: {json.dumps(rag_trace, indent=2)}")
    
    # Let's peek into the data slightly so we don't blow up stdout with embedding vectors and docs
    if rag_res.get("ok"):
        print("RAG executed successfully.")
        answer = rag_res.get("data", {}).get("answer", "")
        print(f"Answer snippet: {answer[:100]}...")
    else:
        print(f"RAG Error (this is expected if no local data exists, but it shouldn't crash!): {rag_res.get('error')}")

    # 2. Test Snowflake Query without setting env vars to test safe failure and graceful handling
    print("\n--- 2. Testing 'bills_analytics' (Snowflake) without env vars ---")
    # Make sure we erase them to test failure
    os.environ.pop("SNOWFLAKE_USER", None)
    os.environ.pop("SNOWFLAKE_ACCOUNT", None)

    sf_args = {"metric": "top_committees"}
    sf_res, sf_trace = execute_tool("bills_analytics", sf_args)
    
    print(f"Snowflake Trace: {json.dumps(sf_trace, indent=2)}")
    print(f"Snowflake Tool Result: {json.dumps(sf_res, indent=2)}")

    print("\n--- 3. Testing an unknown tool ---")
    un_res, un_trace = execute_tool("fake_query", {})
    print(f"Unknown Tool Trace: {json.dumps(un_trace, indent=2)}")
    print(f"Unknown Tool Result: {json.dumps(un_res, indent=2)}")

    print("\nTest execution completed with zero crashes.")

if __name__ == "__main__":
    main()
