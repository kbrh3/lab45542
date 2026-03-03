import os
import sys
import json
from dotenv import load_dotenv

# Add the root directory to sys.path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import rag_query, snowflake_query, bills_analytics, summarize_text
from tool_schemas import TOOLS

def test_tools():
    # Load env vars (mostly for Snowflake)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)
    
    print("=== Testing tool_schemas ===")
    print(f"Loaded {len(TOOLS)} tool schemas.\n")
    
    print("=== Testing rag_query ===")
    res_rag = rag_query("What is the overall SQLENS pipeline and what happens in each step?")
    if res_rag.get("ok"):
        print("rag_query SUCCESS.")
        print("Answer preview:", res_rag["data"]["answer"][:120] + "...")
        print("Metrics:", res_rag["data"]["metrics"])
    else:
        print("rag_query FAILED.")
        print("Error:", res_rag.get("error"))
    print(f"Meta: {res_rag['meta']}\n")
    
    print("=== Testing bills_analytics ===")
    res_bills = bills_analytics("top_committees")
    if res_bills.get("ok"):
        print("bills_analytics SUCCESS.")
        print("Rows returned:", len(res_bills["data"]["rows"]))
        if res_bills["data"]["rows"]:
            print("Preview top row:", res_bills["data"]["rows"][0])
    else:
        print("bills_analytics FAILED (expected if Snowflake env vars are missing).")
        print("Error:", res_bills.get("error"))
    print(f"Meta: {res_bills['meta']}\n")
    
    print("=== Testing summarize_text ===")
    sample_text = "This is a very long text " * 20
    res_sum = summarize_text(sample_text, max_words=10)
    if res_sum.get("ok"):
        print("summarize_text SUCCESS.")
        print("Summary:", res_sum["data"]["summary"])
    else:
        print("summarize_text FAILED.")
        print("Error:", res_sum.get("error"))
    print(f"Meta: {res_sum['meta']}\n")

if __name__ == "__main__":
    test_tools()
