import os
import sys
import json

# Setup import path for direct script execution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from agent.runner import run_agent

def main():
    print("=== Testing Agent Runner ===")
    
    # Check if we have an API key right now
    has_key = os.getenv("GEMINI_API_KEY") is not None
    if not has_key:
        print("Warning: GEMINI_API_KEY is missing. Agent should fail gracefully and return structured response.")
    
    queries = [
        ("Simple (One Tool)", "Show top committees by total bills."),
        ("Medium (Two Tools)", "What is the SQLENS pipeline? Also show bills by year."),
        ("Complex (Stats & Summary)", "Compare bill counts in 2010 vs 2020 and summarize the difference.")
    ]
    
    for name, query in queries:
        print(f"\n--- Running Scenario: {name} ---")
        print(f"User Query: '{query}'")
        
        response = run_agent(user_message=query)
        
        print("\n[Agent Response]")
        print(f"Ok Status: {response['metrics'].get('agent_ok')}")
        print(f"Answer: {response.get('answer')}")
        print(f"Steps Taken: {response['metrics'].get('agent_steps')}")
        print(f"Number of Errors: len({response.get('errors')}) -> {response.get('errors')}")
        print(f"Tool Trace items: {len(response.get('tool_trace', []))}")
        
        for i, t in enumerate(response.get('tool_trace', [])):
            print(f"  [{i+1}] {t['tool_name']} (ok={t['ok']})")
            
    print("\nTest execution completed with zero crashes.")

if __name__ == "__main__":
    main()
