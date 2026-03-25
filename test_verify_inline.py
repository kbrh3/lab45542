import os
import sys

# ensure path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from api.server import app
from rag.pipeline import run_query_and_log
from agent.runner import run_agent

print("Backend syntax checks passed. Health endpoint should be fine.")

print("=== Testing Classic Query ===")
q_item = {"query_id": "test", "question": "What is the status of the Clean Energy act?", "gold_evidence_ids": []}
res = run_query_and_log(q_item, retrieval_mode="text_only")
print("Response:", res["answer"])
print("Evidences:", len(res["ctx"]["evidence"]))
for ev in res["ctx"]["evidence"][:1]:
    print("Example:", ev["citation_tag"])

print("=== Testing Agent ===")
a_res = run_agent(user_message="Summarize the latest action on education bills", max_steps=2)
print("Agent Status:", a_res["metrics"]["agent_ok"])
print("Agent Response:", a_res["answer"])
print("Done.")
