import os
import sys

# ensure path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from fastapi.testclient import TestClient
from api.server import app, BACKEND_API_KEY

client = TestClient(app)
headers = {"internal-api-key": BACKEND_API_KEY or "dummy"}

print("=== Testing /health ===")
r = client.get("/health")
print(f"Status: {r.status_code}")
print(r.json())

print("\n=== Testing Classic Query ===")
q_payload = {
    "query_id": "test1",
    "question": "Which bills discuss internet or technology?",
    "retrieval_mode": "text_only",
    "top_k": 3
}
r2 = client.post("/query", json=q_payload, headers=headers)
print(f"Status: {r2.status_code}")
try:
    data = r2.json()
    print("Answer:", data.get("answer"))
    evidence = data.get("evidence", [])
    print(f"Evidence retrieved: {len(evidence)} items")
    for ev in evidence[:1]:
        print("Sample Evidence:", ev.get("citation_tag"))
except Exception as e:
    print("Error parsing query response:", e)

print("\n=== Testing Agent Mode ===")
a_payload = {
    "message": "What is the latest action on healthcare bills?",
    "max_steps": 3
}
r3 = client.post("/agent_query", json=a_payload, headers=headers)
print(f"Status: {r3.status_code}")
try:
    data = r3.json()
    print("Agent Answer:")
    print(data.get("answer"))
    print(f"Agent Steps: {data.get('metrics', {}).get('agent_steps')}")
except Exception as e:
    print("Error parsing agent query:", e)
