import os
import json
from dotenv import load_dotenv
load_dotenv()

from rag.pipeline import run_query_and_log

query_item = {
    "query_id": "test_1",
    "question": "What bills are related to education?",
    "gold_evidence_ids": []
}

try:
    res = run_query_and_log(query_item, retrieval_mode="mm")
    print(json.dumps(res, indent=2))
except Exception as e:
    print("Error:", e)
