import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.server import query, QueryIn
from dotenv import load_dotenv
import json

load_dotenv()

def test_flow():
    q = QueryIn(
        query_id="test_002",
        question="What bills are related to education?",
        retrieval_mode="mm"
    )
    print(f"--- DEBUG IN PROXY: Sending Payload -> {q.dict()}")
    result = query(q)
    print(f"--- DEBUG IN PROXY: Final JSON Response -> {json.dumps(result, default=str)}")

if __name__ == "__main__":
    test_flow()
