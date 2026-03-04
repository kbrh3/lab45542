from typing import TypedDict, List, Dict, Any, Optional

class ToolTraceItem(TypedDict):
    tool_name: str
    args: Dict[str, Any]
    ok: bool
    latency_ms: float
    error: Optional[str]

class AgentResponse(TypedDict):
    answer: str
    evidence: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    missing_evidence_msg: str
    tool_trace: List[ToolTraceItem]
    errors: List[str]

class AgentQueryIn(TypedDict, total=False):
    query_id: str
    question: str
    retrieval_mode: str
    top_k: int
    use_agent: bool
