# -*- coding: utf-8 -*-
"""
rag/evaluator.py
Metrics calculation for RAG evaluation.
"""
import numpy as np
from typing import List, Dict, Any
from rag.config import MISSING_EVIDENCE_MSG

def precision_at_k_ids(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    if not gold_ids or gold_ids == ["N/A"]:
        return np.nan
    topk = retrieved_ids[:k]
    return len(set(topk) & set(gold_ids)) / float(k)

def recall_at_k_ids(retrieved_ids: List[str], gold_ids: List[str], k: int = 10) -> float:
    if not gold_ids or gold_ids == ["N/A"]:
        return np.nan
    topk = retrieved_ids[:k]
    denom = max(1, len(set(gold_ids)))
    return len(set(topk) & set(gold_ids)) / float(denom)

def faithfulness_heuristic(answer: str, evidence: List[Dict[str, Any]]) -> bool:
    if answer.strip() == MISSING_EVIDENCE_MSG:
        return True
    tags = [ev.get("citation_tag", "") for ev in evidence[:5]]
    return any(t and t in answer for t in tags)

def missing_evidence_behavior(answer: str, evidence: List[Dict[str, Any]]) -> str:
    has_ev = bool(evidence) and max((ev.get("fused_score", 0.0) for ev in evidence), default=0.0) >= 0.05
    if not has_ev:
        return "Pass" if answer.strip() == MISSING_EVIDENCE_MSG else "Fail"
    return "Pass" if answer.strip() != MISSING_EVIDENCE_MSG else "Fail"
