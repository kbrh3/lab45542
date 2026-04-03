# -*- coding: utf-8 -*-
"""
rag/__init__.py
Main entry point orchestrating the execution of queries through the RAG sub-components.
"""
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import fitz
import os

from rag.config import TOP_K_TEXT, TOP_K_IMAGES, TOP_K_EVIDENCE, ALPHA, MISSING_EVIDENCE_MSG
from rag.state import init_pipeline, get_state
from rag.retriever import build_context
from rag.generator import generate_answer
from rag.evaluator import precision_at_k_ids, recall_at_k_ids, faithfulness_heuristic, missing_evidence_behavior
from rag.logger import log_query_metrics

def run_pipeline(
    question: str,
    retrieval_mode: str = "mm",
    top_k_text: int = TOP_K_TEXT,
    top_k_images: int = TOP_K_IMAGES,
    top_k_evidence: int = TOP_K_EVIDENCE,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """
    Core pipeline logic combining modules.
    """
    state = get_state()
    use_multimodal = (retrieval_mode == "mm")
    
    print(f"--- DEBUG: rag/__init__.py -> run_pipeline() calling build_context().")

    ctx = build_context(
        question,
        page_chunks=state["page_chunks"],
        image_items=state["image_items"],
        text_vec=state["text_vec"],
        text_X=state["text_X"],
        img_vec=state["img_vec"],
        img_X=state["img_X"],
        top_k_text=top_k_text,
        top_k_images=top_k_images,
        top_k_evidence=top_k_evidence,
        alpha=alpha,
        use_multimodal=use_multimodal
    )
    
    print(f"--- DEBUG: rag/__init__.py -> run_pipeline() len(ctx['evidence']) right before generate_answer: {len(ctx.get('evidence', []))}")
    answer = generate_answer(question, ctx)
    return {"answer": answer, "ctx": ctx}

def run_query_and_log(
    query_item: Dict[str, Any],
    retrieval_mode: str = "mm",
) -> Dict[str, Any]:
    """
    API Interface: runs pipeline, computes metrics, and logs to CSV.
    """
    state = get_state()
    if not state["initialized"]:
        init_pipeline()
        state = get_state()

    question = query_item["question"]
    gold_ids = query_item.get("gold_evidence_ids", [])

    print(f"--- DEBUG: rag/__init__.py -> run_query_and_log() calling run_pipeline(). Question: {question}")

    t0 = time.time()
    out = run_pipeline(question, retrieval_mode=retrieval_mode)
    latency_ms = (time.time() - t0) * 1000.0

    ctx = out["ctx"]
    answer = out["answer"]

    evidence_ids = [ev["id"] for ev in ctx["evidence"]]
    p5  = precision_at_k_ids(evidence_ids, gold_ids, k=5)
    r10 = recall_at_k_ids(evidence_ids, gold_ids, k=10)

    faithful = faithfulness_heuristic(answer, ctx["evidence"])
    meb = missing_evidence_behavior(answer, ctx["evidence"])

    row = [
        datetime.now(timezone.utc).isoformat(),
        query_item["query_id"],
        retrieval_mode,
        TOP_K_EVIDENCE,
        round(latency_ms, 2),
        p5,
        r10,
        json.dumps(evidence_ids),
        json.dumps(gold_ids),
        "Yes" if faithful else "No",
        meb
    ]

    log_query_metrics(state["log_file"], row)

    return {
        "answer": answer,
        "ctx": ctx,
        "p5": p5,
        "r10": r10,
        "latency_ms": latency_ms,
        "faithful": faithful,
        "meb": meb
    }

# Debug helpers re-exported to maintain backwards compatibility
def list_pdf_pages() -> List[str]:
    state = get_state()
    if not state["initialized"]:
        init_pipeline()
        state = get_state()

    out = []
    for p in state["pdfs"]:
        try:
            doc = fitz.open(p)
            doc_id = os.path.basename(p)
            for i in range(len(doc)):
                out.append(f"{doc_id}::p{i+1}")
        except Exception:
            pass
    return out

def find_pages_containing(term: str, limit: int = 30) -> List[str]:
    state = get_state()
    if not state["initialized"]:
        init_pipeline()
        state = get_state()

    term = (term or "").lower()
    hits = []
    for ch in state["page_chunks"]:
        if term in ch.text.lower():
            hits.append(ch.chunk_id)
            if len(hits) >= limit:
                break
    return hits

def get_mini_gold() -> List[Dict[str, Any]]:
    from rag.config import mini_gold
    return mini_gold
