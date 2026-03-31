# -*- coding: utf-8 -*-
"""
rag/config.py
Configuration constants and predefined settings for the RAG pipeline.
"""
import numpy as np

# Config
MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."
TOP_K_TEXT     = 5
TOP_K_IMAGES   = 3
TOP_K_EVIDENCE = 8
ALPHA          = 0.5  # 0.0=images dominate, 1.0=text dominates

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# Mini gold set — PolicyPulse legislative evaluation queries
mini_gold = [
    {
        "query_id": "Q1",
        "question": "What bills are related to education?",
        "gold_evidence_ids": []
    },
    {
        "query_id": "Q2",
        "question": "What is the latest action on healthcare-related bills?",
        "gold_evidence_ids": []
    },
    {
        "query_id": "Q3",
        "question": "Which bills are currently in committee?",
        "gold_evidence_ids": []
    },
    {
        "query_id": "Q4",
        "question": "Summarize recent legislation related to public safety",
        "gold_evidence_ids": []
    },
    {
        "query_id": "Q5",
        "question": "Who won the FIFA World Cup in 2050?",
        "gold_evidence_ids": ["N/A"]
    },
]

# Captions mapping
caption_map = {
    "Screenshot 2026-02-12 085908.png": "SQLENS pipeline: Error Detector -> Error Selector -> Error Fixer -> SQL Auditor",
    "Screenshot 2026-02-12 085920.png": "Causal graph: semantic errors and DB/LLM signals (ambiguity, evidence violation, join predicate, join tree)",
    "Screenshot 2026-02-12 085933.png": "Signal aggregation via weak supervision: labeling functions -> generative model -> correctness prediction",
    "Screenshot 2026-02-12 090033.png": "ALIGNRAG overview: retrieval + critique synthesis + critique-driven alignment",
    "Screenshot 2026-02-12 090118.png": "FACT example: input/context conflicting hallucinations and correction",
    "Screenshot 2026-02-12 090126.png": "FACT overview: filtering fact text + alternating code-text training + quality assessment",
    "Screenshot 2026-02-12 090137.png": "FACT simplified example: text segment paired with code representation",
}
