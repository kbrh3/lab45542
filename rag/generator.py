# -*- coding: utf-8 -*-
"""
rag/generator.py
Answer generation logic for extracting information out of context.
"""
from typing import Dict, Any
from rag.config import MISSING_EVIDENCE_MSG

def simple_extractive_answer(context: str, n_lines: int = 3) -> str:
    lines = [ln for ln in context.splitlines() if ln.strip()]
    if not lines:
        return MISSING_EVIDENCE_MSG
    return "\n".join(lines[:n_lines])


def generate_answer(question: str, ctx: Dict[str, Any]) -> str:
    # Required missing-evidence behavior
    if not ctx.get("evidence"):
        return MISSING_EVIDENCE_MSG

    best = max((ev.get("fused_score", 0.0) for ev in ctx["evidence"]), default=0.0)
    if best < 0.15:
        return MISSING_EVIDENCE_MSG

    # Extractive baseline (grounded by construction)
    return simple_extractive_answer(ctx.get("context", ""), n_lines=3)
