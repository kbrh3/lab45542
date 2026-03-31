# -*- coding: utf-8 -*-
"""
rag/generator.py
Answer generation logic for extracting information out of context.
"""
from typing import Dict, Any, List
from rag.config import MISSING_EVIDENCE_MSG


def _format_bill_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Produce a human-readable summary from structured bill evidence items."""
    lines = []
    for ev in evidence:
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        tag = ev.get("citation_tag", "")
        # Each bill entry is already formatted as key: value lines
        lines.append(f"{tag}\n{text}")
    return "\n\n".join(lines)


def generate_answer(question: str, ctx: Dict[str, Any]) -> str:
    evidence = ctx.get("evidence", [])

    if not evidence:
        return MISSING_EVIDENCE_MSG

    best = max((ev.get("fused_score", 0.0) for ev in evidence), default=0.0)
    if best < 0.15:
        return MISSING_EVIDENCE_MSG

    # Build a grounded extractive answer from retrieved bill evidence
    summary = _format_bill_evidence(evidence)
    if not summary.strip():
        return MISSING_EVIDENCE_MSG

    intro = f"Based on retrieved legislative records for your query — \"{question}\":\n\n"
    return intro + summary
