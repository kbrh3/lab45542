# -*- coding: utf-8 -*-
"""
rag/state.py
Encapsulation of the global `_STATE` dictionary and orchestration of initialization logic.
"""
import os
import glob
from typing import Dict, Any

from rag.logger import ensure_logfile

_STATE: Dict[str, Any] = {
    "initialized": False,
    "data_dir": None,
    "pdf_dir": None,
    "fig_dir": None,
    "logs_dir": None,
    "log_file": None,
    "pdfs": [],
    "page_chunks": [],
    "image_items": [],
    "text_vec": None,
    "text_X": None,
    "img_vec": None,
    "img_X": None,
}

def init_pipeline(
    data_dir: str = "data",
    logs_dir: str = "logs",
    log_file: str = "query_metrics.csv",
) -> Dict[str, Any]:
    """
    Call once at app start. Builds indexes.
    Safe to call multiple times; rebuilds only if data_dir changes.
    """
    global _STATE

    data_dir = str(data_dir)
    logs_dir = str(logs_dir)
    log_path = os.path.join(logs_dir, log_file)

    pdf_dir = os.path.join(data_dir, "pdfs")
    fig_dir = os.path.join(data_dir, "figures")

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # If already initialized for same dirs, do nothing
    if _STATE["initialized"] and _STATE["data_dir"] == data_dir and _STATE["logs_dir"] == logs_dir:
        return _STATE

    # Bypassed local PDF loading as Snowflake is the primary source
    pdfs = []
    page_chunks = []
    image_items = []
    text_vec, text_X, img_vec, img_X = None, None, None, None

    _STATE.update({
        "initialized": True,
        "data_dir": data_dir,
        "pdf_dir": pdf_dir,
        "fig_dir": fig_dir,
        "logs_dir": logs_dir,
        "log_file": log_path,
        "pdfs": pdfs,
        "page_chunks": page_chunks,
        "image_items": image_items,
        "text_vec": text_vec,
        "text_X": text_X,
        "img_vec": img_vec,
        "img_X": img_X,
    })

    ensure_logfile(_STATE["log_file"])
    return _STATE

def get_state() -> Dict[str, Any]:
    return _STATE
