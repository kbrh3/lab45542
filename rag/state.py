# -*- coding: utf-8 -*-
"""
rag/state.py
Encapsulation of the global `_STATE` dictionary and orchestration of initialization logic.
"""
import os
import glob
from typing import Dict, Any

from rag.config import caption_map
from rag.data_loader import extract_pdf_pages, load_images
from rag.indexer import build_tfidf_index_text, build_tfidf_index_images
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

    # Load PDFs
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    # Extract pages
    page_chunks = []
    for p in pdfs:
        page_chunks.extend(extract_pdf_pages(p))

    # Load images
    image_items = load_images(fig_dir)

    # Apply caption_map
    if caption_map:
        for it in image_items:
            if it.item_id in caption_map:
                it.caption = caption_map[it.item_id]

    # Build indexes
    text_vec, text_X = build_tfidf_index_text(page_chunks)
    img_vec, img_X = build_tfidf_index_images(image_items)

    # Improvement: Graceful file handling log instead of a hard crash loop. 
    if not page_chunks and not image_items:
        print("Warning: No data found. RAG pipeline initialized empty. Please put PDFs in data/pdfs/ and images in data/figures/.")

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
