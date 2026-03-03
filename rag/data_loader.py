# -*- coding: utf-8 -*-
"""
rag/data_loader.py
Functions for reading PDFs and loading multimodal assets, featuring improved error handling.
"""
import os
import re
import glob
from typing import List
import fitz  # pymupdf

from rag.models import TextChunk, ImageItem

def clean_text(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s).strip()

def extract_pdf_pages(pdf_path: str) -> List[TextChunk]:
    doc_id = os.path.basename(pdf_path)
    # Improvement: Graceful error handling for corrupted or unreadable PDFs
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Warning: Failed to load PDF {pdf_path}. Exception: {e}")
        return []
        
    out: List[TextChunk] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = clean_text(page.get_text("text"))
        if text:
            out.append(TextChunk(
                chunk_id=f"{doc_id}::p{i+1}",
                doc_id=doc_id,
                page_num=i+1,
                text=text
            ))
    return out

def load_images(fig_dir: str) -> List[ImageItem]:
    items: List[ImageItem] = []
    # Improvement: Graceful error handling if directory does not exist
    if not os.path.exists(fig_dir):
        print(f"Warning: Image directory {fig_dir} does not exist.")
        return items
        
    for p in sorted(glob.glob(os.path.join(fig_dir, "*.*"))):
        if not p.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        base = os.path.basename(p)
        caption = os.path.splitext(base)[0].replace("_", " ")
        items.append(ImageItem(item_id=base, path=p, caption=caption))
    return items
