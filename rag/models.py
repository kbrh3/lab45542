# -*- coding: utf-8 -*-
"""
rag/models.py
Data classes for the RAG pipeline.
"""
from dataclasses import dataclass

@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str

@dataclass
class ImageItem:
    item_id: str
    path: str
    caption: str
