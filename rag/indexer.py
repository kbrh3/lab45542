# -*- coding: utf-8 -*-
"""
rag/indexer.py
Functions for building and managing TF-IDF indexes for texts and images.
"""
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.models import TextChunk, ImageItem

def build_tfidf_index_text(chunks: List[TextChunk]) -> Tuple[Optional[TfidfVectorizer], Optional[object]]:
    if not chunks:
        return None, None
    corpus = [c.text for c in chunks]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(corpus)
    X = normalize(X)
    return vec, X

def build_tfidf_index_images(items: List[ImageItem]) -> Tuple[Optional[TfidfVectorizer], Optional[object]]:
    if not items:
        return None, None
    corpus = [it.caption for it in items]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(corpus)
    X = normalize(X)
    return vec, X
