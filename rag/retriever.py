# -*- coding: utf-8 -*-
"""
rag/retriever.py
TF-IDF retrieval, score normalization, and multimodal fusion logic.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.models import TextChunk, ImageItem
from rag.config import TOP_K_TEXT, TOP_K_IMAGES, TOP_K_EVIDENCE, ALPHA

def tfidf_retrieve(query: str, vec: TfidfVectorizer, X, top_k: int = 5) -> List[Tuple[int, float]]:
    q = vec.transform([query])
    q = normalize(q)
    
    # Catch empty query vector or bad shapes safely
    try:
        scores = (X @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return []

def _normalize_scores(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if abs(hi - lo) < 1e-12:
        return [(i, 1.0) for i, _ in pairs]
    return [(i, (s - lo) / (hi - lo)) for i, s in pairs]

def build_context(
    question: str,
    page_chunks: List[TextChunk],
    image_items: List[ImageItem],
    text_vec: TfidfVectorizer,
    text_X,
    img_vec: TfidfVectorizer,
    img_X,
    top_k_text: int = TOP_K_TEXT,
    top_k_images: int = TOP_K_IMAGES,
    top_k_evidence: int = TOP_K_EVIDENCE,
    alpha: float = ALPHA,
    use_multimodal: bool = True,
) -> Dict[str, Any]:

    from rag.snowflake_retriever import retrieve
    
    # Bypass original local memory indexes and fetch directly via Snowflake SQL matching
    fused = retrieve(question, top_k=top_k_evidence)

    ctx_lines = []
    image_paths = []
    for ev in fused:
        if ev["modality"] == "text":
            snippet = (ev["text"] or "")[:260].replace("\n", " ")
            ctx_lines.append(f"{ev['citation_tag']} {snippet}")
        else:
            ctx_lines.append(f"{ev['citation_tag']} caption={ev['text']}")
            if ev.get("path"):
                image_paths.append(ev["path"])

    return {
        "question": question,
        "context": "\n".join(ctx_lines),
        "image_paths": image_paths,
        "evidence": fused,
        "alpha": alpha
    }
