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

    text_hits = []
    if text_vec is not None and text_X is not None and page_chunks:
        text_hits = tfidf_retrieve(question, text_vec, text_X, top_k=top_k_text)

    img_hits = []
    if use_multimodal and img_vec is not None and img_X is not None and image_items:
        img_hits = tfidf_retrieve(question, img_vec, img_X, top_k=top_k_images)

    text_norm = _normalize_scores(text_hits)
    img_norm  = _normalize_scores(img_hits)

    fused: List[Dict[str, Any]] = []

    # Text evidence
    for idx, s in text_norm:
        ch = page_chunks[idx]
        fused.append({
            "modality": "text",
            "id": ch.chunk_id,
            "raw_score": float(dict(text_hits).get(idx, 0.0)),
            "fused_score": float(alpha * s),
            "text": ch.text,
            "path": None,
            "citation_tag": f"[{ch.chunk_id}]",
            "source": ch.doc_id,
            "page_num": ch.page_num,
        })

    # Image evidence
    for idx, s in img_norm:
        it = image_items[idx]
        fused.append({
            "modality": "image",
            "id": it.item_id,
            "raw_score": float(dict(img_hits).get(idx, 0.0)),
            "fused_score": float((1.0 - alpha) * s),
            "text": it.caption,
            "path": it.path,
            "citation_tag": f"[{it.item_id}]",
            "source": it.item_id,
            "page_num": None,
        })

    fused = sorted(fused, key=lambda d: d["fused_score"], reverse=True)[:top_k_evidence]

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
