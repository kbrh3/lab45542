# -*- coding: utf-8 -*-
"""
rag/pipeline.py

Lab 4 pipeline module (based on your Lab 3 code), refactored by chat GPT to run in a repo:
- PDFs in:   data/pdfs/
- Images in: data/figures/
- Logs in:   logs/query_metrics.csv

This file is safe to import from Streamlit (no notebook/Colab-only commands).
"""

import os, re, glob, json, time, csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import fitz  # pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ENABLE_ERAG_EVAL = os.environ.get("ENABLE_ERAG_EVAL", "false").lower() == "true"


# -----------------------------
# Config (keep consistent w Lab3)
# -----------------------------
MISSING_EVIDENCE_MSG = "Not enough evidence in the retrieved context."

TOP_K_TEXT     = 5
TOP_K_IMAGES   = 3
TOP_K_EVIDENCE = 8
ALPHA          = 0.5  # 0.0=images dominate, 1.0=text dominates

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


# -----------------------------
# Mini gold set (project-aligned)
# -----------------------------
mini_gold = [
    {
        "query_id": "Q1",
        "question": "What is the overall SQLENS pipeline and what happens in each step?",
        "gold_evidence_ids": [],  # fill later
        "expected_concepts": ["error detector", "error selector", "error fixer"]
    },
    {
        "query_id": "Q2",
        "question": "What semantic error types are shown in the causal graph and what signals are used to detect them?",
        "gold_evidence_ids": [],  # fill later
        "expected_concepts": ["ambiguity", "evidence violation", "join predicate"]
    },
    {
        "query_id": "Q3",
        "question": "How does FACT reduce inconsistent hallucinations, and what kinds of hallucinations does it target?",
        "gold_evidence_ids": [],  # fill later
        "expected_concepts": ["inconsistent hallucinations", "fact text", "code-text training"]
    },
    {
        "query_id": "Q4",
        "question": "Using the figure of the SQLENS pipeline, list the pipeline stages in order.",
        "gold_evidence_ids": [],  # fill later with screenshot filename(s)
        "expected_concepts": ["error detector", "error selector", "error fixer", "sql auditor"]
    },
    {
        "query_id": "Q5",
        "question": "Who won the FIFA World Cup in 2050?",
        "gold_evidence_ids": ["N/A"],
        "expected_concepts": []
    },
]


# -----------------------------
# Captions (carry over from Lab3)
# -----------------------------
caption_map = {
    "Screenshot 2026-02-12 085908.png": "SQLENS pipeline: Error Detector -> Error Selector -> Error Fixer -> SQL Auditor",
    "Screenshot 2026-02-12 085920.png": "Causal graph: semantic errors and DB/LLM signals (ambiguity, evidence violation, join predicate, join tree)",
    "Screenshot 2026-02-12 085933.png": "Signal aggregation via weak supervision: labeling functions -> generative model -> correctness prediction",
    "Screenshot 2026-02-12 090033.png": "ALIGNRAG overview: retrieval + critique synthesis + critique-driven alignment",
    "Screenshot 2026-02-12 090118.png": "FACT example: input/context conflicting hallucinations and correction",
    "Screenshot 2026-02-12 090126.png": "FACT overview: filtering fact text + alternating code-text training + quality assessment",
    "Screenshot 2026-02-12 090137.png": "FACT simplified example: text segment paired with code representation",
}


# -----------------------------
# Data classes (Lab3)
# -----------------------------
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


# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    s = s or ""
    return re.sub(r"\s+", " ", s).strip()


def extract_pdf_pages(pdf_path: str) -> List[TextChunk]:
    doc_id = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
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
    for p in sorted(glob.glob(os.path.join(fig_dir, "*.*"))):
        if not p.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        base = os.path.basename(p)
        caption = os.path.splitext(base)[0].replace("_", " ")
        items.append(ImageItem(item_id=base, path=p, caption=caption))
    return items


def build_tfidf_index_text(chunks: List[TextChunk]):
    corpus = [c.text for c in chunks]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(corpus)
    X = normalize(X)
    return vec, X


def build_tfidf_index_images(items: List[ImageItem]):
    corpus = [it.caption for it in items]
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(corpus)
    X = normalize(X)
    return vec, X


def tfidf_retrieve(query: str, vec: TfidfVectorizer, X, top_k: int = 5):
    q = vec.transform([query])
    q = normalize(q)
    scores = (X @ q.T).toarray().ravel()
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]


def _normalize_scores(pairs):
    if not pairs:
        return []
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if abs(hi - lo) < 1e-12:
        return [(i, 1.0) for i, _ in pairs]
    return [(i, (s - lo) / (hi - lo)) for i, s in pairs]


# -----------------------------
# Pipeline state (built once)
# -----------------------------
_STATE: Dict[str, Any] = {
    "initialized": False,
    "data_dir": None,
    "pdf_dir": None,
    "fig_dir": None,
    "logs_dir": None,
    "base_logs_dir": None,
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
    logs_dir: str = "artifacts/runs",
    log_file: str = "query_metrics.csv",
    use_run_id: bool = True,
) -> Dict[str, Any]:
    """
    Call once at app start. Builds indexes.
    Safe to call multiple times; rebuilds only if data_dir changes.
    """
    global _STATE

    data_dir = str(data_dir)
    logs_dir = str(logs_dir)

    # If already initialized for same dirs, do nothing
    if _STATE["initialized"] and _STATE["data_dir"] == data_dir and _STATE.get("base_logs_dir") == logs_dir:
        return _STATE

    if use_run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_logs_dir = os.path.join(logs_dir, run_id)
    else:
        actual_logs_dir = logs_dir

    log_path = os.path.join(actual_logs_dir, log_file)

    pdf_dir = os.path.join(data_dir, "pdfs")
    fig_dir = os.path.join(data_dir, "figures")

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(actual_logs_dir, exist_ok=True)

    # Load PDFs
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    # Extract pages
    page_chunks: List[TextChunk] = []
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
    text_vec, text_X = (None, None)
    if page_chunks:
        text_vec, text_X = build_tfidf_index_text(page_chunks)

    img_vec, img_X = (None, None)
    if image_items:
        img_vec, img_X = build_tfidf_index_images(image_items)

    if not page_chunks and not image_items:
        raise RuntimeError(
            "No data found. Put PDFs in data/pdfs/ and images in data/figures/."
        )

    _STATE.update({
        "initialized": True,
        "data_dir": data_dir,
        "pdf_dir": pdf_dir,
        "fig_dir": fig_dir,
        "logs_dir": actual_logs_dir,
        "base_logs_dir": logs_dir,
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


# -----------------------------
# Context + Answer (Lab3 style)
# -----------------------------
def build_context(
    question: str,
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


def simple_extractive_answer(context: str, n_lines: int = 3) -> str:
    lines = [ln for ln in context.splitlines() if ln.strip()]
    if not lines:
        return MISSING_EVIDENCE_MSG
    return "\n".join(lines[:n_lines])


def generate_answer(question: str, ctx: Dict[str, Any]) -> str:
    # Required missing-evidence behavior
    if not ctx["evidence"]:
        return MISSING_EVIDENCE_MSG

    best = max(ev.get("fused_score", 0.0) for ev in ctx["evidence"])
    if best < 0.05:
        return MISSING_EVIDENCE_MSG

    # Extractive baseline (grounded by construction)
    return simple_extractive_answer(ctx["context"], n_lines=3)


def run_pipeline(
    question: str,
    retrieval_mode: str = "mm",
    top_k_text: int = TOP_K_TEXT,
    top_k_images: int = TOP_K_IMAGES,
    top_k_evidence: int = TOP_K_EVIDENCE,
    alpha: float = ALPHA,
) -> Dict[str, Any]:
    """
    Main call used by Streamlit.
    retrieval_mode:
      - "mm"        : multimodal (text+image captions) fusion
      - "text_only" : text pages only
    """
    use_multimodal = (retrieval_mode == "mm")
    ctx = build_context(
        question,
        top_k_text=top_k_text,
        top_k_images=top_k_images,
        top_k_evidence=top_k_evidence,
        alpha=alpha,
        use_multimodal=use_multimodal
    )
    answer = generate_answer(question, ctx)
    return {"answer": answer, "ctx": ctx}


# -----------------------------
# Logging + Metrics (Lab4)
# -----------------------------
def ensure_logfile(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp", "query_id", "retrieval_mode", "top_k_evidence", "latency_ms",
        "Precision@5", "Recall@10",
        "evidence_ids_returned", "gold_evidence_ids",
        "faithfulness_pass", "missing_evidence_behavior"
    ]
    if ENABLE_ERAG_EVAL:
        header.extend(["erag_P_1", "erag_P_3", "erag_P_5"])
        
    # Write header if file is missing OR exists but is empty / lacks a header row
    write_header = False
    if not p.exists():
        write_header = True
    else:
        try:
            with open(p, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                write_header = True
            elif ENABLE_ERAG_EVAL and "erag_P_1" not in first_line:
                # If we enabled eRAG but the file doesn't have the header, it's safer to just append anyway.
                # Production apps might rotate logs; we will just leave it.
                pass
        except Exception:
            write_header = True

    if write_header:
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def text_generator(queries_and_documents):
    outputs = {}
    for query, docs in queries_and_documents.items():
        outputs[query] = " ".join([str(d).lower() for d in docs])
    return outputs

def downstream_metric(generated_outputs, expected_outputs):
    scores = {}
    for query, gen_out in generated_outputs.items():
        exp_out = expected_outputs.get(query, [])
        matched = False
        for expected in exp_out:
            if expected.lower() in gen_out.lower():
                matched = True
                break
        scores[query] = 1.0 if matched else 0.0
    return scores

def precision_at_k_ids(retrieved_ids: List[str], gold_ids: List[str], k: int = 5):
    if not gold_ids or gold_ids == ["N/A"]:
        return np.nan
    topk = retrieved_ids[:k]
    return len(set(topk) & set(gold_ids)) / float(k)


def recall_at_k_ids(retrieved_ids: List[str], gold_ids: List[str], k: int = 10):
    if not gold_ids or gold_ids == ["N/A"]:
        return np.nan
    topk = retrieved_ids[:k]
    denom = max(1, len(set(gold_ids)))
    return len(set(topk) & set(gold_ids)) / float(denom)


def faithfulness_heuristic(answer: str, evidence: List[Dict[str, Any]]):
    if answer.strip() == MISSING_EVIDENCE_MSG:
        return True
    tags = [ev.get("citation_tag", "") for ev in evidence[:5]]
    return any(t and t in answer for t in tags)


def missing_evidence_behavior(answer: str, evidence: List[Dict[str, Any]]):
    has_ev = bool(evidence) and max(ev.get("fused_score", 0.0) for ev in evidence) >= 0.05
    if not has_ev:
        return "Pass" if answer.strip() == MISSING_EVIDENCE_MSG else "Fail"
    return "Pass" if answer.strip() != MISSING_EVIDENCE_MSG else "Fail"


def run_query_and_log(
    query_item: Dict[str, Any],
    retrieval_mode: str = "mm",
) -> Dict[str, Any]:
    """
    Required Lab 4 function:
    - runs pipeline
    - computes metrics (if gold exists)
    - appends to logs/query_metrics.csv
    """
    if not _STATE["initialized"]:
        init_pipeline()

    question = query_item["question"]
    gold_ids = query_item.get("gold_evidence_ids", [])

    t0 = time.time()
    out = run_pipeline(question, retrieval_mode=retrieval_mode)
    latency_ms = (time.time() - t0) * 1000.0

    ctx = out["ctx"]
    answer = out["answer"]

    evidence_ids = [ev["id"] for ev in ctx["evidence"]]
    p5  = precision_at_k_ids(evidence_ids, gold_ids, k=5)
    r10 = recall_at_k_ids(evidence_ids, gold_ids, k=10)

    faithful = faithfulness_heuristic(answer, ctx["evidence"])
    meb = missing_evidence_behavior(answer, ctx["evidence"])
    
    erag_metrics = {"P_1": "", "P_3": "", "P_5": ""}
    if ENABLE_ERAG_EVAL:
        try:
            import erag
            texts = [ev["text"] for ev in ctx["evidence"] if ev["modality"] == "text"]
            expected_concepts = query_item.get("expected_concepts", [])
            retrieval_results = {question: texts}
            expected_outputs = {question: expected_concepts}
            
            # Use deterministic functions
            eval_results = erag.eval(
                retrieval_results=retrieval_results,
                expected_outputs=expected_outputs,
                text_generator=text_generator,
                downstream_metric=downstream_metric,
                retrieval_metrics={"P_1", "P_3", "P_5"}
            )
            # eRAG returns {"aggregated": {...}, "per_input": {...}}
            if "aggregated" in eval_results:
                erag_metrics["P_1"] = eval_results["aggregated"].get("P_1", 0.0)
                erag_metrics["P_3"] = eval_results["aggregated"].get("P_3", 0.0)
                erag_metrics["P_5"] = eval_results["aggregated"].get("P_5", 0.0)
        except Exception as e:
            print(f"Warning: eRAG evaluation failed: {e}")

    row = [
        datetime.now(timezone.utc).isoformat(),
        query_item["query_id"],
        retrieval_mode,
        TOP_K_EVIDENCE,
        round(latency_ms, 2),
        p5,
        r10,
        json.dumps(evidence_ids),
        json.dumps(gold_ids),
        "Yes" if faithful else "No",
        meb
    ]
    
    if ENABLE_ERAG_EVAL:
        row.extend([erag_metrics["P_1"], erag_metrics["P_3"], erag_metrics["P_5"]])

    ensure_logfile(_STATE["log_file"])
    with open(_STATE["log_file"], "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    res = {
        "answer": answer,
        "ctx": ctx,
        "p5": p5,
        "r10": r10,
        "latency_ms": latency_ms,
        "faithful": faithful,
        "meb": meb
    }
    if ENABLE_ERAG_EVAL:
        res["erag"] = erag_metrics
    return res


# -----------------------------
# Optional debug helpers
# -----------------------------
def list_pdf_pages() -> List[str]:
    """
    Returns chunk IDs like: MyPaper.pdf::p1, MyPaper.pdf::p2 ...
    Useful to fill gold_evidence_ids correctly.
    """
    if not _STATE["initialized"]:
        init_pipeline()

    out = []
    for p in _STATE["pdfs"]:
        doc = fitz.open(p)
        doc_id = os.path.basename(p)
        for i in range(len(doc)):
            out.append(f"{doc_id}::p{i+1}")
    return out


def find_pages_containing(term: str, limit: int = 30) -> List[str]:
    if not _STATE["initialized"]:
        init_pipeline()

    term = (term or "").lower()
    hits = []
    for ch in _STATE["page_chunks"]:
        if term in ch.text.lower():
            hits.append(ch.chunk_id)
            if len(hits) >= limit:
                break
    return hits


def get_mini_gold() -> List[Dict[str, Any]]:
    return mini_gold

