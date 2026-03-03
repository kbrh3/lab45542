# -*- coding: utf-8 -*-
"""
rag/logger.py
CSV logging functionality with thread locks.
"""
import csv
import threading
from pathlib import Path
from typing import List, Any

# Improvement: Use a lock to ensure thread-safe writes to the CSV from multiple API requests
_log_lock = threading.Lock()

def ensure_logfile(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
         "timestamp", "query_id", "retrieval_mode", "top_k_evidence", "latency_ms",
         "Precision@5", "Recall@10",
         "evidence_ids_returned", "gold_evidence_ids",
         "faithfulness_pass", "missing_evidence_behavior"
    ]
    
    write_header = False
    with _log_lock:
        if not p.exists():
            write_header = True
        else:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    write_header = True
            except Exception:
                write_header = True

        if write_header:
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

def log_query_metrics(log_file: str, row: List[Any]):
    ensure_logfile(log_file)
    with _log_lock:
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
