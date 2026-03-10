# Related Work Reproduction

This document tracks reproduction efforts for various relevant literature, techniques, and benchmarks that we are applying to our own RAG pipeline.

## 1. eRAG (Evaluative RAG)
- **Status**: Scaffolding complete
- **Location**: `related_work/erag/`
- **Description**: We are setting up a golden test set (`gold_erag.json`) to programmatically verify that our native retriever (`rag/retriever.py`) successfully isolates required facts ("concepts") for a given query, allowing an automated evaluation score before generation.

More techniques will be added here as we continue to benchmark and test our system against other standards.
