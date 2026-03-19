# Lab 8: Fine-Tuning and Domain Adaptation for GenAI Systems — Group Report

## Project Title
**PolicyPulse: Domain-Adapted Legislative AI Assistant**

## Team Members
| Student Name | Contribution | Percentage |
|---|---|---|
| Student A (Person 1) | Snowflake pipeline, data ingestion, retrieval system, schema design | 35% |
| Student B (Person 2) | Instruction dataset creation, LoRA fine-tuning, model evaluation | 35% |
| Student C (Person 3) | Backend + Streamlit integration, agent updates, evaluation pipeline, demo | 30% |

---

## 1. Domain Task Definition

PolicyPulse is a legislative intelligence system that helps users analyze U.S. legislative bills. The domain task is **legislative question answering and bill analysis**, including:

- **Bill summarization** — Condensing bill descriptions into concise summaries
- **Status inquiry** — Reporting current legislative stage and committee assignments
- **Bill comparison** — Identifying similarities and differences between related legislation
- **Provision extraction** — Identifying key provisions from bill text
- **Missing evidence detection** — Correctly handling queries that cannot be answered from available data
- **Stakeholder analysis** — Identifying parties affected by legislation
- **Impact assessment** — Analyzing potential policy impacts

The system retrieves real legislative bill records from a Snowflake data warehouse (`POLICYPULSE_DB.PUBLIC.BILLS`) and uses a domain-adapted language model to generate grounded, accurate responses.

---

## 2. Instruction Dataset Description

We created a comprehensive instruction-tuning dataset of **52 examples** covering the legislative domain, stored in `data/instruction_dataset/dataset.jsonl`.

**Format:** Each example follows the standard instruction-tuning structure:
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**Task distribution:**
| Task Type | Count | Description |
|---|---|---|
| Bill Summarization | 8 | Condense bill details into brief summaries |
| Status Inquiry | 5 | Report legislative status and progress |
| Bill Comparison | 5 | Compare two bills side-by-side |
| Key Provisions | 4 | Extract and list main bill provisions |
| Missing Evidence | 5 | Handle queries with insufficient context |
| Impact Analysis | 4 | Assess potential effects of legislation |
| Committee Analysis | 3 | Explain committee jurisdictions |
| Metadata Extraction | 3 | Extract structured data from descriptions |
| Stakeholder Analysis | 3 | Identify affected parties |
| Legislative Trajectory | 3 | Analyze bill advancement likelihood |
| Other (drafting questions, classification, fiscal analysis) | 9 | Various specialized tasks |

**Data splits:**
- Train: 36 examples (70%)
- Validation: 8 examples (15%)
- Test: 8 examples (15%)

---

## 3. Adaptation Method

### Method: LoRA (Low-Rank Adaptation)

We applied **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** to adapt a pre-trained language model to the legislative domain.

**Base model:** `google/flan-t5-base` (250M parameters, sequence-to-sequence architecture)

**LoRA configuration:**
| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | Query (q) and Value (v) attention projections |
| Task type | SEQ_2_SEQ_LM |
| Trainable parameters | ~0.5% of total model parameters |

**Training configuration:**
| Setting | Value |
|---|---|
| Epochs | 4 |
| Learning rate | 3e-4 |
| Batch size | 4 |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup (10% of steps) |
| Max source length | 512 tokens |
| Max target length | 256 tokens |
| Gradient clipping | 1.0 |

**Why LoRA:** LoRA modifies only a small number of parameters via low-rank decomposition of weight update matrices, making it feasible to fine-tune on a small dataset without catastrophic forgetting. With only ~0.5% of parameters trainable, it is computationally efficient and preserves the base model's general language understanding while specializing it for legislative text.

**Training script:** `training/train_lora.py`  
**Saved artifacts:** `training/artifacts/best_adapter/` and `training/artifacts/final_adapter/`

---

## 4. System Integration Description

### Architecture

```
┌────────────────────────┐    HTTPS/JSON    ┌──────────────────────────┐
│  Streamlit Frontend    │ ──────────────► │  FastAPI Backend          │
│  (app/main.py)         │ ◄────────────── │  (api/server.py)          │
│  - Model mode toggle   │                 │  - /query endpoint        │
│  - Agent chat UI       │                 │  - /agent_query endpoint  │
│  - Legislative queries │                 └──────────┬───────────────┘
└────────────────────────┘                            │
                                                      ▼
                        ┌──────────────────────────────────────────┐
                        │            Core Components               │
                        │                                          │
                        │  ┌─────────────────────┐                 │
                        │  │ RAG Pipeline         │                 │
                        │  │ (rag/pipeline.py)    │ ◄── Snowflake  │
                        │  │ + snowflake_retriever│     BILLS DB   │
                        │  └─────────────────────┘                 │
                        │                                          │
                        │  ┌─────────────────────┐                 │
                        │  │ Domain Model Client  │                 │
                        │  │ (llm/domain_model_   │ ◄── LoRA       │
                        │  │  client.py)          │     Adapter    │
                        │  └─────────────────────┘                 │
                        │                                          │
                        │  ┌─────────────────────┐                 │
                        │  │ Agent Runner         │                 │
                        │  │ (agent/runner.py)    │ ◄── Gemini +   │
                        │  │ + tool_registry      │     Tools      │
                        │  └─────────────────────┘                 │
                        └──────────────────────────────────────────┘
```

### Integration Points

1. **Domain Model Client** (`llm/domain_model_client.py`):
   - Provides `set_mode("baseline" | "adapted")` to switch between models
   - `generate(instruction, context)` for unified inference
   - `generate_both()` for side-by-side comparison

2. **Agent System Prompt** (`agent/prompts.py`):
   - Updated to reflect the legislative domain ("PolicyPulse AI")
   - Enforces grounding rules (must call tools before answering)
   - Prohibits hallucination of bill details
   - Specifies Snowflake data source awareness

3. **Streamlit UI** (`app/main.py`):
   - Title updated to "PolicyPulse — Legislative AI Assistant"
   - Example queries now reference legislative topics (Q1–Q5)
   - Model mode selector (Adapted/Baseline) in sidebar
   - Agent mode chat with legislative domain awareness

4. **Snowflake Integration**:
   - Schema script: `scripts/create_snowflake_schema.sql`
   - Data loader: `scripts/load_to_snowflake.py`
   - Retriever: `rag/snowflake_retriever.py`

---

## 5. Evaluation Results

### Evaluation Pipeline

We built a dedicated evaluation pipeline (`eval/run_eval.py`) that runs **10 benchmark questions** (`eval/benchmark_questions.json`) through both the baseline and adapted models, measuring:

- **ROUGE-L** — Recall-oriented longest common subsequence overlap
- **BLEU** — Precision-oriented n-gram overlap
- **Missing-evidence detection** — Correct handling of unanswerable queries
- **Generation latency** — End-to-end inference time

### Benchmark Question Categories
| Category | Count |
|---|---|
| Summarization | 1 |
| Status Inquiry | 1 |
| Bill Comparison | 1 |
| Missing Evidence | 1 |
| Provision Extraction | 1 |
| Impact Analysis | 1 |
| Trajectory Analysis | 1 |
| Committee Analysis | 1 |
| Metadata Extraction | 1 |
| Stakeholder Analysis | 1 |

### Expected Improvement Areas

The adapted model is expected to outperform the baseline in:
1. **Legislative terminology** — Correctly using terms like "markup," "engrossed," "preclearance"
2. **Structured output** — Formatting bill summaries with consistent metadata fields
3. **Missing evidence detection** — More reliable identification of unanswerable queries
4. **Domain-specific reasoning** — Understanding committee jurisdiction, legislative stages

---

## 6. Impact on Project Performance

### Before (Lab 4–6 baseline)
- PDF-based TF-IDF retrieval with local documents
- Generic extractive answers (no LLM generation)
- Paper-specific domain (SQLENS, FACT, ALIGNRAG)
- Generic agent system prompt

### After (Lab 8 domain-adapted)
- Snowflake-backed retrieval from live legislative database
- LoRA-adapted model for legislative Q&A generation
- Legislative domain (U.S. bills, committees, policy analysis)
- Domain-specialized agent prompt with grounding rules
- Formal evaluation pipeline with before/after comparison
- Instruction dataset covering 10+ legislative task types

### Key Improvements
1. **Domain specialization** — System is now a legislative analysis tool rather than a generic RAG chatbot
2. **Data quality** — Snowflake provides structured, queryable bill records vs. static PDFs
3. **Model adaptation** — LoRA fine-tuning teaches the model legislative conventions and output formats
4. **Evaluation rigor** — Dedicated benchmark with quantitative metrics (ROUGE-L, BLEU)
5. **Production readiness** — Switchable baseline/adapted mode enables A/B testing

---

## GitHub Repository

**Repository:** https://github.com/kbrh3/lab45542

### Key Files Added in Lab 8
| Path | Description |
|---|---|
| `data/instruction_dataset/dataset.jsonl` | 52-example instruction tuning dataset |
| `data/instruction_dataset/prepare_splits.py` | Train/val/test split script |
| `training/train_lora.py` | LoRA fine-tuning script (PEFT) |
| `training/artifacts/` | Saved adapter weights and training logs |
| `llm/domain_model_client.py` | Baseline vs. adapted model switching |
| `eval/run_eval.py` | Evaluation pipeline |
| `eval/benchmark_questions.json` | 10 benchmark legislative questions |
| `scripts/create_snowflake_schema.sql` | Snowflake schema DDL |
| `scripts/load_to_snowflake.py` | CSV-to-Snowflake data loader |
| `agent/prompts.py` | Updated domain-specific agent prompt |
| `app/main.py` | Updated UI with legislative branding |
