# MedicalAgent: A Multi-Agent Medical Question Answering System

A multi-agent medical question answering system using retrieval-augmented generation (RAG), ChromaDB, and PydanticAI. Built for CAP-6640.

## Overview

MedicalAgent is a domain-specific medical question answering system that uses Retrieval-Augmented Generation (RAG) to provide grounded, citation-based responses. The system retrieves relevant medical information from a structured knowledge base and supports downstream reasoning through a multi-agent architecture.

This project was developed for CAP-6640: Computational Understanding of Natural Language at the University of Central Florida.

---

## Problem Motivation

Medical question answering requires:
- Accurate information retrieval
- Reliable, citable sources
- Explainable, safe responses

Traditional LLMs may hallucinate or provide unsupported answers. MediRAG addresses this by grounding every response in retrieved medical evidence and validating it with NLP-based checks before delivery.

---

## Architecture Overview

The application runs a four-stage orchestration pipeline:

1. **Router Agent** — classifies the question and assigns a risk level
2. **Specialist Agent** — generates a draft answer using retrieved evidence:
   - General medical path (symptoms, conditions, prevention)
   - Medication path (side effects, interactions, warnings via OpenFDA)
3. **Safety Agent** — validates and revises the answer before delivery
4. **Memory Helper** — updates short-term conversation context for follow-ups

```
User Query → Router → Specialist → Safety Agent → Final Answer
                                        ↑
                               Evidence Validation
                           (TF-IDF + NER + Citation check)
```

Key code locations:

| Path | Description |
|------|-------------|
| `src/agents/` | Router, general specialist, medication specialist, safety agent |
| `src/tools/` | RAG retrieval, OpenFDA drug lookup, evidence validation, memory |
| `src/models/schemas.py` | Shared Pydantic schemas |
| `src/orchestration/pipeline.py` | End-to-end turn logic |
| `src/rag/` | Chunking, embedding, and ChromaDB ingestion |
| `src/main.py` | CLI entrypoint |
| `src/evals/` | Evaluation suite (deterministic, behavioral, judge) |
| `MedicalAgent_Project_Notebook.ipynb` | Full project walkthrough notebook |

---

## Environment Variables

- `CAP6640_API_KEY` — required for live LLM calls via course proxy

Model provider wiring in `src/config.py`:

- Proxy URL: `https://litellm.6640.ucf.spencerlyon.com`
- Router / Safety / Memory model: `claude-haiku-4-5`
- Specialist model: `claude-sonnet-4-6`

---

## Run CLI

Run from the repository root:

1. `uv sync`
2. Build the RAG vector database (only needed once):
   ```bash
   uv run python -m src.rag.ingestion
   # Force rebuild: uv run python -m src.rag.ingestion --force
   ```
3. Start the multi-agent application:
   ```bash
   uv run python -m src.main
   ```

> `CAP6640_API_KEY` must be set in a `.env` file at the project root.

Exit with `exit` or `quit`.

**Printed fields per turn:**
- Route and risk level
- Direct answer
- Evidence summary
- Citations
- Safety note
- Confidence score
- Follow-up question (if applicable)

---

## Dataset

**MedQuAD** — Medical Question Answer Dataset:
- 47,457 question-answer pairs
- Sourced from 12 NIH websites (MedlinePlus, CancerGov, NIHSeniorHealth, and more)
- Covers symptoms, causes, treatments, prevention, and medications

---

## RAG Pipeline Details

### Preprocessing
- Remove missing and duplicate entries
- Normalize whitespace and text formatting

### Chunking Strategy
- Sentence-aware splitting (no mid-sentence cuts)
- ~220-word chunks with 40-word overlap to preserve cross-boundary context
- Stable MD5-based chunk IDs for idempotent ingestion

### Embeddings
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- 384-dimensional dense vector representations

### Vector Database
- ChromaDB with persistent storage (`chroma_db/`)
- Cosine similarity via HNSW index

### Retrieval
- User query is embedded at query time
- Top-k similar chunks retrieved with metadata (source, question, focus area)
- Metadata used for citations in the final answer

---

## Evidence Validation

The Safety Agent runs a three-part grounding check on every draft answer before it reaches the user:

| Check | Method | Catches |
|-------|--------|---------|
| Citation integrity | ID set membership | Hallucinated citation references |
| TF-IDF grounding | Cosine similarity per sentence | Unsupported factual claims |
| NER entity check | spaCy `en_core_web_sm` | Drug/condition names not in evidence |

An answer is only approved if all three checks pass.

---

## Evaluation Pipeline

A modular evaluation suite assesses the agent across three dimensions:

- **Deterministic Evaluator** — structural checks (confidence in [0,1], citations present, citation IDs valid, escalation language on high-risk answers)
- **Behavioral Evaluator** — routing correctness, tool usage rules, memory usage
- **Judge Evaluator (Rule-Based)** — weighted heuristic scoring on answer completeness, evidence summary, safety note, and confidence

Run the full suite:
```bash
uv run python -m src.evals.run_evals
```

Results are written to `src/evals/results.csv` (12 test cases across all routing paths, risk levels, and multi-turn memory scenarios).

---

## Project Notebook

`MedicalAgent_Project_Notebook.ipynb` contains a full end-to-end walkthrough of the system: dataset exploration, preprocessing, chunking, embedding, RAG retrieval, evidence validation, multi-agent architecture, evaluation, and results analysis.

---

## Example Usage

```python
# RAG retrieval
results = search_medical_kb("What are the symptoms of asthma?", top_k=3)
for r in results:
    print(r["text"])
```

```python
# Full pipeline (requires CAP6640_API_KEY)
from src.orchestration.pipeline import run_turn

result = await run_turn("What are the symptoms of anemia?", memory=None)
print(result.answer)
```
