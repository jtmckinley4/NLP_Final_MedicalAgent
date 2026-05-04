# NLP_Final_MedicalAgent
A multi-agent medical question answering system using retrieval-augmented generation (RAG), ChromaDB, and PydanticAI. Built for CAP-6640.
# MediRAG: Medical Question Answering System

## Overview
MedicalAgent is a domain-specific medical question answering system that uses Retrieval-Augmented Generation (RAG) to provide grounded, citation-based responses. The system retrieves relevant medical information from a structured knowledge base and supports downstream reasoning through a multi-agent architecture.

This project was developed for CAP-6640: Computational Understanding of Natural Language at the University of Central Florida.

---

## Problem Motivation
Medical question answering requires:
- accurate information retrieval
- reliable sources
- explainable responses

Traditional LLMs may hallucinate or provide unsupported answers.  
MediRAG addresses this by grounding responses in retrieved medical evidence.

---

## Architecture Overview

The application runs a lightweight orchestration pipeline:

1. Router agent classifies the question.
2. Specialist agent generates a draft answer:
   - general medical path, or
   - medication path.
3. Safety agent validates/revises the answer.
4. Memory helper updates short conversation context.
5. CLI prints a structured response.

Key code locations:

- `src/agents/` - router, general specialist, medication specialist, safety verifier
- `src/tools/` - retrieval stub, drug lookup stub, evidence validation, memory helpers
- `src/models/schemas.py` - shared Pydantic schemas
- `src/orchestration/pipeline.py` - end-to-end turn logic
- `src/main.py` - CLI entrypoint
- `src/evals/placeholder_evals.py` - evaluation TODO placeholders


## Environment Variables

- `CAP6640_API_KEY` - required for live LLM calls via course proxy

Model provider wiring in `src/config.py` uses:

- Proxy URL: `https://litellm.6640.ucf.spencerlyon.com`
- API key env variable: `CAP6640_API_KEY`

## Run CLI

Run from repository root:

1. Run `uv sync`
2. Build the RAG vector database first (only needed once):
   - `uv run python -m src.rag.ingestion`
   - Optional rebuild: `uv run python -m src.rag.ingestion --force`
1. Start the multi-agent application:
   - `uv run python -m src.main`

Note: live model calls are always enabled, so `CAP6640_API_KEY` must be set in `.env`.

Exit commands:

- `exit`
- `quit`

Printed fields:

- Direct answer
- Evidence summary
- Citations
- Safety note
- Confidence
- Follow-up question (if present)

---

## Dataset

We use the MedQuAD dataset:
- 47,457 medical question-answer pairs
- Sourced from 12 NIH websites
- Covers multiple medical domains and question types

---

## RAG Pipeline Details

### Preprocessing
- Removed missing and duplicate entries
- Standardized text formatting

### Chunking Strategy
- Sentence-aware splitting
- 150–300 word chunks
- Overlapping context (40 words)
- Preserves semantic meaning for medical explanations

### Embeddings
- Model: `all-MiniLM-L6-v2`
- Converts text into dense vector representations

### Vector Database
- ChromaDB used for storage and retrieval
- Cosine similarity for semantic search

### Retrieval
- User query is embedded
- Top-k similar chunks retrieved
- Metadata included for citation and reasoning

---

## Evaluation Pipeline Overview

This project includes a modular evaluation suite designed to assess the agent’s
performance across deterministic, behavioral, and rule‑based criteria.

- Deterministic Evaluator
  Validates structural requirements such as presence of evidence summaries,
  safety notes, and confidence scores.

- Behavioral Evaluator
  Ensures the agent follows routing rules, avoids unsafe tool usage, and
  maintains consistent behavior across test cases.

- Judge Evaluator (Rule‑Based)
  Scores the agent’s output using a weighted heuristic based on answer
  completeness, safety, and reasoning quality.

The evaluation suite runs automatically using:
`python -m src.evals.run_evals`
and outputs a consolidated `results.csv` file for analysis.



## Example Usage

```python
results = search_medical_kb("What are the symptoms of asthma?", top_k=3)

for r in results:
    print(r["text"])
