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

## System Architecture

The system consists of three main components:

### 1. RAG Pipeline 
- Data ingestion from MedQuAD dataset
- Text cleaning and preprocessing
- Sentence-aware chunking (150–300 words with overlap)
- Embedding generation using SentenceTransformers
- Storage in ChromaDB vector database
- Semantic retrieval of relevant chunks

### 2. Multi-Agent System 
- Triage Agent (classifies query type and urgency)
- Retrieval Agent (calls vector database)
- Answer Agent (generates structured response)
- Safety Agent (verifies medical reliability)

### 3. Evaluation Framework
- Deterministic checks (citations, formatting)
- LLM-based evaluation (faithfulness, clarity)
- Behavioral evaluation (tool usage and reasoning flow)

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

## Example Usage

```python
results = search_medical_kb("What are the symptoms of asthma?", top_k=3)

for r in results:
    print(r["text"])
