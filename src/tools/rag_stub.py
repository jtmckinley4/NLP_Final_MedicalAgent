"""
rag.py
Medical RAG component built on MedQuAD + ChromaDB + SentenceTransformers.

Usage from another script:
    from rag import search_medical_kb
    results = search_medical_kb("What are symptoms of diabetes?", top_k=5)

Run this file directly once to build the persistent vector DB:
    python rag.py
After that, the DB is reused from ./chroma_db on subsequent imports.
"""

import os
import re
import hashlib

import pandas as pd
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(PROJECT_ROOT, "medquad.csv")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "medical_qa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_WORDS = 220
OVERLAP_WORDS = 40
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text):
    """Normalize whitespace and handle missing values."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def sentence_split(text):
    """Split text into sentences on terminal punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS):
    """Split a long answer into overlapping word-bounded chunks."""
    sentences = sentence_split(text)
    chunks = []
    current = []

    for sent in sentences:
        current_words = " ".join(current).split()
        sent_words = sent.split()

        if len(current_words) + len(sent_words) <= max_words:
            current.append(sent)
        else:
            if current:
                chunks.append(" ".join(current))
            overlap = " ".join(current_words[-overlap_words:])
            current = [overlap, sent] if overlap else [sent]

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def make_id(text):
    """Stable unique ID for a piece of text."""
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Lazy singletons (so importing this module is cheap)
# ---------------------------------------------------------------------------

_model = None
_collection = None


def get_model():
    """Load the embedding model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    """Open (or create) the persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Index build (only needed once; persists to disk)
# ---------------------------------------------------------------------------

def load_and_clean_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "answer"])
    print("Cleaned dataset size:", len(df))
    return df


def build_records(df):
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        question = row["question"]
        answer = row["answer"]
        source_id = make_id(question + answer)

        for j, chunk in enumerate(chunk_text(answer)):
            records.append({
                "id": f"{source_id}_{j}",
                "document": f"Question: {question}\nAnswer: {chunk}",
                "metadata": {
                    "question": question,
                    "source": str(row.get("source", "")),
                    "question_type": str(row.get("question_type", "")),
                },
            })
    print("Total chunk records:", len(records))
    return records


def index_records(records, batch_size=BATCH_SIZE):
    model = get_model()
    collection = get_collection()

    for i in tqdm(range(0, len(records), batch_size), desc="Embedding"):
        batch = records[i:i + batch_size]
        docs = [r["document"] for r in batch]
        ids = [r["id"] for r in batch]
        metas = [r["metadata"] for r in batch]
        embeddings = model.encode(docs).tolist()

        collection.upsert(
            documents=docs,
            ids=ids,
            metadatas=metas,
            embeddings=embeddings,
        )


def build_vector_db(force=False):
    """
    Build the persistent vector DB from medquad.csv.
    Skips the build if the collection already has data, unless force=True.
    """
    collection = get_collection()
    if not force and collection.count() > 0:
        print(f"Vector DB already populated ({collection.count()} chunks). "
              "Pass force=True to rebuild.")
        return

    df = load_and_clean_dataset()
    records = build_records(df)
    index_records(records)
    print("Vector DB built!")


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------

def search_medical_kb(query, top_k=5):
    """
    Search the medical knowledge base for the most relevant chunks.

    Args:
        query: The user's medical question.
        top_k: Number of retrieved chunks to return.

    Returns:
        A list of dicts: {text, question, source, score}.
        Higher score = better match (range ~0-1).
    """
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = float(1 / (1 + distance))
        output.append({
            "text": results["documents"][0][i],
            "question": results["metadatas"][0][i].get("question", ""),
            "source": results["metadatas"][0][i].get("source", ""),
            "score": score,
        })
    return output


# ---------------------------------------------------------------------------
# Run as a script: build the DB, then sanity-check retrieval
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_vector_db()

    sample_queries = [
        "What are symptoms of diabetes?",
        "How is asthma treated?",
        "What causes high blood pressure?",
        "What are side effects of ibuprofen?",
        "How do you prevent heart disease?",
    ]

    for q in sample_queries:
        print("\n\nQUERY:", q)
        for r in search_medical_kb(q, top_k=3):
            print("- Score:", round(r["score"], 3))
            print("  Question:", r["question"])
            print("  Text:", r["text"][:200])