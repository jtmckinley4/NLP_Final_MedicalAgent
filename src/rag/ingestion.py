"""Ingestion pipeline for building the persistent medical Chroma DB."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.rag.config import (
    BATCH_SIZE,
    DEFAULT_DATA_PATH,
    MAX_WORDS,
    OVERLAP_WORDS,
)
from src.rag.vector_store import get_collection, get_model


def clean_text(text: object) -> str:
    """Normalize whitespace and handle missing values."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def sentence_split(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
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


def make_id(text:str) -> str:
    """Stable unique ID for a piece of text."""
    return hashlib.md5(text.encode()).hexdigest()


def load_and_clean_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load, normalize, and deduplicate source Q/A rows."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "answer"])
    print("Cleaned dataset size:", len(df))
    return df


def build_records(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert cleaned rows into chunked vector records."""
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


def index_records(records: list[dict[str, object]], batch_size: int = BATCH_SIZE) -> None:
    """Embed and upsert records into Chroma in batches."""
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


def build_vector_db(force: bool = False) -> None:
    """
    Build the persistent vector DB from source CSV.

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


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion execution."""
    parser = argparse.ArgumentParser(
        description="Build the persistent ChromaDB index from medical raw data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the vector collection already contains records.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for ingestion pipeline."""
    args = _parse_args()
    build_vector_db(force=args.force)


if __name__ == "__main__":
    main()
