"""Shared configuration for the RAG package."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "medquad.csv"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "medical_qa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_WORDS = 220
OVERLAP_WORDS = 40
BATCH_SIZE = 64
