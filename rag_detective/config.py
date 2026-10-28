"""
Configuration management for the RAG Detective backend.
Provides type-safe, environment-variable-backed settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    """Loads environment variables from a project-root .env file if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


@dataclass
class RAGDetectiveConfig:
    """
    Centralized configuration for the RAG Detective pipeline and backend service.
    Values can be overridden by environment variables.
    """

    # --- Embedding Configuration ---
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )
    embedding_normalize: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_NORMALIZE", "true").lower() in ("true", "1", "yes")
    )

    # --- Vector Database (Chroma) Configuration ---
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "collection0")
    )
    chroma_persist_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR") or None
    )
    chroma_distance_metric: str = field(
        default_factory=lambda: os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
    )

    # --- Google GenAI (Gemini) Configuration ---
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    )

    # --- Query Rewriter Configuration ---
    rewriter_temperature: float = field(
        default_factory=lambda: float(os.getenv("REWRITER_TEMPERATURE", "0.0"))
    )
    rewriter_top_k: int = field(
        default_factory=lambda: int(os.getenv("REWRITER_TOP_K", "50"))
    )

    # --- Sparse Retriever (BM25) Parameters ---
    bm25_k1: float = field(
        default_factory=lambda: float(os.getenv("BM25_K1", "1.2"))
    )
    bm25_b: float = field(
        default_factory=lambda: float(os.getenv("BM25_B", "0.75"))
    )

    # --- Rank Fusion (Weighted RRF) Parameters ---
    rrf_alpha: float = field(
        default_factory=lambda: float(os.getenv("RRF_ALPHA", "0.7"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )

    # --- Retrieval Defaults ---
    retrieval_dense_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_DENSE_K", "3"))
    )
    retrieval_sparse_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_SPARSE_K", "3"))
    )
    final_top_k: int = field(
        default_factory=lambda: int(os.getenv("FINAL_TOP_K", "3"))
    )

    # --- Detective Synthesizer Configuration ---
    synthesizer_temperature: float = field(
        default_factory=lambda: float(os.getenv("SYNTHESIZER_TEMPERATURE", "0.3"))
    )
    synthesizer_top_k: int = field(
        default_factory=lambda: int(os.getenv("SYNTHESIZER_TOP_K", "50"))
    )

    # --- Markdown Header Splitting Defaults ---
    headers_to_split_on: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
    )

    # --- Session Isolation & Rate Limiting Configuration ---
    session_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "1800"))
    )
    max_active_sessions: int = field(
        default_factory=lambda: int(os.getenv("MAX_ACTIVE_SESSIONS", "100"))
    )
    rate_limit_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "300"))
    )
    rate_limit_ingest_max: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_INGEST_MAX", "15"))
    )
    rate_limit_query_max: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_QUERY_MAX", "30"))
    )

    # --- Upload & Resource Limits (DoS Protection) ---
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))
    )
    max_chunks_per_doc: int = field(
        default_factory=lambda: int(os.getenv("MAX_CHUNKS_PER_DOC", "1500"))
    )


# Global default configuration instance
default_config = RAGDetectiveConfig()
