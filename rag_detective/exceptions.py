"""
Custom exception hierarchy for the RAG Detective framework.
"""

from __future__ import annotations

from typing import Optional


class RAGDetectiveError(Exception):
    """Base class for all exceptions raised by RAG Detective."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(RAGDetectiveError):
    """Raised when document ingestion or loading fails."""
    pass


class DocumentParseError(IngestionError):
    """Raised when a specific document cannot be converted or parsed."""
    pass


class IndexingError(RAGDetectiveError):
    """Raised when vector indexing or embedding generation fails."""
    pass


class RetrievalError(RAGDetectiveError):
    """Raised when dense, sparse, or hybrid retrieval fails."""
    pass


class QueryRewritingError(RAGDetectiveError):
    """Raised when LLM query rewriting fails or produces invalid JSON."""
    pass


class FusionError(RAGDetectiveError):
    """Raised during reciprocal rank fusion calculation."""
    pass


class SynthesisError(RAGDetectiveError):
    """Raised during the detective synthesis stage."""
    pass


class ConfigurationError(RAGDetectiveError):
    """Raised when configuration parameters or credentials are invalid."""
    pass
