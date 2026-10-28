"""
Base interfaces for retrievers in RAG Detective.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_detective.models import SearchResult


class BaseRetriever(ABC):
    """Abstract interface for all search and retrieval engines."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Retrieves the top_k most relevant chunks for a given query.

        Args:
            query: The search query string.
            top_k: Number of candidates to return.

        Returns:
            List of SearchResult objects sorted by descending relevance.
        """
        pass
