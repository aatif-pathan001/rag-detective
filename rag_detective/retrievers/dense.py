"""
Dense Semantic Retriever leveraging Chroma Vector Store.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from rag_detective.config import default_config
from rag_detective.exceptions import RetrievalError
from rag_detective.indexing.vector_store import ChromaVectorStore
from rag_detective.models import SearchResult
from rag_detective.retrievers.base import BaseRetriever

logger = logging.getLogger("uvicorn.error")


class DenseRetriever(BaseRetriever):
    """
    Semantic vector similarity search retriever.
    Wraps Chroma vector store and normalizes similarity scores.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        default_k: Optional[int] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.default_k = default_k or default_config.retrieval_dense_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Executes dense semantic similarity search against indexed vector embeddings.

        Args:
            query: The semantic natural language search query.
            top_k: Number of matches to return (defaults to configured dense k).

        Returns:
            List of SearchResult objects ranked by cosine similarity.
        """
        k = top_k or self.default_k
        if not query or not query.strip():
            logger.warning("Empty query passed to DenseRetriever.")
            return []

        try:
            results_with_score = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error("Dense retrieval failed for query '%s': %s", query, e)
            raise RetrievalError(f"Dense vector retrieval failed: {e}") from e

        search_results: List[SearchResult] = []
        for rank, (doc, dist) in enumerate(results_with_score, start=1):
            chunk_id = getattr(doc, "id", None) or doc.metadata.get("id", f"doc_{rank}")
            # Convert cosine distance to a similarity score (1 - distance)
            similarity = max(0.0, 1.0 - float(dist))

            search_results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=doc.page_content,
                    metadata=dict(doc.metadata or {}),
                    score=round(similarity, 4),
                    rank=rank,
                    engine="dense",
                )
            )

        logger.info("DenseRetriever retrieved %d chunks for query: '%s'", len(search_results), query[:50])
        return search_results

    def invoke(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """LangChain style alias for retrieve()."""
        return self.retrieve(query, top_k=top_k)
